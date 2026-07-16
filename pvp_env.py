"""Reinforcement-learning environment for one PvP fighter.

The mod connects in as a socket client (same protocol as ai_server); this class is
the server side. Each tick the RL policy chooses a COMPOSITE action:
  - a movement primitive (spacing / knockback management, incl. W-tap by choosing a
    no-sprint primitive after a hit),
  - whether to swing (attack),
  - whether to sword-block (blockhit),
  - a small aim residual on top of the computed angle.
Only the base aim geometry and the periodic crit-jump stay computed (CombatController.
act_policy). So the agent learns spacing AND its own attack/block/W-tap timing and can
refine aim - the parts where a learned policy can beat hardcoded rules.

Self-play = two of these on two ports, stepped in lockstep by the trainer. Kept to a
single fighter here so it stays composable.

Throughput note: step() blocks on the next game tick (~50 ms), so this runs at real
time, 20 steps/sec. That's the wall. Warm-starting the policy from the BC weights
means we're fine-tuning, not learning from scratch, which is what makes an overnight
run worthwhile instead of hopeless.
"""
import socket
import json
import math
from collections import deque

import numpy as np

from features import frame_features, STACK, FRAME_DIM, INPUT_DIM
from combat import CombatController, AIM_RES_MAX

# Discrete movement vocabulary. Sprint only engages moving forward in 1.8, so the
# back/strafe actions don't bother with it. This is deliberately small - a compact
# action space is what makes movement-only RL cheap.
#         w      a      s      d      sprint
ACTIONS = [
    (True,  False, False, False, True),   # 0 sprint straight in
    (True,  True,  False, False, True),   # 1 sprint circle-left
    (True,  False, False, True,  True),   # 2 sprint circle-right
    (True,  False, False, False, False),  # 3 walk in (precise spacing / W-tap)
    (False, True,  False, False, False),  # 4 strafe left
    (False, False, False, True,  False),  # 5 strafe right
    (False, True,  True,  False, False),  # 6 retreat left
    (False, False, True,  True,  False),  # 7 retreat right
]
N_ACTIONS = len(ACTIONS)

# Composite action space the policy chooses from each tick:
#   move  : Discrete(N_ACTIONS)     - which movement primitive
#   click : Discrete(2)             - swing or not (tick rate is the only CPS cap)
#   block : Discrete(2)             - hold sword-block or not
#   aim   : Box(2)                  - yaw/pitch residual (deg), clamped to +-AIM_RES_MAX
MOVE_DIM = N_ACTIONS
CLICK_DIM = 2
BLOCK_DIM = 2
AIM_DIM = 2

KILL_REWARD = 10.0
DEATH_REWARD = -10.0
TIME_PENALTY = 0.01   # per tick, so stalling forever is never optimal
MAX_HEALTH = 20.0

# --- dense shaping ----------------------------------------------------------
# All of these are small next to a landed sword hit (~6 health units), so they steer
# style without ever beating "actually win the fight".
#
# Range: potential-based shaping toward the sweet spot just inside sword reach -
# reward for moving toward IDEAL_RANGE, equal charge for drifting off it (too far OR
# face-hugging). Potential-based => telescopes to ~0 over any closed loop, so it can't
# be farmed by oscillating. Replaces the old pure-approach term, which paid all the
# way down to distance 0 and taught face-hugging.
IDEAL_RANGE = 2.8
RANGE_COEF = 0.05
# Blockhit spam: holding sword-block cuts movement to a crawl. But block reduction
# is decided SERVER-side - a reactive block at the tick you're hit registers too
# late and reduces nothing, so the only blocking that works is PROACTIVE, held
# through the exchange window. Taxing every block tick taught "never block" and
# collapsed the run (blk 38%->1%, dmg +180/-300 vs still-blocking snapshots). So:
# blocking inside exchange range is FREE (the game's own mobility tradeoff decides);
# only far-range block spam - which kills approach momentum for zero possible
# benefit - pays the cost.
BLOCK_COST = 0.02
BLOCK_FREE_RANGE = 4.0   # blocks; within this, blocking costs nothing extra
# First blood decides momentum: whoever lands first controls the combo. Paid once
# per round, signed (bonus if we land it, penalty if we take it).
FIRST_HIT_BONUS = 1.0
# Combos: each consecutive hit landed within COMBO_WINDOW ticks of the previous one
# pays an escalating bonus. This is the incentive to HOLD a combo (keep range, keep
# sprint reset) instead of trading 1-for-1.
COMBO_WINDOW = 25     # ticks (1.25 s) - proper follow-up cadence is ~10-15
COMBO_BONUS = 0.5     # per combo step, capped at 4 extra hits
# Sprint-reset (W-tap): a hit with fresh sprint launches the target much further,
# which is what makes the follow-up land. Measured directly - the target's horizontal
# speed on the tick AFTER our hit - so the policy is paid for real knockback, however
# it achieves it (that's exactly the W-tap skill).
KB_COEF = 0.4
KB_NORM = 0.6         # blocks/tick; ~sprint-hit launch speed, so bonus tops out ~KB_COEF
# Aim-residual regularizer: the aim head gets NO entropy bonus (a Gaussian's entropy is
# just log-std) and nothing else anchors it to 0, so it random-walks into one-sided,
# state-dependent offsets that fight the computed geometry - the crosshair orbits and
# the intermittent "spin". Tax the squared residual so a lead only survives if it's
# earning hits: at 0.002 a ~1deg correction is ~free (0.002/tick), a 3deg lead is cheap
# (0.036/tick), and pinning the full +-8deg is expensive (0.256/tick, ~a hit over 20
# ticks) - exactly the ordering we want.
AIM_RES_COEF = 0.004


class PvPEnv:
    def __init__(self, port=9999, reset_commands=None, max_ticks=1200, host='127.0.0.1'):
        self.port = port
        self.host = host
        # Chat commands run at the start of each episode (teleport/heal/regear).
        # Needs op on your own world. e.g. ["/tp SelfBot 0 64 0", "/effect SelfBot 6 1 255"]
        self.reset_commands = reset_commands or []
        self.max_ticks = max_ticks
        self.conn = None
        self._buffer = ""
        self.last_state = None   # freshest parsed state; also what _send acks
        self.history = deque([[0.0] * FRAME_DIM] * STACK, maxlen=STACK)
        self.combat = CombatController()

    # --- socket plumbing ---------------------------------------------------
    def connect(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        print(f"[env:{self.port}] waiting for Minecraft...")
        self.conn, _ = srv.accept()
        srv.close()
        print(f"[env:{self.port}] connected.")

    def _recv_state(self):
        # The mod streams one state per game tick whether or not we've replied, so
        # any time Python runs slower than 20Hz (PPO updates especially) frames pile
        # up in the socket buffer. Acting on the oldest queued frame means acting on
        # the past - stale round-end frames re-trigger "kill" and spam the reset
        # commands. Block for one complete line, then drain everything else already
        # buffered and use only the FRESHEST complete frame.
        while "\n" not in self._buffer:
            chunk = self.conn.recv(4096).decode('utf-8')
            if not chunk:
                raise ConnectionError("mod disconnected")
            self._buffer += chunk
        self.conn.setblocking(False)
        try:
            while True:
                chunk = self.conn.recv(4096)
                if not chunk:
                    raise ConnectionError("mod disconnected")
                self._buffer += chunk.decode('utf-8')
        except BlockingIOError:
            pass
        finally:
            self.conn.setblocking(True)
        lines = self._buffer.split("\n")
        self._buffer = lines[-1]  # trailing partial line, if any
        complete = [l for l in lines[:-1] if l.strip()]
        if not complete:
            return self._recv_state()
        return json.loads(complete[-1])

    def _send(self, action):
        # Echo the tick of the state this action was computed from (ack_tick). The
        # mod uses it to measure control-loop staleness - and, when the reply to the
        # previous tick hasn't landed yet, to wait a few ms for it instead of acting
        # on a stale action (the phase-lock that reads as the aim spinning).
        if self.last_state and "tick" in self.last_state:
            action = dict(action, ack_tick=self.last_state["tick"])
        self.conn.sendall((json.dumps(action) + "\n").encode('utf-8'))

    def _obs_vector(self):
        return np.array([v for f in self.history for v in f], dtype=np.float32)

    # --- gym-style API -----------------------------------------------------
    def reset(self):
        self.combat.reset()
        self.history = deque([[0.0] * FRAME_DIM] * STACK, maxlen=STACK)

        # Fire reset commands piggybacked on an idle action. If a fighter just died
        # the mod auto-respawns it, but a dead/mid-respawn player can't be teleported,
        # so the volley goes out TWICE: the first catches the survivor, the second
        # lands after the respawn. Then idle so the first real observation isn't a
        # mid-teleport frame.
        reset_msg = ({"command": "; ".join(self.reset_commands)}
                     if self.reset_commands else {})
        self._send(reset_msg)
        for _ in range(8):
            state = self._recv_state()
            self.last_state = state  # keep acks current through the idle
            self._send({})  # idle while the respawn settles
        self._send(reset_msg)
        for _ in range(10):
            state = self._recv_state()
            self.last_state = state
            self._send({})  # idle while the teleport settles

        self.prev_my_health = state.get("my_health", MAX_HEALTH)
        self.prev_tgt_health = state.get("tgt_health", MAX_HEALTH)
        self.ticks = 0
        # Per-round shaping state
        self.first_hit_done = False
        self.combo = 0
        self.ticks_since_my_hit = 999
        self.kb_check = False
        self.last_state = state
        self.history.append(frame_features(state))
        return self._obs_vector()

    def step_send(self, action):
        """Dispatch the action for this tick WITHOUT waiting for the result.

        Split out from step() so the self-play harness can send BOTH fighters'
        actions before blocking on either recv. If we send+recv one fighter fully
        before even sending the other, the second fighter's command lands a full
        tick later every step - an extra tick of control-loop delay that, at the
        same aim gain, turns a clean tracking loop into an overshooting/orbiting
        one (the crosshair 'rubber-bands' and never settles on the hitbox). Send
        both first, then recv both, so the two fighters share the same latency.
        """
        self._pending_action = action  # step_recv scores it (block cost etc.)
        w, a, s, d, sprint = ACTIONS[int(action["move"])]
        movement = {"w": w, "a": a, "s": s, "d": d, "sprint": sprint}
        self._send(self.combat.act_policy(
            self.last_state, movement,
            click=bool(action["click"]), block=bool(action["block"]),
            aim_res=action["aim"]))

    def step_recv(self):
        """Block for the freshest post-action frame and score it. Pairs with
        step_send() for the same tick."""
        action = self._pending_action
        state = self._recv_state()
        self.history.append(frame_features(state))
        self.ticks += 1

        my_h = state.get("my_health", self.prev_my_health)
        tgt_h = state.get("tgt_health", self.prev_tgt_health)
        # Only decreases count as damage (health rises on regen/reset)
        dealt = max(0.0, self.prev_tgt_health - tgt_h)
        taken = max(0.0, self.prev_my_health - my_h)
        reward = dealt - taken - TIME_PENALTY

        # Range: potential-based pull toward the sweet spot (see constants)
        prev_d = self.last_state.get("target_dist", -1.0)
        new_d = state.get("target_dist", -1.0)
        if prev_d is not None and new_d is not None and prev_d > 0.0 and new_d > 0.0:
            reward += RANGE_COEF * (abs(prev_d - IDEAL_RANGE) - abs(new_d - IDEAL_RANGE))

        # Block cost only OUT of exchange range (see constants: in-range blocking must
        # be free because server-side reduction requires proactive holding)
        if action["block"] and (new_d is None or new_d <= 0.0 or new_d > BLOCK_FREE_RANGE):
            reward -= BLOCK_COST

        # Aim-residual tax (see AIM_RES_COEF): penalize the squared residual the policy
        # actually emitted this tick, so the aim head can't drift into a free-floating
        # offset that spins the crosshair. Raw (pre-clamp) value, so the pushback keeps
        # growing even past +-AIM_RES_MAX instead of saturating with the applied clamp.
        res_y, res_p = float(action["aim"][0]), float(action["aim"][1])
        reward -= AIM_RES_COEF * (res_y * res_y + res_p * res_p)

        # First blood, once per round, signed
        if not self.first_hit_done and (dealt > 0.0 or taken > 0.0):
            self.first_hit_done = True
            reward += FIRST_HIT_BONUS if dealt > taken else -FIRST_HIT_BONUS

        # Sprint-reset: pay for the knockback our LAST hit actually produced (the
        # target's launch speed shows up in the frame after the health drop)
        if self.kb_check:
            kb = math.hypot(state.get("tgt_vx", 0.0) or 0.0, state.get("tgt_vz", 0.0) or 0.0)
            reward += KB_COEF * min(kb / KB_NORM, 1.0)
            self.kb_check = False

        # Combos: escalating bonus for consecutive CLEAN hits inside the window.
        # Taking ANY damage breaks the chain, so a 1-for-1 trade no longer pays a
        # combo bonus - only genuinely uncontested chains ("hits landed without
        # getting touched") do. This is the behaviour we actually want to price;
        # the old counter incremented through trades and mislabeled them as combos.
        self.ticks_since_my_hit += 1
        if taken > 0.0:
            self.combo = 0  # got touched -> chain broken, next hit starts fresh
        if dealt > 0.0:
            if self.combo > 0 and self.ticks_since_my_hit <= COMBO_WINDOW:
                self.combo += 1
            else:
                self.combo = 1
            if self.combo >= 2:
                reward += COMBO_BONUS * min(self.combo - 1, 4)
            self.ticks_since_my_hit = 0
            self.kb_check = True

        self.prev_my_health, self.prev_tgt_health = my_h, tgt_h

        done = False
        # combo = current CLEAN-hit chain length (0 between chains, reset by taken>0).
        # Surfaced so the trainer can log combo-length-over-time - the direct readout
        # of whether the clean-combo shaping is actually producing longer chains.
        # staleness = mod-reported control-loop lag in ticks (healthy 1, spin-phase 2;
        # -1/absent until the updated jar is running).
        info = {"dealt": dealt, "taken": taken, "combo": self.combo,
                "staleness": state.get("staleness", -1)}
        if my_h <= 0.0:
            reward += DEATH_REWARD
            done = True
            info["result"] = "death"
        elif tgt_h <= 0.0 or "tgt_health" not in state:
            reward += KILL_REWARD
            done = True
            info["result"] = "kill"
        elif self.ticks >= self.max_ticks:
            done = True
            info["result"] = "timeout"

        self.last_state = state
        return self._obs_vector(), reward, done, info

    def step(self, action):
        """Single-client convenience: send then immediately recv. The self-play
        harness does NOT use this - it calls step_send on both fighters first,
        then step_recv on both, to keep their control latency equal."""
        self.step_send(action)
        return self.step_recv()

    def close(self):
        if self.conn:
            self.conn.close()


# Observation/action dims for the trainer to import
OBS_DIM = INPUT_DIM
ACT_DIM = N_ACTIONS          # kept for back-compat; = MOVE_DIM
# Composite heads: (move, click, block, aim); AIM_RES_MAX bounds the aim residual.
ACTION_DIMS = {"move": MOVE_DIM, "click": CLICK_DIM, "block": BLOCK_DIM, "aim": AIM_DIM}
