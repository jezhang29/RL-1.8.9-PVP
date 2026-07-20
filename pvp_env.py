"""Reinforcement-learning environment for one PvP fighter.

The mod connects in as a socket client (same protocol as ai_server); this class is
the server side. Each tick the RL policy chooses a COMPOSITE action - only the
situational judgment, the part a learned policy can beat hardcoded rules at:
  - a movement primitive (spacing / engage / disengage / strafe / s-tap),
  - whether to swing (attack),
  - whether to hold a defensive sword-block posture,
  - a small aim residual on top of the computed angle.
The mechanical, event-timed techniques stay COMPUTED (CombatController.act_policy),
because their correct timing is a known function of an event, not something worth
learning from sparse reward at 20 Hz: the base aim geometry, the post-hit blockhit
sprint-reset, and the crit-jump (fired only in the victim's falling i-frame window).

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
import random
from collections import deque

import numpy as np

from features import (frame_features, calculate_optimal_angles, wrap_degrees,
                      STACK, FRAME_DIM, INPUT_DIM)
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
    # Appended (checkpoints row-pad via adapt_ckpt_move_dim): pure S. The diagonal
    # retreats above drift sideways; an S-tap is a STRAIGHT momentum kill that holds
    # the spacing line - the "back off half a step, stay at max reach" move that
    # separates trading from combo control.
    (False, False, True,  False, False),  # 8 s-tap straight back
    # Appended: full key release. Distinct from the s-tap - no brake, momentum
    # decays through friction/knockback naturally. This is the physical primitive
    # behind the whole hitselection family (release on getting hit, repress W at
    # the knockback arc's peak to control hit sequencing in a trade); the policy
    # already sees my_hurt / my_vy / my_ground_h, it just couldn't express the
    # release until now.
    (False, False, False, False, False),  # 9 idle (hitselect release)
]
N_ACTIONS = len(ACTIONS)
IDLE_MOVE = 9   # index of the all-released primitive; trainer logs its uptake (idle_frac)

# Composite action space the policy chooses from each tick:
#   move  : Discrete(N_ACTIONS)     - which movement primitive (the situational read)
#   click : Discrete(2)             - swing or not (tick rate is the only CPS cap)
#   block : Discrete(2)             - hold a defensive sword-block posture or not
#   aim   : Box(2)                  - yaw/pitch residual (deg), clamped to +-AIM_RES_MAX
# Jump is NOT here: crit-jump is a scripted reflex (combat.act_policy / _crit_jump), and
# the post-hit blockhit sprint-reset is scripted too - the mechanical, event-timed
# techniques whose correct timing is known, so the policy only owns the situational parts.
MOVE_DIM = N_ACTIONS
CLICK_DIM = 2
BLOCK_DIM = 2
AIM_DIM = 2

KILL_REWARD = 10.0
DEATH_REWARD = -10.0
# Mutual death (both fighters die in the same exchange). Previously this had NO
# outcome of its own: whichever client's terminal frame the harness read first decided
# it, so the same event scored +10 half the time and -10 the other half - a coin flip
# on packet timing, which is pure variance to the critic and taught that suicidal
# trades are fine roughly half the time. Now it is one deterministic outcome, and a
# NEGATIVE one: dying is a loss even when you take them with you. Sized between the
# two extremes so both orderings stay sane - trading your life for a kill (-4) must be
# worse than disengaging alive (~0), so at low health the bot prefers to survive; but
# it must stay better than being killed cleanly (-10), so a fighter that is dying
# anyway is never discouraged from swinging back.
TIE_REWARD = -4.0
TIME_PENALTY = 0.01   # per tick, so stalling forever is never optimal
MAX_HEALTH = 20.0
# Firewall: hard bound on any single tick's reward. A legit tick tops out near a
# kill (finishing hit ~6 + KILL_REWARD 10 + bonuses), so +-20 never clips real
# gameplay - but it caps any shaping bug so it can't blow up the value target. An
# un-clamped aim tax once drove returns to -1e5 and saturated every head into a
# "hold jump+block" corner; this is the last line of defense against a repeat.
REWARD_CLIP = 20.0

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
# Getting comboed (Jul 18 night): the mirror. Raw taken-damage is linear, so eating 4
# chained hits was priced the same as 4 hits spread across a round - but the chained
# case is far worse (airborne on knockback, sprint broken, geometry theirs). Charge an
# escalating penalty for consecutive hits TAKEN inside COMBO_WINDOW. Landing our own
# hit resets THEIR chain: answering back is the escape, which is exactly the hitselect
# timing the trader style demonstrates. Because every answered hit resets the chain, an
# even 1-for-1 trade never reaches chain 2 and pays nothing - this prices LOSING
# exchanges and getting ragdolled, not engagement itself. Same cap as the bonus, so
# it is bounded and safe under REWARD_CLIP (not a repeat of the aim-tax detonation).
COMBO_TAKEN_PENALTY = 0.5   # per chain step past the first, capped at 4
# Initiative (Jul 19): the shaping above made COUNTER-hitting the only paid opener -
# answering back resets their chain, the scripted reflexes (blockhit, crit-jump) are
# all post-hit, and first blood pays once a ROUND. Observed result: the bot waits to
# get hit before every exchange instead of using hitselection as a combo STARTER
# (trade_frac 0.55 but almost always as the answerer, mean_combo stuck ~1.8). Pay the
# opening hit of a COLD exchange - neither side has landed for EXCHANGE_GAP ticks and
# we didn't trade into it - so initiating is worth the same as answering. First blood
# is excluded (already paid, and double-paying would over-weight the round opener).
# Once per exchange by construction: landing the hit resets ticks_since_my_hit.
OPENER_BONUS = 0.5
EXCHANGE_GAP = 40     # ticks (2 s) of neither side landing = the exchange went cold
# Follow-up conversion (Jul 19): the escalating combo bonus prices long chains, but
# the hard step is the FIRST follow-up - after hit 1 the target is flung past reach,
# the hesitation tax goes silent (ray off target), and nothing pays for re-closing,
# so "land one, drift, wait to counter" was the cheap corner. Two terms:
# - FOLLOWUP_BONUS: extra, once per chain, when it reaches 2 (converting the opener
#   into a chain is worth more than the same hit later in an established chain).
# - PRESSURE_COEF: while OUR chain is live (combo >= 1, inside COMBO_WINDOW) and the
#   target sits beyond IDEAL_RANGE, pay per block of ground actually closed. Signed
#   (backing off during your own window is charged), clamped to PRESSURE_STEP_MAX
#   per tick (real sprint speed - teleports/resets can't spike it), and it can't be
#   farmed: the window only opens by landing a real hit and lasts 1.25 s.
FOLLOWUP_BONUS = 0.5
PRESSURE_COEF = 0.1
PRESSURE_STEP_MAX = 0.35   # blocks/tick; ~sprint speed, caps the term at 0.035
# Telemetry only (no reward term): a landed hit within this many ticks of taking one is
# counted a TRADE, so the trainer can log trade_frac - "is it trading or comboing".
TRADE_WINDOW = 10     # ticks (0.5 s)
# Sprint-reset (W-tap): a hit with fresh sprint launches the target much further,
# which is what makes the follow-up land. Measured directly - the target's horizontal
# speed on the tick AFTER our hit - so the policy is paid for real knockback, however
# it achieves it (that's exactly the W-tap skill).
KB_COEF = 0.4
KB_NORM = 0.6         # blocks/tick; ~sprint-hit launch speed, so bonus tops out ~KB_COEF
# Blocked hit: a registered block shows up in the reward only as taken-damage-halved -
# a counterfactual PPO can barely see at 20Hz. Pay a small explicit bonus when we take
# a hit WHILE our block is registered (my_block true in the post-hit frame), sharpening
# the credit for proactive blocking. Small, so "stand there absorbing blocked hits"
# can never beat not being hit at all (a blocked sword hit still costs ~3.5 - 0.3).
BLOCKED_HIT_BONUS = 0.3
# Simulated network latency. On the LAN training rig real ping is ~0, so the ping
# features would be constant and the policy could never learn latency adaptation.
# Instead each round samples sim_lag in sim_lag_range (inclusive) and outgoing
# actions are queued that many ticks before being sent - one tick of action delay
# adds the same control-loop lag as ~50ms of real RTT, so the equivalent ping is
# injected into the my_ping/tgt_ping telemetry. That makes the features readable
# (resampled per round, so the policy must READ them, not memorize a delay) and
# routes through combat.py's block-hold/crit-shift rules - the exact path used at
# real ping. Approximation: only OUR actions lag, not the opponent's telemetry.
SIM_LAG_MS_PER_TICK = 50
# Aim-residual regularizer: the aim head gets NO entropy bonus (a Gaussian's entropy is
# just log-std) and nothing else anchors it to 0, so it random-walks into one-sided,
# state-dependent offsets that fight the computed geometry - the crosshair orbits and
# the intermittent "spin". Tax the squared residual so a lead only survives if it's
# earning hits.
# Jul 17 retune (0.004 -> 0.002, WHIFF 0.05 -> 0.10): 250+ updates of data showed
# whiff flat at ~33% and aim_bias parked at -1.5deg while the residual stayed ~0.
# The economics were inverted: a useful ~3deg lead cost 0.036/tick under the old tax,
# while eating the whiffs it would prevent cost only ~0.013/tick (0.05 x ~0.76 click
# rate x ~0.33 whiff) - so "no lead, accept the misses" was the cheaper corner, a
# priced-in local optimum. Now a 3deg lead is 0.018/tick vs ~0.025/tick of expected
# whiff penalty: correcting aim pays. Both terms stay bounded (residual is clamped to
# +-AIM_RES_MAX before squaring, so the tax maxes at 0.064/tick; REWARD_CLIP and the
# trainer's v_loss firewall backstop the rest) - this is NOT the unbounded tax that
# caused the detonation.
AIM_RES_COEF = 0.002
# Whiff penalty: 1.8 hit registration is a client-side ray from the eye through the
# crosshair against the target's AABB (reach ~3). The mod reports whether that ray
# connects THIS tick as my_ray_hit. Swinging inside reach while the ray MISSES is a
# wasted click AND a sign the aim (residual/lead) is pulling the crosshair off the
# hitbox - so tax it. A dense, direct "keep the crosshair ON the person when you
# commit" gradient for the aim head, where the sparse landed-hit reward is too slow.
# Only applied when the mod actually reports my_ray_hit (new jar); absent -> skipped,
# so an old jar never eats a blanket swing penalty. Small, so it shapes aim without
# ever discouraging swinging itself (a connecting swing is worth ~6 health).
WHIFF_PENALTY = 0.10
WHIFF_RANGE = 3.0     # blocks; matches the mod's 3.0 ray reach - past this a dead-on
                      # ray legitimately falls short, so no penalty out there
# Hesitation: the mirror image of the whiff tax. Observed live: the bot sometimes
# stands at reach and only starts swinging AFTER it takes a hit. The whiff tax alone
# makes that a priced-in habit - clicking with unsettled aim costs 0.10 a miss, while
# passing on a free hit costs nothing, so "wait until something (usually getting hit)
# stabilizes the geometry" is the cheap corner. Charge the pass: crosshair ray ON the
# target (my_ray_hit in the frame the policy acted from), inside reach, target's
# damage window OPEN (tgt_hurt 0 - clicking into i-frames deals nothing, no charge),
# and the policy neither clicked nor blocked (a held block is a legitimate posture,
# not hesitation). In 1.8 there is no attack cooldown, so a click under exactly these
# conditions is always correct - this taxes pure passivity only. Sized under the
# whiff tax so "click when the ray connects, hold off when it doesn't" stays the
# ordering; both are trivially bounded per tick.
HESITATION_PENALTY = 0.04
# NOTE: jump used to be a learned policy head and ran away to ~35% of ticks (constant
# bunny-hopping = permanently airborne = full knockback taken, the "how is this so bad"
# fight). It cost an AIRBORNE_HIT_PENALTY + JUMP_TAX pair of shaping band-aids. The
# Jul 17 re-hybridization removed the head entirely: crit-jump is now a scripted reflex
# (combat.act_policy / _crit_jump) that only fires in the victim's i-frame window, so
# there is nothing to tax and those two constants are gone.


class PvPEnv:
    def __init__(self, port=9999, reset_commands=None, max_ticks=1200, host='127.0.0.1',
                 sim_lag_range=(0, 0)):
        self.port = port
        self.host = host
        # Simulated-latency state (see SIM_LAG_MS_PER_TICK). sim_lag resamples each
        # reset; peer_lag is the OTHER fighter's sim_lag, set by the self-play
        # harness so tgt_ping tells the truth about the opponent's delay.
        self.sim_lag_range = sim_lag_range
        self.sim_lag = 0
        self.peer_lag = 0
        self._action_queue = deque()
        # This round's kit, injected into every incoming state (see annotate_ping):
        # {"my_armor", "tgt_armor", "my_atk", "tgt_atk"} in the frame_features v8
        # units. The trainer sets it per round alongside the /replaceitem reset
        # commands - the mod never reports gear, so Python (which assigned it) is
        # the source of truth. Empty dict = no gear info = features read 0.
        self.gear_info = {}
        # Chat commands run at the start of each episode (teleport/heal/regear).
        # Needs op on your own world. e.g. ["/tp SelfBot 0 64 0", "/effect SelfBot 6 1 255"]
        # May also be a CALLABLE returning the list, evaluated once per reset - that's
        # how the trainer randomizes spawn geometry between rounds.
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
        return self.annotate_ping(json.loads(complete[-1]))

    def annotate_ping(self, state):
        """Add the simulated latency (ours and the opponent's) on top of the real
        - on LAN, near-zero - tab-list readings, so the ping features and combat.py's
        timing rules see the delay the action queue is actually creating."""
        if self.sim_lag:
            state["my_ping"] = (max(state.get("my_ping") or 0, 0)
                                + self.sim_lag * SIM_LAG_MS_PER_TICK)
        if self.peer_lag and "target_dist" in state and (state["target_dist"] or 0) > 0:
            state["tgt_ping"] = (max(state.get("tgt_ping") or 0, 0)
                                 + self.peer_lag * SIM_LAG_MS_PER_TICK)
        # Gear is Python-assigned, not mod telemetry (see gear_info) - stamp it on
        # the same path so features and _drain both see it.
        if self.gear_info:
            state.update(self.gear_info)
        return state

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
        # Roll this round's simulated latency; in-flight actions from the old round
        # are dropped (reset commands below bypass the queue on purpose).
        self.sim_lag = random.randint(*self.sim_lag_range)
        self._action_queue.clear()

        # Fire reset commands piggybacked on an idle action. If a fighter just died
        # the mod auto-respawns it, but a dead/mid-respawn player can't be teleported,
        # so the volley goes out TWICE: the first catches the survivor, the second
        # lands after the respawn. Then idle so the first real observation isn't a
        # mid-teleport frame.
        # Evaluate once per reset (same commands for both volleys, so the respawned
        # fighter lands on the same spot the survivor was already sent to).
        cmds = (self.reset_commands() if callable(self.reset_commands)
                else self.reset_commands)
        reset_msg = {"command": "; ".join(cmds)} if cmds else {}
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
        self.combo_taken = 0
        self.ticks_since_tgt_hit = 999
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
        mod_action = self.combat.act_policy(
            self.last_state, movement,
            click=bool(action["click"]), block=bool(action["block"]),
            aim_res=action["aim"],
            latch_block=bool(action.get("latch_block", True)),
            crit_jump=bool(action.get("crit_jump", True)))
        # Jump is now a SCRIPTED crit reflex inside act_policy (not a policy head).
        # Surface whether it fired this tick so the rollout can log a crit-jump rate.
        self._scripted_jump = bool(mod_action.get("jump"))
        # Simulated latency: hold the BODY inputs (move/click/block/jump) sim_lag
        # ticks before they reach the mod - that models the ping-delayed control
        # loop and gives the ping features signal. But the AIM (yaw/pitch) is NOT
        # delayed: in real Minecraft your own view is client-authoritative and
        # rotates instantly; ping only makes the OPPONENT you aim at stale, which
        # the telemetry already reflects. Delaying it also fed a proportional aim
        # controller (combat.py closes 60%/tick) its own error sim_lag ticks late -
        # textbook loop instability, so the crosshair overshoots and orbits (the
        # "spinning"). Pop the delayed body, then stamp THIS tick's fresh aim on it.
        self._action_queue.append(mod_action)
        if len(self._action_queue) > self.sim_lag:
            out = dict(self._action_queue.popleft())
        else:
            out = {}  # pipeline still filling: no body inputs yet (dead control loop)
        out["yaw_delta"] = mod_action["yaw_delta"]
        out["pitch_delta"] = mod_action["pitch_delta"]
        self._send(out)

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

        # Blocked hit: got hit while our block was registered -> the damage above was
        # halved; pay the explicit bonus that makes that visible (see constants)
        if taken > 0.0 and (state.get("my_block") or 0.0):
            reward += BLOCKED_HIT_BONUS

        # Aim-residual tax (see AIM_RES_COEF): penalize the residual the policy emitted
        # this tick, so the aim head can't drift into a free-floating offset that spins
        # the crosshair. Bounded to +-AIM_RES_MAX (the clamp already applied to the
        # game) before squaring: residuals past the clamp don't change gameplay, so
        # taxing them un-bounded bought nothing - but it let a drifting Gaussian mean
        # feed a quadratic penalty with no ceiling, which detonated the value loss and
        # collapsed the policy into "hold jump+block". The tax now maxes at a fixed cost.
        res_y = max(-AIM_RES_MAX, min(AIM_RES_MAX, float(action["aim"][0])))
        res_p = max(-AIM_RES_MAX, min(AIM_RES_MAX, float(action["aim"][1])))
        reward -= AIM_RES_COEF * (res_y * res_y + res_p * res_p)

        # Whiff: swung inside reach but the crosshair ray missed the AABB (the exact
        # client-side hit test, see WHIFF_PENALTY). state.get is None on an old jar
        # (no my_ray_hit key) -> no penalty; 0.0 = a real miss -> penalized.
        ray_hit = state.get("my_ray_hit")
        click_eval = bool(action["click"] and ray_hit is not None
                          and new_d is not None and 0.0 < new_d <= WHIFF_RANGE)
        whiffed = click_eval and not ray_hit
        if whiffed:
            reward -= WHIFF_PENALTY

        # Hesitation (see HESITATION_PENALTY): judged on the PRE-action frame - the
        # state the policy actually decided from. Free hit available (ray on target,
        # in reach, their damage window open) and the policy neither swung nor held
        # block -> charged. None-safe: an old jar never streams my_ray_hit, so this
        # stays off there, exactly like the whiff tax.
        if (not action["click"] and not action["block"]
                and (self.last_state.get("my_ray_hit") or 0.0)
                and (self.last_state.get("tgt_hurt") or 0) == 0
                and prev_d is not None and 0.0 < prev_d <= WHIFF_RANGE):
            reward -= HESITATION_PENALTY

        # (The old JUMP_TAX / AIRBORNE_HIT_PENALTY shaping is gone: jump is no longer a
        # policy head but a scripted crit reflex that only fires inside the victim's
        # i-frame window, when the opponent can't hit back - so there is no runaway hop
        # rate to tax, and taxing an involuntary knockback-airborne frame the policy
        # can't avoid only added noise. See combat.act_policy / _crit_jump.)

        # First blood, once per round, signed. Surfaced to the trainer so the CSV can
        # log what fraction of rounds we actually win the opening-hit race.
        first_blood = 0
        if not self.first_hit_done and (dealt > 0.0 or taken > 0.0):
            self.first_hit_done = True
            first_blood = 1 if dealt > taken else -1
            reward += FIRST_HIT_BONUS * first_blood

        # Sprint-reset: pay for the knockback our LAST hit actually produced (the
        # target's launch speed shows up in the frame after the health drop)
        if self.kb_check:
            kb = math.hypot(state.get("tgt_vx", 0.0) or 0.0, state.get("tgt_vz", 0.0) or 0.0)
            reward += KB_COEF * min(kb / KB_NORM, 1.0)
            self.kb_check = False

        # Opener (see OPENER_BONUS): this hit started a cold exchange - neither side
        # had landed for EXCHANGE_GAP ticks (counters still hold last tick's values
        # here), we didn't trade into it, and it isn't the round's first blood
        # (paid above). Prices INITIATING an exchange the same as answering one.
        opener = (dealt > 0.0 and taken == 0.0 and first_blood == 0
                  and self.ticks_since_my_hit >= EXCHANGE_GAP
                  and self.ticks_since_tgt_hit >= EXCHANGE_GAP)
        if opener:
            reward += OPENER_BONUS

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
                if self.combo == 2:
                    # First follow-up landed: the opener became a chain (see
                    # FOLLOWUP_BONUS). Once per chain by construction.
                    reward += FOLLOWUP_BONUS
            self.ticks_since_my_hit = 0
            self.kb_check = True

        # Pressure (see PRESSURE_COEF): our chain is live and the target is beyond
        # ideal range (knocked back) - pay for ground actually closed this tick, so
        # the moment AFTER landing a hit has a gradient toward converting instead
        # of drifting. Signed and clamped; stacks with the potential range term.
        if (self.combo >= 1 and self.ticks_since_my_hit <= COMBO_WINDOW
                and prev_d is not None and new_d is not None
                and prev_d > IDEAL_RANGE and new_d > IDEAL_RANGE):
            reward += PRESSURE_COEF * max(-PRESSURE_STEP_MAX,
                                          min(PRESSURE_STEP_MAX, prev_d - new_d))

        # Getting comboed: the mirror of the block above (see COMBO_TAKEN_PENALTY).
        # Landing our own hit breaks THEIR chain - answering back is the escape - so a
        # 1-for-1 trade never escalates; only unanswered chained hits do. `traded` is
        # computed BEFORE the resets: our hit counts as a trade if we took damage this
        # same tick or within TRADE_WINDOW before it (telemetry only, no reward).
        self.ticks_since_tgt_hit += 1
        traded = dealt > 0.0 and (taken > 0.0
                                  or self.ticks_since_tgt_hit <= TRADE_WINDOW)
        if dealt > 0.0:
            self.combo_taken = 0
        if taken > 0.0:
            if self.combo_taken > 0 and self.ticks_since_tgt_hit <= COMBO_WINDOW:
                self.combo_taken += 1
            else:
                self.combo_taken = 1
            if self.combo_taken >= 2:
                reward -= COMBO_TAKEN_PENALTY * min(self.combo_taken - 1, 4)
            self.ticks_since_tgt_hit = 0

        self.prev_my_health, self.prev_tgt_health = my_h, tgt_h

        done = False
        # combo = current CLEAN-hit chain length (0 between chains, reset by taken>0).
        # Surfaced so the trainer can log combo-length-over-time - the direct readout
        # of whether the clean-combo shaping is actually producing longer chains.
        # staleness = mod-reported control-loop lag in ticks (healthy 1, spin-phase 2;
        # -1/absent until the updated jar is running).
        info = {"dealt": dealt, "taken": taken, "combo": self.combo,
                # Exchange telemetry (Jul 18 night): combo_taken = current chain of
                # unanswered hits taken; first_blood = +-1 on the tick the opening hit
                # resolved (0 otherwise); trade = this tick's landed hit came within
                # TRADE_WINDOW of taking one; blocked_hit = took a hit with block
                # registered (the damage above was halved).
                "combo_taken": self.combo_taken, "first_blood": first_blood,
                "trade": traded, "opener": opener,
                "blocked_hit": bool(taken > 0.0 and (state.get("my_block") or 0.0)),
                "staleness": state.get("staleness", -1),
                # Aim-quality telemetry: whiff = in-range click whose crosshair ray
                # missed; click_eval = the denominator (in-range clicks with the new
                # jar's ray key present). Rollout aggregates these into a whiff rate.
                "whiff": whiffed, "click_eval": click_eval,
                # Whether the scripted crit-jump reflex fired this tick (jump is no
                # longer a policy head); the rollout aggregates it into a crit rate.
                "jumped": getattr(self, "_scripted_jump", False)}
        # Signed geometric yaw error (no lead, no residual) at combat range, for the
        # aim-bias metric. Positive = the target sits RIGHT of the crosshair (we still
        # need to turn right); NEGATIVE = the crosshair sits right of the target - a
        # persistent negative mean is exactly the reported "always aims to the right".
        if new_d is not None and 0.0 < new_d <= 6.0:
            t_yaw, _ = calculate_optimal_angles(
                state["player_x"], state["player_y"], state["player_z"],
                state["target_x"], state["target_y"], state["target_z"])
            info["yaw_err"] = wrap_degrees(t_yaw - state["player_yaw"])
        # Mutual death seen in ONE frame (both healths at zero together). Checked before
        # the plain death branch, which would otherwise swallow it as a clean loss.
        # The cross-client case - each client seeing only its own side of the same
        # mutual death a tick apart - is reconciled in SelfPlayHarness.step.
        if my_h <= 0.0 and (tgt_h <= 0.0 or "tgt_health" not in state):
            reward += TIE_REWARD
            done = True
            info["result"] = "tie"
        elif my_h <= 0.0:
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
        # Firewall (see REWARD_CLIP): last bound before the value target sees this.
        reward = max(-REWARD_CLIP, min(REWARD_CLIP, reward))
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
# Jump is scripted (see above), so there is no jump head.
ACTION_DIMS = {"move": MOVE_DIM, "click": CLICK_DIM, "block": BLOCK_DIM,
               "aim": AIM_DIM}
