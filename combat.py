"""Rule-based combat: aim, attack timing, W-tap, blockhit, crit-jump.

Movement (WASD/sprint) is NOT decided here - it's supplied by whoever owns the
policy (behavioral cloning in ai_server, reinforcement learning in pvp_env). This
module only owns the parts that are solved control problems rather than learned
behaviour, and it keeps the per-fighter timing state (CPS cap, W-tap window, etc.).

Aim is computed, never learned: we know the target's exact position, so pointing at
it is closed-form geometry. A learned aim head had no proportional feedback and sank
the crosshair into the ground, which then fed the movement net garbage inputs.
"""
import math
import random

from features import calculate_optimal_angles, wrap_degrees

# Aim
AIM_GAIN = 0.6      # fraction of remaining angle error closed each tick
MAX_YAW = 30.0      # deg/tick, so a flick still looks human
MAX_PITCH = 20.0
# The state Python aims from is 1-2 ticks stale by the time the rotation is applied
# (mod->socket->policy->socket->next tick). Against a strafing/jumping target that
# lag reads as "aim trails then overshoots". Lead both fighters by their current
# velocity x this many ticks so the crosshair points where things WILL be when the
# correction lands. tgt_vy needs the updated mod; absent it we lead horizontally.
AIM_LEAD_TICKS = 2.0
# ...but CAP the lead displacement. A hit spikes the target's velocity (self-play pays
# for knockback, so launches are big), and an uncapped 2-tick projection then throws the
# aim point far past the target - yaw_err saturates at MAX_YAW and flips sign as the
# projection crosses the true bearing, which reads in-game as the crosshair spinning.
# Capping the lead vector's MAGNITUDE keeps leading useful for normal strafing/sprinting
# (~0.56 blocks) while trimming the spike-driven overshoot. Applies to both fighters.
AIM_LEAD_MAX = 1.5   # blocks, max magnitude of the velocity-lead displacement
# Learned-aim mode (self-play) adds a policy-chosen RESIDUAL on top of the computed
# angle - a small correction the RL agent learns (leading a strafing target, faking a
# flick) without the sparse-reward cold start of learning to aim from nothing. Bounded
# so the computed geometry stays the floor and a bad residual can't bury the crosshair.
AIM_RES_MAX = 8.0   # deg, max magnitude of the learned yaw/pitch correction

# Attack timing - a rule, not a habit to imitate (human clicking is a metronome
# with no state signal to learn from).
REACH = 3.4         # center-to-center; vanilla entity reach is ~3.0 to the hitbox
# Swing while CLOSING, not just when already in reach - a whiffed click out of range
# costs nothing in 1.8 (no attack cooldown), and clicking through the approach means
# the hit lands the instant the opponent enters reach instead of a tick or two late.
SWING_RANGE = 5.0   # start swinging this far out
SWING_YAW = 35.0    # ...as long as we're roughly facing them
# Click cadence. One tick is the 20 CPS ceiling; jitter to a fixed gap of 2 sometimes
# to land in the mid-teens with human-looking variation. avg gap = 2 - FAST_PROB,
# so CPS = 20 / (2 - FAST_PROB): 0.75 -> ~16, 0.9 -> ~18, 1.0 -> 20. Bump to taste.
CPS_FAST_PROB = 0.75

WTAP_TICKS = 1      # ticks to drop W+sprint after a hit (sprint-knockback reset)
BLOCK_TICKS = 3     # ticks to hold block after a hit (blockhit, rule mode)
BLOCK_MIN_TICKS = 3 # policy mode: min hold once block is chosen, so the SERVER
                    # registers the block (reduction is server-side, see act_policy)
CRIT_GAP = 12       # ticks between crit jumps


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def idle_action():
    return {"w": False, "a": False, "s": False, "d": False, "sprint": False,
            "jump": False, "left_click": False, "right_click": False,
            "yaw_delta": 0.0, "pitch_delta": 0.0}


class CombatController:
    """Holds one fighter's combat timing state across ticks."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.prev_tgt_hurt = 0
        self.ticks_since_click = 99
        self.click_gap = 1
        self.ticks_since_jump = 99
        self.wtap_left = 0
        self.block_left = 0

    def _aim_errors(self, obs):
        """Angle errors to the LED target position (see AIM_LEAD_TICKS/AIM_LEAD_MAX)."""
        def v(key):
            return obs.get(key, 0.0) or 0.0

        def lead(vx, vy, vz):
            # 2-tick velocity projection, clamped to AIM_LEAD_MAX so a knockback spike
            # can't fling the aim point past the target (see AIM_LEAD_MAX).
            dx, dy, dz = vx * AIM_LEAD_TICKS, vy * AIM_LEAD_TICKS, vz * AIM_LEAD_TICKS
            mag = math.sqrt(dx * dx + dy * dy + dz * dz)
            if mag > AIM_LEAD_MAX:
                scale = AIM_LEAD_MAX / mag
                dx, dy, dz = dx * scale, dy * scale, dz * scale
            return dx, dy, dz

        mdx, mdy, mdz = lead(v("my_vx"), v("my_vy"), v("my_vz"))
        tdx, tdy, tdz = lead(v("tgt_vx"), v("tgt_vy"), v("tgt_vz"))
        t_yaw, t_pitch = calculate_optimal_angles(
            obs["player_x"] + mdx, obs["player_y"] + mdy, obs["player_z"] + mdz,
            obs["target_x"] + tdx, obs["target_y"] + tdy, obs["target_z"] + tdz)
        yaw_err = wrap_degrees(t_yaw - obs["player_yaw"])
        pitch_err = t_pitch - obs["player_pitch"]
        return yaw_err, pitch_err

    def act(self, obs, movement):
        """Merge externally-chosen movement with computed aim + attack.

        movement: dict with keys w, a, s, d, sprint (bools).
        Returns a full action dict ready to send to the mod. Returns idle when
        there is no target (dist <= 0).
        """
        action = idle_action()
        action.update({k: bool(movement.get(k, False))
                        for k in ("w", "a", "s", "d", "sprint")})

        dist = obs.get("target_dist", -1.0)
        if dist is None or dist <= 0.0:
            return idle_action()

        # --- Aim ---
        yaw_err, pitch_err = self._aim_errors(obs)
        action["yaw_delta"] = clamp(yaw_err * AIM_GAIN, -MAX_YAW, MAX_YAW)
        action["pitch_delta"] = clamp(pitch_err * AIM_GAIN, -MAX_PITCH, MAX_PITCH)

        # --- Detect a landed hit: hurtTime is slammed to 10 on damage ---
        tgt_hurt = obs.get("tgt_hurt", 0) or 0
        if tgt_hurt > self.prev_tgt_hurt:
            self.wtap_left = WTAP_TICKS
            self.block_left = BLOCK_TICKS
        self.prev_tgt_hurt = tgt_hurt

        in_reach = dist <= REACH

        # --- Attack: swing through the approach, CPS-capped with jitter ---
        swinging = dist <= SWING_RANGE and abs(yaw_err) <= SWING_YAW
        self.ticks_since_click += 1
        if swinging and self.ticks_since_click >= self.click_gap:
            action["left_click"] = True
            self.ticks_since_click = 0
            self.click_gap = 1 if random.random() < CPS_FAST_PROB else 2

        # --- W-tap: cut sprint for a tick after a hit so the next carries full knockback ---
        if self.wtap_left > 0:
            action["w"] = False
            action["sprint"] = False
            self.wtap_left -= 1

        # --- Blockhit: sword-block for a few ticks after connecting ---
        if self.block_left > 0:
            action["right_click"] = True
            self.block_left -= 1

        # --- Crit: jump so the follow-up lands while falling (1.5x) ---
        self.ticks_since_jump += 1
        if in_reach and obs.get("on_ground", False) and self.ticks_since_jump >= CRIT_GAP:
            action["jump"] = True
            self.ticks_since_jump = 0

        return action

    def act_policy(self, obs, movement, click, block, aim_res):
        """Self-play variant: the RL policy owns attack, block, W-tap and an aim
        residual; only the base aim geometry and crit-jump stay computed.

        movement : dict of w/a/s/d/sprint (bools) - the movement primitive. W-tap is
                   just the policy choosing a no-sprint primitive right after a hit, so
                   it is NOT forced here the way rule mode forces it.
        click    : bool - swing this tick (policy-decided; the tick rate is the only cap).
        block    : bool - hold sword-block this tick (policy-decided blockhit timing).
        aim_res  : (yaw_res, pitch_res) degrees, the learned correction on the computed
                   angle. Clamped to +-AIM_RES_MAX so the geometry stays the floor.
        """
        action = idle_action()
        action.update({k: bool(movement.get(k, False))
                        for k in ("w", "a", "s", "d", "sprint")})

        dist = obs.get("target_dist", -1.0)
        if dist is None or dist <= 0.0:
            return idle_action()

        # --- Aim: computed base + learned residual ---
        yaw_err, pitch_err = self._aim_errors(obs)
        res_y = clamp(float(aim_res[0]), -AIM_RES_MAX, AIM_RES_MAX)
        res_p = clamp(float(aim_res[1]), -AIM_RES_MAX, AIM_RES_MAX)
        action["yaw_delta"] = clamp(yaw_err * AIM_GAIN + res_y, -MAX_YAW, MAX_YAW)
        action["pitch_delta"] = clamp(pitch_err * AIM_GAIN + res_p, -MAX_PITCH, MAX_PITCH)

        # Track hits for parity/telemetry, but do NOT auto W-tap or auto blockhit -
        # those are exactly what the policy is here to learn.
        tgt_hurt = obs.get("tgt_hurt", 0) or 0
        self.prev_tgt_hurt = tgt_hurt

        # --- Attack & block: policy-driven ---
        action["left_click"] = bool(click)
        self.ticks_since_click = 0 if click else self.ticks_since_click + 1
        # Block reduction is decided SERVER-side: the use-item packet must already be
        # processed when the attacker's hit arrives, so a 1-tick blip never registers.
        # Once the policy commits to a block, hold it for BLOCK_MIN_TICKS so the server
        # actually sees it (the policy re-choosing block just keeps refreshing this).
        if block:
            self.block_left = BLOCK_MIN_TICKS
        action["right_click"] = self.block_left > 0
        if self.block_left > 0:
            self.block_left -= 1

        # --- Crit: still computed (a periodic jump, not a thing worth learning) ---
        in_reach = dist <= REACH
        self.ticks_since_jump += 1
        if in_reach and obs.get("on_ground", False) and self.ticks_since_jump >= CRIT_GAP:
            action["jump"] = True
            self.ticks_since_jump = 0

        return action
