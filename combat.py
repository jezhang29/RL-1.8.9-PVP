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
# Deliberately NOT scaled by ping: 1.8 hit selection is a CLIENT-side raytrace
# (3-block reach against the entity as this client renders it; the server only
# sanity-checks ~6 blocks in processUseEntity), so hitting the rendered - delayed -
# image registers fine, and leading past that hitbox just makes the raytrace miss.
# Ping shifts WHEN inputs land server-side, not where to point; see _uplink_ticks/
# _rtt_ticks below for the timing rules that do scale with it.
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
# The residual offsets the aim POINT, inside the proportional loop: (err + res) * GAIN.
# It used to be added OUTSIDE, to the turn rate (err * GAIN + res) - and a constant
# rate-residual parks the crosshair where the P-term exactly cancels it, i.e. at
# res / GAIN = 1.67x res off-target. With the old 8 deg cap that allowed a parked
# offset of ~13 deg, while the hitbox half-width (0.3 blocks) subtends only ~6 deg at
# IDEAL_RANGE 2.8 - a learned one-sided habit could park the crosshair clean off the
# side of the hitbox at max reach ("always aiming right of the person"). Inside the
# loop the parked offset equals the residual itself, and 4 deg keeps even a fully
# pinned residual on the box at combat range.
AIM_RES_MAX = 4.0   # deg, max magnitude of the learned yaw/pitch aim-point offset

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
# Crit jump. A hit only crits while FALLING (fallDistance>0 && !onGround - verified in
# EntityPlayer.attackTargetEntityWithCurrentItem), and after we land a hit the victim
# can't take damage again until their hurtTime (10 -> 0) reaches 0. A jump is ~12
# ticks: ~6 up, ~6 falling. So the ONLY profitable jump is right after our own hit,
# timed so the falling half overlaps the reopening damage window: leave the ground at
# tgt_hurt ~6-8 and we're falling from tgt_hurt ~0 onward, crit ready. Every other
# jump at close range is pure downside - airborne means ~1/5 the input authority and
# 0.91/tick friction, so a hit taken mid-jump becomes a full knockback flight (the
# old jump-every-CRIT_GAP-ticks-in-reach rule kept both bots airborne mid-trade,
# which is exactly the "both fly back and reset" pattern). Grounded + holding toward
# the attacker is the anti-knockback posture; stay on the floor unless juggling.
CRIT_HURT_HI = 8    # jump while target hurtTime is in [LO, HI] (counting down)...
CRIT_HURT_LO = 5
CRIT_RANGE = 4.5    # ...and they're still close enough to chase the crit down
CRIT_GAP = 12       # min ticks between crit jumps (one jump per i-frame cycle)
# Step-up hop: the one OTHER profitable jump - vaulting a 1-block step between us
# and the target (the mod's my_step_up: shin-height block ahead, headroom clear).
# Without it a fighter W-holds into the ledge forever; the jump head is gone, so
# terrain jumps must be scripted like the crit. Terrain locomotion, not combat
# timing - so unlike blockhit/crit it applies to scripted opponents too (a style
# stuck under a ledge is a degenerate curriculum, not a grounded-by-design one).
STEP_GAP = 6        # min ticks between hops so a mis-read ledge can't pogo-lock


TICK_MS = 50.0

def _uplink_ticks(obs):
    """Ticks for one of OUR packets to reach the server: half the tab-list RTT.
    0 on LAN or with an old jar (no my_ping key). Block reduction is decided
    server-side, so a block must be held this much LONGER to cover the same
    server-side window it covers at 0 ping."""
    ping = obs.get("my_ping") or 0
    return int(math.ceil(max(ping, 0.0) / 2.0 / TICK_MS))

def _rtt_ticks(obs):
    """Our full round trip in ticks. The observed tgt_hurt is downlink-stale AND
    our next attack registers uplink-late, so in observed-hurtTime terms every
    server-side timing window sits one full RTT earlier than it reads."""
    ping = obs.get("my_ping") or 0
    return int(round(max(ping, 0.0) / TICK_MS))


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
            self.block_left = BLOCK_TICKS + _uplink_ticks(obs)
        self.prev_tgt_hurt = tgt_hurt

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

        # --- Crit: jump only while juggling, timed to the i-frame reopen (see CRIT_*) ---
        self.ticks_since_jump += 1
        if self._crit_jump(obs, dist):
            action["jump"] = True
            self.ticks_since_jump = 0
        elif self._step_up_hop(obs, action):
            action["jump"] = True
            self.ticks_since_jump = 0

        return action

    def _step_up_hop(self, obs, action):
        """True when we're W-holding into a 1-block jumpable step toward the
        target (see STEP_GAP comment): grounded, unhurt (a knockback flight is
        not the moment to add air time), and moving forward into it."""
        return (bool(obs.get("my_step_up"))
                and action["w"]
                and obs.get("on_ground", False)
                and (obs.get("my_hurt", 0) or 0) == 0
                and self.ticks_since_jump >= STEP_GAP)

    def _crit_jump(self, obs, dist):
        """True when a jump right now yields a falling crit as the target's damage
        window reopens: we just hit them (their hurtTime mid-countdown), we're
        grounded and unhurt (not mid-knockback ourselves), and they're chaseable."""
        tgt_hurt = obs.get("tgt_hurt", 0) or 0
        my_hurt = obs.get("my_hurt", 0) or 0
        # Ping shifts the whole window earlier in OBSERVED-hurt terms (see
        # _rtt_ticks); hurtTime tops out at 10, so past ~100ms the window clips
        # and the crit is simply attempted as early as the readout allows.
        shift = _rtt_ticks(obs)
        hurt_hi = min(10, CRIT_HURT_HI + shift)
        hurt_lo = min(hurt_hi, CRIT_HURT_LO + shift)
        return (hurt_lo <= tgt_hurt <= hurt_hi
                and my_hurt == 0
                and dist <= CRIT_RANGE
                and obs.get("on_ground", False)
                and self.ticks_since_jump >= CRIT_GAP)

    def act_policy(self, obs, movement, click, block, aim_res, latch_block=True,
                   crit_jump=True):
        """Self-play variant: the RL policy owns MOVEMENT, ATTACK, a defensive BLOCK
        posture and an aim residual. The mechanical, event-timed techniques stay
        COMPUTED - like aim - because their correct timing is a known function of an
        event, not something worth learning from sparse reward at 20 Hz:

          - blockhit sprint-reset : a short block pulse fired the tick we land a hit
            (block cancels sprint, so the next hit re-triggers sprint knockback, and
            the raised sword buys spacing/reduction). This is the human "tap block
            right after connecting", which is only correct RELATIVE TO THE HIT - a
            learned per-tick coin flip could never time it, and un-gated it just spams.
          - crit-jump : jump ONLY inside the falling i-frame window right after our own
            hit (see _crit_jump). The learned jump head is GONE: a hop at any other
            time is pure airborne-knockback downside, which is exactly what produced
            the ~35% bunny-hop equilibrium a grounded human trivially punished.

        What stays LEARNED is the situational judgment: which movement primitive
        (engage / retreat / strafe / straight-line / s-tap), whether to commit a swing,
        whether to hold a SUSTAINED defensive block (the "trade against a cornered
        opponent" read), and the aim residual.

        movement : dict of w/a/s/d/sprint (bools) - the movement primitive.
        click    : bool - swing this tick (policy-decided; the tick rate is the only cap).
        block    : bool - hold a defensive sword-block this tick (policy-decided posture).
        aim_res  : (yaw_res, pitch_res) degrees, the learned correction on the computed
                   angle. Clamped to +-AIM_RES_MAX so the geometry stays the floor.
        latch_block : True (the learner) gets the scripted post-hit blockhit reflex and
                   holds a chosen block for BLOCK_MIN_TICKS so the SERVER registers it.
                   False (scripted opponents) passes block through verbatim - each style
                   owns its own block timing, so injecting a blockhit would corrupt the
                   curriculum. Their sprint reset comes from the style itself instead
                   (the rusher/trader/boxer W-tap; the turtle's block drops sprint).
        crit_jump : whether the post-hit crit reflex fires. True for EVERYONE including
                   scripted styles - unlike blocking, a crit is not a style choice but
                   the baseline damage every competent 1.8 fighter does, and gating it
                   on latch_block silently handicapped the whole curriculum. The
                   terrain step-up hop is likewise universal (see STEP_GAP).
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
        action["yaw_delta"] = clamp((yaw_err + res_y) * AIM_GAIN, -MAX_YAW, MAX_YAW)
        action["pitch_delta"] = clamp((pitch_err + res_p) * AIM_GAIN, -MAX_PITCH, MAX_PITCH)

        # --- Event: did we land a hit this tick? hurtTime is slammed to 10 on damage. ---
        tgt_hurt = obs.get("tgt_hurt", 0) or 0
        landed = tgt_hurt > self.prev_tgt_hurt
        self.prev_tgt_hurt = tgt_hurt

        # --- Attack (policy-driven) ---
        action["left_click"] = bool(click)
        self.ticks_since_click = 0 if click else self.ticks_since_click + 1

        # --- Block: policy defensive posture OR the scripted post-hit blockhit, both
        # feeding one hold counter. Reduction is decided SERVER-side, so a block must
        # be held (BLOCK_MIN_TICKS / BLOCK_TICKS + uplink) to register - a 1-tick blip
        # never does. Scripted opponents (latch_block False) own their block verbatim.
        if latch_block:
            if landed:
                self.block_left = max(self.block_left, BLOCK_TICKS + _uplink_ticks(obs))
            if block:
                self.block_left = max(self.block_left, BLOCK_MIN_TICKS + _uplink_ticks(obs))
            holding = self.block_left > 0
            if self.block_left > 0:
                self.block_left -= 1
        else:
            holding = bool(block)
        # No autoblock: in 1.8 you cannot swing while the sword is up, so a swing always
        # wins the tick over the block (release-to-hit = the manual blockhit). And you
        # cannot SPRINT while blocking, so a held block drops sprint - which is exactly
        # what resets it for the next hit's sprint-knockback.
        action["right_click"] = holding and not action["left_click"]
        if action["right_click"]:
            action["sprint"] = False

        # --- Crit-jump: scripted reflex (see _crit_jump / the docstring). The falling
        # half of the jump must land inside the victim's reopening damage window, right
        # after our own hit - the one profitable jump in the fight.
        # Jul 18 night: this used to be gated on latch_block, i.e. the LEARNER ONLY.
        # That handed the learner a permanent 1.5x damage multiplier on the bulk of its
        # hits that no scripted style could ever answer - the curriculum was unwinnable
        # by construction (wr_turtle pinned at 100%), not merely farmed. Crits are now
        # on their own flag and every fighter gets them: the window requires having
        # just landed a hit while grounded and unhurt, so it is the one jump that is
        # never bunny-hopping and never contradicts a "stays grounded" style. ---
        self.ticks_since_jump += 1
        if crit_jump and self._crit_jump(obs, dist):
            action["jump"] = True
            self.ticks_since_jump = 0
        # Step-up hop for EVERYONE, scripted styles included: it's terrain
        # locomotion, not combat timing (see STEP_GAP comment).
        elif self._step_up_hop(obs, action):
            action["jump"] = True
            self.ticks_since_jump = 0

        return action
