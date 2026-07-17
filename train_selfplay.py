"""Self-play PPO for the movement policy.

The learner controls one Minecraft client (port 9999); a frozen snapshot of a past
self controls a second client (port 9998). Aim and attack are the shared computed
CombatController on both sides, so the ONLY thing under optimization is movement -
spacing, strafing, when to commit and when to retreat. That tiny action space, plus
a feature extractor warm-started from behavioral cloning, is what makes real-time
(20 Hz) self-play converge in an overnight run instead of never.

Opponent = a pool of past snapshots (not just the latest self), which stops the
policy from collapsing into a single strategy that only beats its current self.

Run:  python train_selfplay.py
See the two-client setup notes at the bottom before running.
"""
import copy
import csv
import glob
import os
import random
import select
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from features import frame_features
from pvp_env import PvPEnv, OBS_DIM, ACTION_DIMS, KILL_REWARD, DEATH_REWARD
from train_model import PvPCloner, adapt_ckpt_frame_dim, adapt_ckpt_move_dim

# Simulated-latency range (ticks of action delay, ~50ms RTT each) for BOTH fighters,
# resampled per round - see pvp_env.SIM_LAG_MS_PER_TICK. 0-4 covers LAN through
# ~200ms, the range friends-over-internet actually shows up with.
SIM_LAG_RANGE = (0, 4)

# The server grants a freshly RESPAWNED player a few seconds of spawn-protection
# invulnerability, which rigs every round's opening exchange: the previous round's
# survivor swings through a ghost (0 damage) while the respawned fighter hits back
# for real - a free FIRST_HIT_BONUS and an unearned head start, paid to the LOSER of
# the last round. Both fighters idle this many ticks after the reset teleport (in
# lockstep, so it costs one window, not two) before the round starts; health is
# re-baselined afterwards so nothing during the wait leaks into the reward. Set to
# your server's protection window; 60 = 3 s (~SPAWN_PROT_TICKS/20 s slower rounds).
SPAWN_PROT_TICKS = 60

# Scripted sparring personalities (see ScriptedOpponent): this fraction of rounds is
# fought against a rule-driven style instead of a pool snapshot. Pure self-play is a
# monoculture - measured: blk collapsed 5% -> 0% over one night, because no snapshot
# ever blocks, so blocking never shows its value AND the counter to a blocker is
# never in the training data. A human turtling (hold block, click on cooldown) then
# wins every trade 2:1.
# Jul 17 rebalance: wr_script hit 86% mean with HALF of updates at 100% - all three
# original styles are farmed, so the scripted rounds had stopped producing gradient
# (and the counters that farm them fold instantly against a real human). The boxer
# (strafe mixups + W-tap + block-hit + irregular aggression bursts) is the new
# pressure and is weighted with the turtle; prob raised 0.35 -> 0.45 because the
# snapshot pool is the part that is a monoculture, not the scripts.
SCRIPTED_PROB = 0.45
SCRIPTED_STYLES = ("turtle", "rusher", "kiter", "boxer")
SCRIPTED_WEIGHTS = (2, 1, 1, 2)

# --- hyperparameters --------------------------------------------------------
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ROLLOUT_STEPS = 2048
UPDATE_EPOCHS = 4
MINIBATCH = 256
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP = 0.2
# 1e-4, not 3e-4: we are FINE-TUNING a working fighter, and at 3e-4 the policy could
# walk itself off a cliff faster than the noisy 20Hz reward could pull it back
# (that's what the upd-55 -> upd-270 collapse looked like).
LR = 1e-4
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
TOTAL_UPDATES = 2000
# 10, not 25: at 25 a restarted run spent its first ~2 hours fighting only the
# resume-point snapshot (winrate drifted to 90-100% = no learning signal left).
# 10 keeps the pool tracking the learner; POOL_SIZE 10 still spans 100 updates.
SNAPSHOT_EVERY = 10       # updates between adding the current policy to the pool
POOL_SIZE = 10

LEARNER_PORT = 9999
OPPONENT_PORT = 9998

# One row per update, appended across restarts (resumed runs keep extending the same
# file). plot_progress.py turns this into the over-time graph. Header written once.
METRICS_CSV = "selfplay_metrics.csv"
METRICS_HEADER = ["update", "rounds", "timeouts", "winrate", "wr_old", "wr_new",
                  "wr_script", "wr_pin", "dealt", "taken", "hits", "mean_combo", "max_combo",
                  "blk", "clk", "jmp", "whiff", "aim_bias", "avg_reward",
                  "ploss", "vloss", "staleness",
                  # Jul 17 additions. Per-style winrates: the aggregate wr_script hid
                  # "turtle farmed, boxer unbeaten". Entropies (nats): the direct "is
                  # it still exploring?" readout the log never had - ent_move healthy
                  # range ~1.0-2.0, near-0 = frozen policy. res_yaw/res_pitch: mean
                  # LEARNED aim offset (deg) - compare against aim_bias to see whether
                  # the residual is actually canceling the systematic miss.
                  # move_top_frac: share of the single most-used movement primitive
                  # (movement-collapse detector, 1/9 = uniform, ~1.0 = one key held).
                  "wr_turtle", "wr_rusher", "wr_kiter", "wr_boxer",
                  "ent_move", "ent_click", "ent_block", "ent_jump",
                  "aim_std", "res_yaw", "res_pitch", "move_top", "move_top_frac"]


def append_metrics(row, path=METRICS_CSV):
    # Columns get appended over time (whiff/aim_bias/wr_script). If the file on disk
    # predates the current header, rewrite it once under the new header - old rows
    # keep empty cells for the new columns - so one file stays parseable across
    # feature additions instead of pandas choking on ragged rows.
    if os.path.exists(path):
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            old_fields = reader.fieldnames
            old_rows = list(reader) if old_fields != METRICS_HEADER else None
        if old_rows is not None:
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=METRICS_HEADER)
                w.writeheader()
                for r in old_rows:
                    w.writerow({k: r.get(k, "") for k in METRICS_HEADER})
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRICS_HEADER)
        if new_file:
            w.writeheader()
        w.writerow(row)

# --- arena geometry & spawns -------------------------------------------------
# Reset = chat commands run by the mod (it SPLITS on ";"). EDIT the two names to
# your accounts. /effect uses NUMERIC ids only in 1.8.9 (6=instant_health heals the
# survivor - the loser respawns full anyway; 23=saturation keeps sprint fed). The
# arena (user's survey): open rectangle floor y=65, x in [-5,8], z in [-6,12],
# enclosed by 4-tall walls at x=9 (E), z=13 (N), z=-7 (S), x=-6 (W). A 2-wide
# corridor opens off the west into x in [-8,-7], z in [-6,3] (dead-ends at the
# south). All fight coords stay ~2 blocks inside a wall. Yaw in 1.8: 0=+z(S),
# 90=-x(W), 180=-z(N), 270=+x(E).
LEARNER_NAME = "SantiagoSea"
OPPONENT_NAME = "Bee__Bot"

# Open spawns are JITTERED: fixed openings make every round an exact mirror (first
# hit is a coin flip, one approach geometry only). Kept under the 16-block feature
# range.
X_JITTER = 3
Z_JITTER = 2

# WALL_PIN_PROB of rounds spawn as a PIN instead of the open center: one fighter
# starts ~2 blocks off a wall with the other in front, so the aggressor's knockback
# can't open the gap to combo. That is the no-knockback forced trade where blocking
# wins (the user's turtle insight) and which open spawns never produce.
WALL_PIN_PROB = 0.30
# tag -> (pinned placement, aggressor placement) as (x, z, yaw); pinned's back is to
# the wall, both facing each other. pinC is the 2-wide corridor: no lateral room, so
# neither side can strafe out of the trade - the purest forced-trade case.
PIN_SPAWNS = {
    "pinE": ((7, 3, 90), (3, 3, 270)),     # east wall x=9  (wall runs along z)
    "pinN": ((1, 11, 180), (1, 7, 0)),     # north wall z=13 (wall runs along x)
    "pinS": ((1, -5, 0), (1, -1, 180)),    # south wall z=-7 (wall runs along x)
    "pinC": ((-7, -5, 0), (-7, -1, 180)),  # corridor dead end (x=-7, 2-wide)
}


def _effects():
    return [f"/effect {n} {e} 1 {a}"
            for n in (LEARNER_NAME, OPPONENT_NAME)
            for e, a in ((6, 5), (23, 9))]   # 6=instant_health, 23=saturation


def open_spawn():
    """Both fighters near center, jittered, facing roughly toward each other."""
    ax = random.randint(-X_JITTER, X_JITTER)
    az = -2 + random.randint(-Z_JITTER, Z_JITTER)
    bx = random.randint(-X_JITTER, X_JITTER)
    bz = 8 + random.randint(-Z_JITTER, Z_JITTER)
    return [f"/tp {LEARNER_NAME} {ax} 65 {az} 0 0",
            f"/tp {OPPONENT_NAME} {bx} 65 {bz} 180 0"] + _effects()


def wall_pin_spawn(tag, pin_learner):
    """A pin round. pin_learner True -> the LEARNER is backed against the wall (must
    fight out of the corner); False -> the OPPONENT is pinned (the learner is the
    aggressor who must NOT keep trading into a wall-blocker). Both fighters jitter
    TOGETHER along the wall direction so the pin geometry is preserved."""
    (px, pz, pyaw), (ax, az, ayaw) = PIN_SPAWNS[tag]
    if tag == "pinE":                    # wall along z -> shift both in z
        j = random.randint(-2, 2); pz, az = pz + j, az + j
    elif tag in ("pinN", "pinS"):        # wall along x -> shift both in x
        j = random.randint(-2, 2); px, ax = px + j, ax + j
    # pinC: corridor is only 2 wide, no room to jitter.
    pinned, other = ((LEARNER_NAME, OPPONENT_NAME) if pin_learner
                     else (OPPONENT_NAME, LEARNER_NAME))
    return [f"/tp {pinned} {px} 65 {pz} {pyaw} 0",
            f"/tp {other} {ax} 65 {az} {ayaw} 0"] + _effects()


# Constructor default (the very first reset, before new_episode picks per-round).
reset_commands = open_spawn


# --- network ----------------------------------------------------------------
class ActorCritic(nn.Module):
    """Multi-head actor-critic. One shared feature trunk feeds four action heads
    (movement, click, a defensive-block posture, and a continuous aim residual) plus
    the value head. The heads are treated as INDEPENDENT given the state, so the joint
    log-prob is the sum of the per-head log-probs and the joint entropy the sum of
    entropies - standard factorized-policy PPO. (There is no jump head: crit-jump is a
    scripted reflex, see combat.act_policy - the Jul 17 re-hybridization.)
    """

    def __init__(self, obs_dim=OBS_DIM, dims=ACTION_DIMS):
        super().__init__()
        # Widths MATCH the BC net (512->256->128) so all three feature layers
        # warm-start. No dropout here (unlike BC) - on-policy PPO wants the same
        # distribution at collection and update time; dropout has no params so the
        # Linear shapes still line up for copying.
        self.base = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.move_head = nn.Linear(128, dims["move"])
        self.click_head = nn.Linear(128, dims["click"])
        self.block_head = nn.Linear(128, dims["block"])
        self.aim_mean = nn.Linear(128, dims["aim"])
        # State-independent log-std for the aim residual, started small (std~0.37 deg)
        # so early aim ~= the computed angle (the working floor) and RL explores gently
        # instead of flinging the crosshair around at init.
        self.aim_logstd = nn.Parameter(torch.full((dims["aim"],), -1.0))
        self.critic = nn.Linear(128, 1)
        # Zero-init the policy heads: at step 0 move/click/block are uniform-random
        # and the aim residual is exactly 0 (aim == the computed geometry). Default
        # random init gave each head a constant state-dependent bias - e.g. a pinned
        # aim offset that read as "staring at the sky" on the first live run.
        for head in (self.move_head, self.click_head, self.block_head, self.aim_mean):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _heads(self, x):
        h = self.base(x)
        return (self.move_head(h), self.click_head(h), self.block_head(h),
                self.aim_mean(h), self.critic(h).squeeze(-1))

    def _dists(self, move_l, click_l, block_l, aim_m):
        # Clamp: the entropy-ish pressure in PPO otherwise inflates aim std without
        # bound (measured drift 0.37deg -> 0.67deg over one night), which reads as
        # crosshair wobble/overshoot in game. 1 deg is plenty of exploration.
        std = torch.exp(self.aim_logstd.clamp(-2.5, 0.0))
        return (Categorical(logits=move_l), Categorical(logits=click_l),
                Categorical(logits=block_l), Normal(aim_m, std))

    def act(self, obs_np, greedy=False):
        x = torch.as_tensor(obs_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            move_l, click_l, block_l, aim_m, value = self._heads(x)
            md, cd, bd, ad = self._dists(move_l, click_l, block_l, aim_m)
            if greedy:
                move, click, block = (move_l.argmax(-1), click_l.argmax(-1),
                                      block_l.argmax(-1))
                aim = aim_m
            else:
                move, click, block, aim = (md.sample(), cd.sample(), bd.sample(),
                                           ad.sample())
            logp = (md.log_prob(move) + cd.log_prob(click) + bd.log_prob(block)
                    + ad.log_prob(aim).sum(-1))
        action = {
            "move": int(move.item()), "click": int(click.item()),
            "block": int(block.item()),
            "aim": aim.squeeze(0).cpu().numpy().astype(np.float32),
        }
        return action, float(logp.item()), float(value.item())

    @torch.no_grad()
    def head_entropies(self, obs):
        """Mean per-head policy entropy (nats) over a batch of observations - the
        direct 'is it still trying new things?' readout. Uniform-random ceilings:
        move ln(9)~=2.20, click/block ln(2)~=0.69. A head sliding toward 0 has
        stopped exploring; the CSV never recorded this, so exploration collapse was
        only ever visible through its downstream symptoms (blk hitting 0%)."""
        md, cd, bd, _ = self._dists(*self._heads(obs)[:4])
        return tuple(float(d.entropy().mean()) for d in (md, cd, bd))

    def evaluate(self, obs, move_a, click_a, block_a, aim_a):
        move_l, click_l, block_l, aim_m, values = self._heads(obs)
        md, cd, bd, ad = self._dists(move_l, click_l, block_l, aim_m)
        logp = (md.log_prob(move_a) + cd.log_prob(click_a) + bd.log_prob(block_a)
                + ad.log_prob(aim_a).sum(-1))
        # Entropy bonus on the CATEGORICAL heads only. A Gaussian's entropy is just
        # log-std, so including it pays the optimizer to make aim noisier forever;
        # the aim head gets its exploration from the (clamped) learned std instead.
        entropy = md.entropy() + cd.entropy() + bd.entropy()
        return logp, entropy, values


def warm_start_from_bc(model, path="pvp_model_v2.pth"):
    """Copy the BC feature extractor's linear layers into the actor-critic base.
    The BC base has dropout (different layer indices), so copy the 3 Linears by
    position rather than load_state_dict. The actor/critic heads stay fresh.
    """
    try:
        bc = PvPCloner()
        # adapt: BC weights may predate newly appended per-frame features
        bc.load_state_dict(adapt_ckpt_frame_dim(torch.load(path, map_location="cpu")))
    except FileNotFoundError:
        print("No BC checkpoint found - starting from random weights.")
        return
    bc_linears = [m for m in bc.base if isinstance(m, nn.Linear)]
    ac_linears = [m for m in model.base if isinstance(m, nn.Linear)]
    copied = 0
    for src, dst in zip(bc_linears, ac_linears):
        if src.weight.shape == dst.weight.shape:
            dst.weight.data.copy_(src.weight.data)
            dst.bias.data.copy_(src.bias.data)
            copied += 1
    print(f"Warm-started {copied}/{len(ac_linears)} base layers from BC.")


# --- scripted sparring personalities ------------------------------------------
class ScriptedOpponent:
    """Rule-driven sparring styles the snapshot pool can never produce.

    Quacks like ActorCritic.act() (returns action, logp, value) but reads the RAW
    telemetry (env.last_state) instead of the feature vector and follows a fixed
    rule. Aim stays the computed geometry (zero residual) via act_policy, and these
    pass latch_block=False so their block is verbatim, not the learner's held latch.
    Styles:
      turtle - the both-buttons-held human. Sprints in with the sword DOWN (blocking
               is sneak speed, so it can't chase blocked), then at trade range holds
               block AND clicks (~10 CPS block-hitting: half damage in, full out). Wins
               a sustained trade 2:1; the counter (don't stand and trade a blocker -
               block back, or make space and poke) is what the learner must discover.
      rusher - sprint-in W-tapper, ~10 CPS, never blocks: punishes passivity and
               makes PROACTIVE blocking actually pay during exchanges.
      kiter  - holds the spacing line by MOVEMENT alone: s-taps when crowded, sprints
               back onto the line when too far, weaves at max reach. Deliberately does
               NOT block - its defense is distance; blocking would kill its mobility.
    """

    def __init__(self, env, style):
        self.env = env
        self.style = style
        self.prev_tgt_hurt = 0
        self.wtap = 0
        self.tick = 0
        self.starved = 0   # turtle: consecutive band ticks with the target out of reach
        self.reclose = 0   # turtle: sprint-burst ticks left (block down to re-close)
        # boxer state (see the boxer branch)
        self.prev_my_h = None                      # own-health tracker (block-hit trigger)
        self.block_hold = 0                        # ticks of block-hit left
        self.rush = 0                              # aggression-burst ticks left
        self.next_rush = random.randint(60, 120)   # ticks until the next burst
        self.strafe_dir = random.choice((1, 2))    # sprint circle-left / circle-right
        self.strafe_left = random.randint(8, 20)   # ticks until the direction flips

    def act(self, obs_np, greedy=False):
        s = self.env.last_state or {}
        self.tick += 1
        d = s.get("target_dist") or -1.0
        tgt_hurt = s.get("tgt_hurt") or 0
        swing = self.tick % 2 == 0     # ~10 CPS for everyone
        move, click, block = 0, 0, 0
        if self.style == "turtle":
            # Gate on the THREAT band (learner reach ~3.4 + lead/knockback slop), not
            # own reach: the old in_reach<=3.5 gate meant a knocked-back turtle dropped
            # block and stopped clicking exactly while the learner's edge-of-reach
            # follow-ups landed - it read as a punching bag that only swings once it's
            # already being hit. Inside the band it is the both-buttons-held human:
            # walk in, block up, ~10 CPS (whiffed clicks are free in 1.8, so the hit
            # lands the first tick reach is entered; block-hitting = half damage in,
            # full out). Outside it, sprint sword-down to close (can't chase at sneak
            # speed - that mobility tradeoff is the style's real weakness, on purpose).
            # The threat band alone got FARMED once the learner found the counter:
            # sit at ~3.4 (just past the turtle's ~3 reach), poke, and let sneak
            # speed guarantee the turtle never closes - it read as "held block for
            # seconds and died swinging at air" (wr vs turtle hit 100%). A human
            # turtle answers edge-farming by dropping block and sprinting the gap
            # shut. So: track consecutive band ticks with the target OUT of own
            # reach ("starved"); past the threshold, burst in sword-down for a few
            # ticks, then resume turtling. Hysteresis matters - the threshold (8)
            # rides out ordinary knockback bounces, so this does NOT reintroduce
            # the old drop-block-while-being-hit punching-bag flap.
            if 0.0 < d <= 3.2:
                self.starved = 0                        # in reach: trades are landing
            elif 0.0 < d <= 4.25:
                self.starved += 1                       # blocking but can't reach
            if self.reclose > 0:
                self.reclose -= 1                       # sprint-burst: block down
                move, click = 0, int(swing and 0.0 < d <= 4.5)
            elif 0.0 < d <= 4.25:
                move, block, click = 3, 1, int(swing)
                if self.starved > 8:                    # edge-farmed: force the gap shut
                    self.reclose, self.starved = 6, 0
            else:
                move = 0                                # sprint in, sword down
        elif self.style == "rusher":
            if tgt_hurt > self.prev_tgt_hurt:           # landed a hit -> W-tap
                self.wtap = 2
            move = 3 if self.wtap > 0 else 0            # drop sprint briefly, then re-engage
            self.wtap = max(0, self.wtap - 1)
            # Swing through the approach (same idea as combat.py's SWING_RANGE): a
            # whiff costs a rusher nothing and connects the instant reach is entered.
            click = int(swing and 0.0 < d <= 4.5)
        elif self.style == "boxer":
            # The closest scripted approximation of a competent human, added when the
            # first three styles were fully farmed (wr_script ~86%, half of updates at
            # 100% = no gradient left in the scripted rounds). What it layers together:
            # circle-strafe at reach with RANDOM direction flips (the kiter's fixed
            # (tick//8) weave phase is memorizable; a random timer is not), W-tap after
            # landing a hit, a short block-hit after TAKING one (proactive, through the
            # follow-up window), s-taps when crowded, and irregular sprint-in bursts so
            # its spacing rhythm can't be locked onto. Never jumps: staying grounded is
            # also what punishes a hop-spamming learner (airborne targets take full
            # knockback and lose sprint).
            my_h = s.get("my_health")
            took_hit = (my_h is not None and self.prev_my_h is not None
                        and my_h < self.prev_my_h)
            if my_h is not None:
                self.prev_my_h = my_h
            self.strafe_left -= 1
            if self.strafe_left <= 0:
                self.strafe_dir = 1 if self.strafe_dir == 2 else 2
                self.strafe_left = random.randint(8, 20)
            self.next_rush -= 1
            if self.next_rush <= 0 and self.rush <= 0:
                self.rush, self.next_rush = 12, random.randint(60, 120)
            if tgt_hurt > self.prev_tgt_hurt:
                self.wtap = 2                    # sprint-reset the hit just landed
            if took_hit and self.rush <= 0:
                self.block_hold = 6              # block-hit their follow-up window
            if self.wtap > 0:
                move = 3                         # W-tap: drop sprint, keep walking in
            elif self.rush > 0:
                move = 0                         # aggression burst: sprint straight in
            elif d > 3.6 or d <= 0.0:
                move = 0                         # off the spacing line: close it
            elif d < 2.2 and (self.tick % 30) < 5:
                move = 8                         # crowded: s-tap back out to reach
            else:
                move = self.strafe_dir           # circle at reach
            block = int(self.block_hold > 0 and 0.0 < d <= 4.0)
            click = int(swing and 0.0 < d <= (4.5 if self.rush > 0 else 3.6))
            self.wtap = max(0, self.wtap - 1)
            self.rush = max(0, self.rush - 1)
            self.block_hold = max(0, self.block_hold - 1)
        else:  # kiter
            if 0.0 < d < 2.6:
                move = 8                                # s-tap: kill closing momentum
            elif d > 3.4 or d <= 0.0:
                move = 0                                # sprint back onto the line
            else:
                move = 4 if (self.tick // 8) % 2 == 0 else 5   # weave at max reach
            click = int(swing and 0.0 < d <= 3.8)       # poke anything that crosses the line
            # A kiter's defense is SPACING, not block: blocking would drop it to sneak
            # speed and it could never hold the line, so it does NOT block (that would
            # make it a slow turtle). Blocking is the turtle's job.
        self.prev_tgt_hurt = tgt_hurt
        action = {"move": move, "click": click, "block": block,
                  "aim": np.zeros(ACTION_DIMS["aim"], dtype=np.float32),
                  "latch_block": False}   # scripted styles own their own block timing
        return action, 0.0, 0.0


# --- self-play harness ------------------------------------------------------
class SelfPlayHarness:
    """Two live clients stepped in lockstep. Exposes the learner's view only."""

    def __init__(self, learner_env, opp_env, spawn_wait_ticks=0):
        self.learner = learner_env
        self.opp = opp_env
        self.opp_policy = None
        self.spawn_wait_ticks = spawn_wait_ticks

    def set_opponent(self, policy):
        self.opp_policy = policy

    def reset(self):
        # Learner env issues the teleport/heal for BOTH fighters; opp just re-syncs.
        obs = self.learner.reset()
        self.opp_obs = self.opp.reset()
        # Did a HUMAN take over the opponent client this round (pressed P)? Detected
        # per tick in step() from the opp's ai_active telemetry; when true the round
        # is retagged "human" so it never pollutes a scripted style's winrate.
        self.opp_human = False
        # Cross-wire the simulated lags so each fighter's tgt_ping reports the
        # OTHER's delay (a round late on the reset frames themselves; those are
        # idle-teleport frames, so nothing meaningful reads it early).
        self.learner.peer_lag = self.opp.sim_lag
        self.opp.peer_lag = self.learner.sim_lag
        # Wait out spawn-protection invulnerability before the round starts (see
        # SPAWN_PROT_TICKS). Same send-both-then-recv-both order as step(), so the
        # two fighters stay in lockstep through the idle.
        for _ in range(self.spawn_wait_ticks):
            self.opp._send({})
            self.learner._send({})
            for env in (self.opp, self.learner):
                st = env._recv_state()
                env.last_state = st
                env.history.append(frame_features(st))
        if self.spawn_wait_ticks:
            # Re-baseline health and rebuild the obs from post-wait frames, so the
            # wait itself (heal effects settling, etc.) can't show up as reward.
            for env in (self.opp, self.learner):
                env.prev_my_health = env.last_state.get("my_health", env.prev_my_health)
                env.prev_tgt_health = env.last_state.get("tgt_health", env.prev_tgt_health)
            obs = self.learner._obs_vector()
            self.opp_obs = self.opp._obs_vector()
        return obs

    def step(self, learner_action):
        # Opponent SAMPLES from its frozen snapshot. Greedy argmax of an untrained
        # net is a degenerate constant action (one strafe key held forever, straight
        # off the arena); sampling reproduces how that snapshot actually behaved.
        if self.opp_policy is not None:
            opp_action, _, _ = self.opp_policy.act(self.opp_obs)
        else:
            opp_action = {"move": 0, "click": 0, "block": 0,
                          "aim": np.zeros(ACTION_DIMS["aim"], dtype=np.float32)}
        # A human-controlled opponent (P pressed) reports ai_active=false; the actions
        # we send it are ignored, so this round isn't really the tagged sparring style.
        # Absent key (old jar) defaults True, so nothing changes until the jar is rebuilt.
        if not (self.opp.last_state or {}).get("ai_active", True):
            self.opp_human = True
        # Dispatch BOTH fighters' actions before blocking on either recv, so they
        # share the same control latency. Sending+recving the opponent fully before
        # even sending the learner's action gave the learner an extra tick of loop
        # delay every step - which, at the same aim gain, made ONLY the learner's
        # crosshair overshoot/orbit (the opponent, dispatched first, tracked clean).
        self.opp.step_send(opp_action)
        self.learner.step_send(learner_action)
        self.opp_obs, _, opp_done, opp_info = self.opp.step_recv()
        obs, reward, done, info = self.learner.step_recv()
        # If either fighter died, the round is over for both. The two clients see
        # the same death on different ticks: the victim's own client gets its health
        # packet a tick or two before the killer's client sees the synced entity
        # health hit zero - and the mod insta-respawns, so the killer's env can miss
        # the zero-health frame entirely. When only the OPPONENT env saw the
        # terminal, credit the learner now; without this, those rounds pay no
        # KILL/DEATH reward and fall out of the winrate as result "?".
        if opp_done and not done:
            opp_result = opp_info.get("result")
            if opp_result == "death":       # opponent died -> we killed it
                info["result"] = "kill"
                reward += KILL_REWARD
            elif opp_result == "kill":      # opponent killed us
                info["result"] = "death"
                reward += DEATH_REWARD
            else:
                info.setdefault("result", opp_result or "timeout")
        return obs, reward, done or opp_done, info

    def resync(self):
        # After a pause (PPO update), drop any telemetry that piled up in the
        # sockets so we resume on the current frame, not a stale backlog.
        for env in (self.learner, self.opp):
            _drain(env)

    def close(self):
        self.learner.close()
        self.opp.close()


def _drain(env):
    """Non-blocking read of all buffered lines, keeping only the newest state."""
    latest = None
    while True:
        r, _, _ = select.select([env.conn], [], [], 0.0)
        if not r:
            break
        try:
            chunk = env.conn.recv(4096).decode("utf-8")
        except BlockingIOError:
            break
        if not chunk:
            break
        env._buffer += chunk
        while "\n" in env._buffer:
            line, env._buffer = env._buffer.split("\n", 1)
            if line.strip():
                latest = line
    if latest is not None:
        import json
        from features import frame_features
        env.last_state = env.annotate_ping(json.loads(latest))
        env.history.append(frame_features(env.last_state))


# --- PPO --------------------------------------------------------------------
def compute_gae(rewards, values, dones, last_value):
    adv = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_val = last_value if t == len(rewards) - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + GAMMA * next_val * next_nonterminal - values[t]
        gae = delta + GAMMA * GAE_LAMBDA * next_nonterminal * gae
        adv[t] = gae
    returns = adv + values
    return adv, returns


def ppo_update(model, optimizer, batch):
    obs = torch.as_tensor(np.array(batch["obs"]), dtype=torch.float32, device=DEVICE)
    move_a = torch.as_tensor(batch["move"], dtype=torch.long, device=DEVICE)
    click_a = torch.as_tensor(batch["click"], dtype=torch.long, device=DEVICE)
    block_a = torch.as_tensor(batch["block"], dtype=torch.long, device=DEVICE)
    aim_a = torch.as_tensor(np.array(batch["aim"]), dtype=torch.float32, device=DEVICE)
    old_logp = torch.as_tensor(batch["logprobs"], dtype=torch.float32, device=DEVICE)
    adv = torch.as_tensor(batch["adv"], dtype=torch.float32, device=DEVICE)
    returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=DEVICE)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    n = len(move_a)
    idx = np.arange(n)
    for _ in range(UPDATE_EPOCHS):
        np.random.shuffle(idx)
        for start in range(0, n, MINIBATCH):
            mb = idx[start:start + MINIBATCH]
            logp, entropy, values = model.evaluate(
                obs[mb], move_a[mb], click_a[mb], block_a[mb], aim_a[mb])
            ratio = torch.exp(logp - old_logp[mb])
            s1 = ratio * adv[mb]
            s2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv[mb]
            policy_loss = -torch.min(s1, s2).mean()
            value_loss = ((values - returns[mb]) ** 2).mean()
            loss = policy_loss + VF_COEF * value_loss - ENT_COEF * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
    return policy_loss.item(), value_loss.item()


def collect_rollout(harness, model, n_steps, state):
    """Run the env for n_steps, returning a flat batch. `state` carries obs across
    rollout boundaries so episodes aren't cut artificially."""
    obs_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], []
    move_buf, click_buf, block_buf, aim_buf = [], [], [], []
    obs = state["obs"]
    ep_reward, ep_results = state.get("ep_reward", 0.0), []
    total_dealt = total_taken = 0.0
    combo_at_hit = []   # chain length on each tick a hit lands (see info["combo"])
    stale_vals = []     # mod-reported control-loop lag (healthy 1, spin-phase 2)
    whiffs = click_evals = 0   # in-range clicks whose crosshair ray missed / total
    yaw_errs = []       # signed combat-range yaw error; mean < 0 = crosshair sits right
    jump_ticks = 0      # ticks the SCRIPTED crit-jump reflex fired (info["jumped"])

    for _ in range(n_steps):
        action, logp, value = model.act(obs)
        next_obs, reward, done, info = harness.step(action)
        total_dealt += info.get("dealt", 0.0)
        total_taken += info.get("taken", 0.0)
        if info.get("dealt", 0.0) > 0.0:
            combo_at_hit.append(info.get("combo", 0))
        st = info.get("staleness", -1)
        if st is not None and st > 0:
            stale_vals.append(st)
        if info.get("click_eval"):
            click_evals += 1
            whiffs += 1 if info.get("whiff") else 0
        if info.get("yaw_err") is not None:
            yaw_errs.append(info["yaw_err"])
        if info.get("jumped"):
            jump_ticks += 1

        obs_buf.append(obs)
        move_buf.append(action["move"]); click_buf.append(action["click"])
        block_buf.append(action["block"])
        aim_buf.append(action["aim"])
        logp_buf.append(logp)
        rew_buf.append(reward); done_buf.append(float(done)); val_buf.append(value)
        ep_reward += reward

        if done:
            # Tag the result with which snapshot we were fighting, so the log can
            # show winrate vs old selves separately from winrate vs recent selves.
            # A round a human hijacked (P on the opponent) is tagged "human" so it
            # can't be attributed to the scripted style that was NOMINALLY scheduled.
            tag = ("human" if getattr(harness, "opp_human", False)
                   else getattr(harness, "opp_tag", "?"))
            ep_results.append((info.get("result", "?"), ep_reward, tag,
                               getattr(harness, "pinned", False)))
            ep_reward = 0.0
            # New round: resample opponent from the pool happens in the caller
            obs = state["on_episode_end"]()
        else:
            obs = next_obs

    _, _, last_value = model.act(obs)
    state["obs"] = obs
    state["ep_reward"] = ep_reward
    adv, returns = compute_gae(
        np.array(rew_buf, dtype=np.float32),
        np.array(val_buf, dtype=np.float32),
        np.array(done_buf, dtype=np.float32),
        last_value)
    return {
        "obs": obs_buf, "move": move_buf, "click": click_buf, "block": block_buf,
        "aim": aim_buf, "logprobs": logp_buf,
        "adv": adv, "returns": returns,
    }, ep_results, {
        "dealt": total_dealt, "taken": total_taken,
        "hits": len(combo_at_hit),
        "mean_combo": float(np.mean(combo_at_hit)) if combo_at_hit else 0.0,
        "max_combo": max(combo_at_hit) if combo_at_hit else 0,
        "staleness": float(np.mean(stale_vals)) if stale_vals else float("nan"),
        "whiff": whiffs / max(click_evals, 1),
        "aim_bias": float(np.mean(yaw_errs)) if yaw_errs else float("nan"),
        # Fraction of rollout ticks the scripted crit-jump fired (was the learned
        # 'jmp' head rate; kept under the same CSV column so plot_progress still works).
        "jump_rate": jump_ticks / max(n_steps, 1),
    }


def main():
    learner_env = PvPEnv(port=LEARNER_PORT, reset_commands=reset_commands,
                         sim_lag_range=SIM_LAG_RANGE)
    opp_env = PvPEnv(port=OPPONENT_PORT, reset_commands=[],
                     sim_lag_range=SIM_LAG_RANGE)
    print("Start the LEARNER client, then the OPPONENT client (they connect in order).")
    learner_env.connect()
    opp_env.connect()
    harness = SelfPlayHarness(learner_env, opp_env,
                              spawn_wait_ticks=SPAWN_PROT_TICKS)

    model = ActorCritic().to(DEVICE)
    warm_start_from_bc(model)

    def load_adapted(path):
        """Load a self-play checkpoint, remapping older layouts (frame dim, move dim)
        onto the current architecture and DROPPING any jump_head.* keys - the jump head
        was removed when crit-jump became a scripted reflex, so every checkpoint on disk
        (BC clone, best, latest, pool snapshots) still carries it. The remaining keys
        (base + move/click/block/aim/critic) match the current jumpless model exactly."""
        sd = adapt_ckpt_move_dim(
            adapt_ckpt_frame_dim(torch.load(path, map_location=DEVICE)),
            ACTION_DIMS["move"])
        return {k: v for k, v in sd.items() if not k.startswith("jump_head")}

    # Startup priority: resume a live run > start from the cloned policy > fall back to
    # the trunk-only BC floor. Resuming keeps a run going across reward tweaks and
    # crashes. Absent that, pvp_selfplay_bc.pth (train_bc.py) is a policy whose
    # move/click/block heads are cloned from human play, so PPO fine-tunes a bot that
    # already blocks like a person - instead of exploring up from random heads into the
    # no-block corner. (Jump is scripted, not cloned; a pre-removal bc.pth still loads -
    # load_adapted drops its jump_head.) warm_start_from_bc above already seeded the
    # trunk, so if neither checkpoint exists we still start there.
    try:
        model.load_state_dict(load_adapted("pvp_selfplay_latest.pth"))
        print("Resumed from pvp_selfplay_latest.pth")
    except FileNotFoundError:
        try:
            model.load_state_dict(load_adapted("pvp_selfplay_bc.pth"))
            print("Started from pvp_selfplay_bc.pth (all decision heads cloned from "
                  "human play).")
        except FileNotFoundError:
            print("No self-play/BC checkpoint - starting from the trunk-only BC warm start.")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Opponent pool: (tag, weights) so results can be split by snapshot age.
    # Seed with BEST alongside the resume point: a restart otherwise spars only
    # against one frozen self until the first snapshot lands, and best is the
    # strongest partner on disk (it also can't be weaker than start by much, so
    # it never dilutes the pool the way an arbitrary old checkpoint could).
    pool = [("start", copy.deepcopy(model.state_dict()))]
    # Also seed a spread of DISK snapshots (oldest / middle / newest). Restarts are
    # frequent in practice (14 in the metrics file), and each one used to reset the
    # pool to start+best - both ~= the resume point, so the learner overfit to
    # sparring near-copies of itself within hours. Old snapshots put yesterday's
    # styles back in the mix. Inserted oldest-first, BEFORE best, so the pool's
    # oldest->newest ordering (which the wr_old/wr_new split reads) stays honest.
    snaps = sorted(glob.glob(os.path.join("pool_snapshots", "*.pth")),
                   key=os.path.getmtime)
    for i in (sorted({0, len(snaps) // 2, len(snaps) - 1}) if snaps else []):
        try:
            pool.append((os.path.splitext(os.path.basename(snaps[i]))[0],
                         load_adapted(snaps[i])))
            print(f"Seeded pool with {snaps[i]}")
        except Exception as e:
            print(f"Could not seed {snaps[i]}: {e}")
    try:
        pool.append(("best", load_adapted("pvp_selfplay_best.pth")))
        print("Seeded pool with pvp_selfplay_best.pth")
    except FileNotFoundError:
        pass
    except Exception as e:
        # A stale-architecture best.pth (e.g. pre-jump-head-removal that somehow
        # doesn't remap) must not crash startup - skip it like the snapshot loop does.
        print(f"Could not seed pvp_selfplay_best.pth: {e}")
    opponent = ActorCritic().to(DEVICE)

    def set_opponent_from(entry):
        tag, weights = entry
        opponent.load_state_dict(weights)
        opponent.eval()
        harness.set_opponent(opponent)
        harness.opp_tag = tag

    def new_episode():
        # Sample a sparring partner: mostly past selves, but SCRIPTED_PROB of rounds
        # against a rule-driven personality (fresh instance each round so its little
        # bits of state - wtap timer, click phase - start clean).
        if random.random() < SCRIPTED_PROB:
            style = random.choices(SCRIPTED_STYLES, weights=SCRIPTED_WEIGHTS)[0]
            harness.set_opponent(ScriptedOpponent(opp_env, style))
            harness.opp_tag = style
        else:
            style = None
            set_opponent_from(random.choice(pool))

        # Spawn geometry for the round. WALL_PIN_PROB of rounds are a pin; bias WHO
        # is pinned by opponent style: pin the TURTLE so the learner is the aggressor
        # who must learn to disengage instead of trading into a wall-blocker; pin the
        # LEARNER against a rusher so it practices fighting out of a corner; else
        # coin-flip. Assign learner_env.reset_commands (opp_env issues none); PvPEnv
        # calls it each reset.
        if random.random() < WALL_PIN_PROB:
            tag = random.choice(list(PIN_SPAWNS))
            if style == "turtle":
                pin_learner = False
            elif style == "rusher":
                pin_learner = True
            else:
                pin_learner = random.random() < 0.5
            learner_env.reset_commands = lambda t=tag, p=pin_learner: wall_pin_spawn(t, p)
            harness.pinned = True
        else:
            learner_env.reset_commands = open_spawn
            harness.pinned = False
        return harness.reset()

    state = {"obs": harness.reset(), "on_episode_end": new_episode}
    set_opponent_from(pool[0])
    harness.pinned = False   # first round uses the constructor's open spawn

    best_avg = -1e9
    recent_avg = deque(maxlen=5)   # best = best 5-update MEAN, so one lucky rollout
                                   # right after a restart can't clobber best.pth
    snap_wins = snap_rounds = 0    # winrate since last snapshot, gates pool entry
    for update in range(1, TOTAL_UPDATES + 1):
        batch, results, dmg = collect_rollout(harness, model, ROLLOUT_STEPS, state)
        p_loss, v_loss = ppo_update(model, optimizer, batch)
        harness.resync()

        if results:
            wins = sum(1 for r, _, _, _ in results if r == "kill")
            timeouts = sum(1 for r, _, _, _ in results if r == "timeout")
            avg_r = np.mean([er for _, er, _, _ in results])
            # Winrate split by opponent age: the pool is ordered oldest->newest, so
            # beating the older half >50% while staying ~50% vs the newer half is
            # the signature of real improvement (both sides of a mirror can't beat
            # each other, but the present should beat the past).
            tag_order = [t for t, _ in pool]
            half = len(tag_order) / 2.0
            old_res = [r for r, _, t, _ in results
                       if t in tag_order and tag_order.index(t) < half]
            new_res = [r for r, _, t, _ in results
                       if t in tag_order and tag_order.index(t) >= half]
            scr_res = [r for r, _, t, _ in results if t in SCRIPTED_STYLES]
            pin_res = [r for r, _, _, p in results if p]
            def wr(rs):
                return (f"{sum(1 for r in rs if r == 'kill')/len(rs):4.0%}"
                        if rs else "  - ")
            def wr_num(rs):
                return (sum(1 for r in rs if r == "kill") / len(rs)
                        if rs else float("nan"))
            # blk/clk: fraction of ticks holding block / swinging - the style dials
            # the shaping is trying to move. dmg: total dealt/taken this rollout -
            # the ground truth under winrate (which timeouts drag below 50%).
            blk = float(np.mean(batch["block"]))
            clk = float(np.mean(batch["click"]))
            jmp = dmg["jump_rate"]   # ticks the SCRIPTED crit-jump reflex fired (jump
                                     # is no longer a learned head - kept as 'jmp' so
                                     # the CSV/plot columns stay put)
            # Exploration / aim-head introspection (see METRICS_HEADER comment). The
            # jump head is gone; ent_jump stays in the CSV as a dead 0.0 column so the
            # header and plot_progress.py don't need touching.
            ent_move, ent_click, ent_block = model.head_entropies(
                torch.as_tensor(np.array(batch["obs"]), dtype=torch.float32,
                                device=DEVICE))
            ent_jump = 0.0
            aim_std = float(torch.exp(model.aim_logstd.detach().clamp(-2.5, 0.0)).mean())
            res = np.array(batch["aim"], dtype=np.float32)
            res_yaw, res_pitch = float(res[:, 0].mean()), float(res[:, 1].mean())
            mv_counts = np.bincount(batch["move"], minlength=ACTION_DIMS["move"])
            move_top = int(mv_counts.argmax())
            move_top_frac = float(mv_counts.max() / max(mv_counts.sum(), 1))
            per_style = {st: wr_num([r for r, _, t, _ in results if t == st])
                         for st in SCRIPTED_STYLES}
            print(f"upd {update:4d} | rounds {len(results):3d} ({timeouts} tmo) | "
                  f"winrate {wins/len(results):4.0%} (old {wr(old_res)} new {wr(new_res)} "
                  f"scr {wr(scr_res)} pin {wr(pin_res)}) | "
                  f"dmg +{dmg['dealt']:3.0f}/-{dmg['taken']:3.0f} | "
                  f"blk {blk:4.0%} clk {clk:4.0%} jmp {jmp:4.0%} | "
                  f"whiff {dmg['whiff']:4.0%} bias {dmg['aim_bias']:+5.1f} "
                  f"res {res_yaw:+4.1f} | ent {ent_move:.2f} | "
                  f"avg reward {avg_r:+6.2f} | ploss {p_loss:+.3f} vloss {v_loss:.3f} | "
                  f"lat {dmg['staleness']:.2f}")
            # Per-opponent W-L (kills-deaths from the LEARNER's side), so a single
            # dominant sparring partner is identifiable by tag instead of showing
            # up only as an anonymous dip in the aggregate winrate.
            by_tag = {}
            for r, _, t, _ in results:
                by_tag.setdefault(t, []).append(r)
            print("           vs " + "  ".join(
                f"{t} {sum(1 for r in rs if r == 'kill')}-"
                f"{sum(1 for r in rs if r == 'death')}"
                for t, rs in sorted(by_tag.items())))
            append_metrics({
                "update": update, "rounds": len(results), "timeouts": timeouts,
                "winrate": wins / len(results),
                "wr_old": wr_num(old_res), "wr_new": wr_num(new_res),
                "wr_script": wr_num(scr_res), "wr_pin": wr_num(pin_res),
                "dealt": dmg["dealt"], "taken": dmg["taken"], "hits": dmg["hits"],
                "mean_combo": dmg["mean_combo"], "max_combo": dmg["max_combo"],
                "blk": blk, "clk": clk, "jmp": jmp,
                "whiff": dmg["whiff"], "aim_bias": dmg["aim_bias"],
                "avg_reward": avg_r,
                "ploss": p_loss, "vloss": v_loss, "staleness": dmg["staleness"],
                "wr_turtle": per_style["turtle"], "wr_rusher": per_style["rusher"],
                "wr_kiter": per_style["kiter"], "wr_boxer": per_style["boxer"],
                "ent_move": ent_move, "ent_click": ent_click,
                "ent_block": ent_block, "ent_jump": ent_jump,
                "aim_std": aim_std, "res_yaw": res_yaw, "res_pitch": res_pitch,
                "move_top": move_top, "move_top_frac": move_top_frac,
            })
            recent_avg.append(avg_r)
            # Divergence firewall. With the reward now bounded (pvp_env REWARD_CLIP) a
            # healthy value loss stays well under 100; the historic collapse ran it into
            # the millions while every action head saturated to "hold jump+block". A
            # spike past this ceiling (or a NaN) means training has detonated - stop
            # immediately, BEFORE any save, so the clean best/latest checkpoints and the
            # pool snapshots are never overwritten with poison. Better to wake to a
            # halted run at a good checkpoint than to hours of a broken one.
            if not np.isfinite(v_loss) or not np.isfinite(avg_r) or v_loss > 300:
                print(f"  !! DIVERGENCE DETECTED (vloss {v_loss:.0f}, avg_r {avg_r:+.0f}) "
                      f"- aborting without saving; best/latest/pool preserved")
                break
            # Snapshot gate counts SNAPSHOT rounds only: early losses to the scripted
            # styles (expected - the turtle beats today's policy by design) must not
            # freeze pool growth, which is gated on beating one's own past.
            pool_res = [r for r, _, t, _ in results if t in tag_order]
            snap_wins += sum(1 for r in pool_res if r == "kill")
            snap_rounds += len(pool_res)
            if len(recent_avg) == recent_avg.maxlen:
                smoothed = float(np.mean(recent_avg))
                if smoothed > best_avg:
                    best_avg = smoothed
                    torch.save(model.state_dict(), "pvp_selfplay_best.pth")

        if update % SNAPSHOT_EVERY == 0:
            # Gate pool entry on winrate since the last snapshot: a degrading policy
            # must not keep seeding the pool with weaker sparring partners (that
            # feedback loop is how a collapse sustains itself for hours).
            wr_since = snap_wins / max(snap_rounds, 1)
            if wr_since >= 0.45:
                pool.append((f"u{update}", copy.deepcopy(model.state_dict())))
                if len(pool) > POOL_SIZE:
                    pool.pop(0)
                # Also persist the snapshot: pool entries otherwise live only in
                # this process, so a strong sparring partner (the kind worth
                # keeping) would vanish on pool rotation or a trainer restart.
                # Timestamped because the update counter restarts at 1 every run.
                os.makedirs("pool_snapshots", exist_ok=True)
                snap_path = f"pool_snapshots/{time.strftime('%b%d_%H%M')}_u{update}.pth"
                torch.save(model.state_dict(), snap_path)
                print(f"  snapshot added (pool size {len(pool)}) -> {snap_path}")
            else:
                print(f"  snapshot SKIPPED (winrate {wr_since:.0%} since last < 45%)")
            snap_wins = snap_rounds = 0
            torch.save(model.state_dict(), "pvp_selfplay_latest.pth")

    harness.close()


if __name__ == "__main__":
    main()
