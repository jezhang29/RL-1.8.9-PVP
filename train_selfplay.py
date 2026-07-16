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
import os
import random
import select
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from pvp_env import PvPEnv, OBS_DIM, ACTION_DIMS, KILL_REWARD, DEATH_REWARD
from train_model import PvPCloner, adapt_ckpt_frame_dim, adapt_ckpt_move_dim

# Simulated-latency range (ticks of action delay, ~50ms RTT each) for BOTH fighters,
# resampled per round - see pvp_env.SIM_LAG_MS_PER_TICK. 0-4 covers LAN through
# ~200ms, the range friends-over-internet actually shows up with.
SIM_LAG_RANGE = (0, 4)

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
SNAPSHOT_EVERY = 25       # updates between adding the current policy to the pool
POOL_SIZE = 10

LEARNER_PORT = 9999
OPPONENT_PORT = 9998

# One row per update, appended across restarts (resumed runs keep extending the same
# file). plot_progress.py turns this into the over-time graph. Header written once.
METRICS_CSV = "selfplay_metrics.csv"
METRICS_HEADER = ["update", "rounds", "timeouts", "winrate", "wr_old", "wr_new",
                  "dealt", "taken", "hits", "mean_combo", "max_combo",
                  "blk", "clk", "avg_reward", "ploss", "vloss", "staleness"]


def append_metrics(row, path=METRICS_CSV):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRICS_HEADER)
        if new_file:
            w.writeheader()
        w.writerow(row)

# Chat commands that reset a round. EDIT the player names to match your accounts.
# Semicolons are split by the mod into separate commands.
# The /effect heals the round's SURVIVOR back to full (the loser respawns at full
# anyway, so it's redundant-but-harmless for them). 1.8.9 /effect only accepts
# NUMERIC ids - 6 = instant_health; the minecraft:instant_health form is 1.9+ and
# just errors in chat.
# 23 = saturation: refills hunger so sprint (needs >6 food) never dies out and nobody
# starves during a long run.
#
# Spawns are JITTERED each round. Fixed spawns make every opening an exact mirror,
# so who lands the first hit is a coin flip and the policy only ever sees one
# approach geometry. Small random offsets vary distance (6-14 blocks) and angle
# while staying near the original pad; the cap keeps spawn distance under the
# 16-block feature range. Widen X_JITTER/Z_JITTER if the arena allows.
X_JITTER = 3
Z_JITTER = 2


def reset_commands():
    ax = random.randint(-X_JITTER, X_JITTER)
    az = -2 + random.randint(-Z_JITTER, Z_JITTER)
    bx = random.randint(-X_JITTER, X_JITTER)
    bz = 8 + random.randint(-Z_JITTER, Z_JITTER)
    return [
        f"/tp SantiagoSea {ax} 65 {az}",
        f"/tp Bee__Bot {bx} 65 {bz}",
        "/effect SantiagoSea 6 1 5",
        "/effect Bee__Bot 6 1 5",
        "/effect SantiagoSea 23 1 9",
        "/effect Bee__Bot 23 1 9",
    ]


# --- network ----------------------------------------------------------------
class ActorCritic(nn.Module):
    """Multi-head actor-critic. One shared feature trunk feeds four action heads
    (movement, click, block, and a continuous aim residual) plus the value head.
    The heads are treated as INDEPENDENT given the state, so the joint log-prob is
    the sum of the per-head log-probs and the joint entropy the sum of entropies -
    standard factorized-policy PPO.
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
                move, click, block = move_l.argmax(-1), click_l.argmax(-1), block_l.argmax(-1)
                aim = aim_m
            else:
                move, click, block, aim = md.sample(), cd.sample(), bd.sample(), ad.sample()
            logp = (md.log_prob(move) + cd.log_prob(click) + bd.log_prob(block)
                    + ad.log_prob(aim).sum(-1))
        action = {
            "move": int(move.item()), "click": int(click.item()),
            "block": int(block.item()),
            "aim": aim.squeeze(0).cpu().numpy().astype(np.float32),
        }
        return action, float(logp.item()), float(value.item())

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


# --- self-play harness ------------------------------------------------------
class SelfPlayHarness:
    """Two live clients stepped in lockstep. Exposes the learner's view only."""

    def __init__(self, learner_env, opp_env):
        self.learner = learner_env
        self.opp = opp_env
        self.opp_policy = None

    def set_opponent(self, policy):
        self.opp_policy = policy

    def reset(self):
        # Learner env issues the teleport/heal for BOTH fighters; opp just re-syncs.
        obs = self.learner.reset()
        self.opp_obs = self.opp.reset()
        # Cross-wire the simulated lags so each fighter's tgt_ping reports the
        # OTHER's delay (a round late on the reset frames themselves; those are
        # idle-teleport frames, so nothing meaningful reads it early).
        self.learner.peer_lag = self.opp.sim_lag
        self.opp.peer_lag = self.learner.sim_lag
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

        obs_buf.append(obs)
        move_buf.append(action["move"]); click_buf.append(action["click"])
        block_buf.append(action["block"]); aim_buf.append(action["aim"])
        logp_buf.append(logp)
        rew_buf.append(reward); done_buf.append(float(done)); val_buf.append(value)
        ep_reward += reward

        if done:
            # Tag the result with which snapshot we were fighting, so the log can
            # show winrate vs old selves separately from winrate vs recent selves.
            ep_results.append((info.get("result", "?"), ep_reward,
                               getattr(harness, "opp_tag", "?")))
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
        "aim": aim_buf, "logprobs": logp_buf, "adv": adv, "returns": returns,
    }, ep_results, {
        "dealt": total_dealt, "taken": total_taken,
        "hits": len(combo_at_hit),
        "mean_combo": float(np.mean(combo_at_hit)) if combo_at_hit else 0.0,
        "max_combo": max(combo_at_hit) if combo_at_hit else 0,
        "staleness": float(np.mean(stale_vals)) if stale_vals else float("nan"),
    }


def main():
    learner_env = PvPEnv(port=LEARNER_PORT, reset_commands=reset_commands,
                         sim_lag_range=SIM_LAG_RANGE)
    opp_env = PvPEnv(port=OPPONENT_PORT, reset_commands=[],
                     sim_lag_range=SIM_LAG_RANGE)
    print("Start the LEARNER client, then the OPPONENT client (they connect in order).")
    learner_env.connect()
    opp_env.connect()
    harness = SelfPlayHarness(learner_env, opp_env)

    model = ActorCritic().to(DEVICE)
    warm_start_from_bc(model)
    # Resume from the last self-play checkpoint if there is one, so restarting the
    # trainer (reward tweaks, crashes) continues the run instead of starting over.
    try:
        model.load_state_dict(adapt_ckpt_move_dim(adapt_ckpt_frame_dim(
            torch.load("pvp_selfplay_latest.pth", map_location=DEVICE)),
            ACTION_DIMS["move"]))
        print("Resumed from pvp_selfplay_latest.pth")
    except FileNotFoundError:
        print("No self-play checkpoint - starting from the BC warm start.")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Opponent pool: (tag, weights) so results can be split by snapshot age
    pool = [("start", copy.deepcopy(model.state_dict()))]
    opponent = ActorCritic().to(DEVICE)

    def set_opponent_from(entry):
        tag, weights = entry
        opponent.load_state_dict(weights)
        opponent.eval()
        harness.set_opponent(opponent)
        harness.opp_tag = tag

    def new_episode():
        # Sample a sparring partner from the pool at the start of each round
        set_opponent_from(random.choice(pool))
        return harness.reset()

    state = {"obs": harness.reset(), "on_episode_end": new_episode}
    set_opponent_from(pool[0])

    best_avg = -1e9
    recent_avg = deque(maxlen=5)   # best = best 5-update MEAN, so one lucky rollout
                                   # right after a restart can't clobber best.pth
    snap_wins = snap_rounds = 0    # winrate since last snapshot, gates pool entry
    for update in range(1, TOTAL_UPDATES + 1):
        batch, results, dmg = collect_rollout(harness, model, ROLLOUT_STEPS, state)
        p_loss, v_loss = ppo_update(model, optimizer, batch)
        harness.resync()

        if results:
            wins = sum(1 for r, _, _ in results if r == "kill")
            timeouts = sum(1 for r, _, _ in results if r == "timeout")
            avg_r = np.mean([er for _, er, _ in results])
            # Winrate split by opponent age: the pool is ordered oldest->newest, so
            # beating the older half >50% while staying ~50% vs the newer half is
            # the signature of real improvement (both sides of a mirror can't beat
            # each other, but the present should beat the past).
            tag_order = [t for t, _ in pool]
            half = len(tag_order) / 2.0
            old_res = [r for r, _, t in results
                       if t in tag_order and tag_order.index(t) < half]
            new_res = [r for r, _, t in results
                       if t in tag_order and tag_order.index(t) >= half]
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
            print(f"upd {update:4d} | rounds {len(results):3d} ({timeouts} tmo) | "
                  f"winrate {wins/len(results):4.0%} (old {wr(old_res)} new {wr(new_res)}) | "
                  f"dmg +{dmg['dealt']:3.0f}/-{dmg['taken']:3.0f} | blk {blk:4.0%} clk {clk:4.0%} | "
                  f"avg reward {avg_r:+6.2f} | ploss {p_loss:+.3f} vloss {v_loss:.3f} | "
                  f"lat {dmg['staleness']:.2f}")
            append_metrics({
                "update": update, "rounds": len(results), "timeouts": timeouts,
                "winrate": wins / len(results),
                "wr_old": wr_num(old_res), "wr_new": wr_num(new_res),
                "dealt": dmg["dealt"], "taken": dmg["taken"], "hits": dmg["hits"],
                "mean_combo": dmg["mean_combo"], "max_combo": dmg["max_combo"],
                "blk": blk, "clk": clk, "avg_reward": avg_r,
                "ploss": p_loss, "vloss": v_loss, "staleness": dmg["staleness"],
            })
            recent_avg.append(avg_r)
            snap_wins += wins
            snap_rounds += len(results)
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
                print(f"  snapshot added (pool size {len(pool)})")
            else:
                print(f"  snapshot SKIPPED (winrate {wr_since:.0%} since last < 45%)")
            snap_wins = snap_rounds = 0
            torch.save(model.state_dict(), "pvp_selfplay_latest.pth")

    harness.close()


if __name__ == "__main__":
    main()
