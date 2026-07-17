"""Behavioral cloning of FULL human PvP play into the self-play policy.

The recorder (server.py, 'O' in-game) already logs every human input - WASD, sprint,
jump, both mouse buttons - next to world state, into pvp_dataset_v3.csv. But
train_model.py only ever cloned MOVEMENT, and into a SEPARATE PvPCloner net whose
weights warm-started just the self-play TRUNK. The actual decision heads of the
self-play policy (move/click/block) always started fresh - uniform-random move,
50/50 click/block - so there was no human prior holding them back, which is exactly why
self-play kept sliding into degenerate corners (zero blocking, spam-swing).

This clones all THREE discrete heads of the self-play ActorCritic - move, click, block
- straight from the recording, and saves a checkpoint (pvp_selfplay_bc.pth) in the
exact ActorCritic layout. train_selfplay.py picks it up as the STARTING policy (when
there's no live run to resume), so PPO fine-tunes a bot that already plays like the
demonstrations instead of exploring up from random behaviour. That's the imitation-first
pivot: at 20 steps/sec real time, one minute of good human play is worth thousands of
random exploration steps, and it directly teaches "what a real PvPer does".

Jump is NOT cloned: crit-jump is a scripted reflex now (combat.act_policy), not a policy
head, so there is nothing to clone it into - the human's jump column is ignored.

Aim is deliberately NOT cloned. The self-play aim is computed geometry + a bounded
+-4deg residual, and the geometry already tracks near-perfectly; a human's raw flicks
don't fit a +-4deg residual and cloning them adds noise for no gain. Leaving the aim
head at its zero init keeps aim == the computed angle (the safe floor), and RL learns
any small lead residual on top, exactly as it does today.

Usage:
    python server.py        # then press 'O' in-game to record; play good PvP
    python train_bc.py      # clone pvp_dataset_v3.csv -> pvp_selfplay_bc.pth
    python train_selfplay.py  # RL fine-tunes from the cloned policy
"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from features import frame_features, STACK, INPUT_DIM
from pvp_env import ACTIONS, ACTION_DIMS
from train_selfplay import ActorCritic, warm_start_from_bc, DEVICE

CSV_PATH = "pvp_dataset_v3.csv"
OUT_PATH = "pvp_selfplay_bc.pth"

# The movement vocabulary as a (9, 5) bit matrix, for nearest-match mapping of the raw
# human (w, a, s, d, sprint) each tick onto a primitive index. ACTIONS was designed
# around exactly these human key combos, so most frames hit a vocab entry exactly; a
# rare off-vocab combo (e.g. w+a+d) snaps to its closest Hamming neighbour.
_ACTION_BITS = np.array([[int(b) for b in a] for a in ACTIONS], dtype=np.int32)


def map_move(wasd_sprint):
    """(N, 5) raw key bits -> (N,) nearest ACTIONS index by Hamming distance."""
    d = (wasd_sprint[:, None, :] != _ACTION_BITS[None, :, :]).sum(axis=2)  # (N, 9)
    return d.argmin(axis=1).astype(np.int64)


class BCDataset(Dataset):
    """Stacked-frame observations paired with the human's composite action that tick.

    Reuses the same frame construction, contiguity gating and target-in-range filter
    as train_model.PvPDataset, but extracts the composite label (move, click, block)
    the self-play policy actually decides. The action the human is executing while
    observing state t is the button state logged at row t (standard BC alignment:
    a_t given obs_t)."""

    def __init__(self, csv_file):
        print(f"Loading {csv_file} ...")
        df = pd.read_csv(csv_file)
        rows = df.to_dict("records")

        # Some early recordings read the sprint KEYBIND, which double-tap-W sprinting
        # never presses, so out_sprint came out all-zero. Recover it from speed (1.8.9
        # walk cap ~0.215 b/t, sprint ~0.28) - same rule as train_model.
        if df["out_sprint"].mean() < 0.01:
            speed = np.hypot(df["my_vx"], df["my_vz"])
            df["out_sprint"] = (speed > 0.235).astype(int)
            print(f"  ! out_sprint was empty - recovered {df['out_sprint'].mean():.1%} "
                  f"of frames as sprinting from movement speed.")

        frames = np.array([frame_features(r) for r in rows], dtype=np.float32)
        ticks = df["tick"].values
        wasd_sprint = df[["out_w", "out_a", "out_s", "out_d", "out_sprint"]] \
            .values.astype(np.int32)
        move_lab = map_move(wasd_sprint)
        click_lab = df["out_left_click"].values.astype(np.int64)
        block_lab = df["out_right_click"].values.astype(np.int64)

        # contiguous[i] True iff row i+1 is the very next game tick, so a stack never
        # bridges a recording gap / new session.
        contiguous = np.diff(ticks) == 1

        X, M, C, B = [], [], [], []
        for t in range(STACK - 1, len(rows)):
            # Every gap inside the window [t-STACK+1 .. t] must be contiguous.
            if STACK > 1 and not contiguous[t - STACK + 1 : t].all():
                continue
            # Only learn from moments with a target actually in range (feature 0 is the
            # in-range flag) - out-of-range chasing is the computed rule's job, not the
            # combat policy's.
            if frames[t][0] != 1.0:
                continue
            X.append(frames[t - STACK + 1 : t + 1].flatten())
            M.append(move_lab[t]); C.append(click_lab[t]); B.append(block_lab[t])

        self.X = np.array(X, dtype=np.float32)
        self.M = np.array(M, dtype=np.int64)
        self.C = np.array(C, dtype=np.int64)
        self.B = np.array(B, dtype=np.int64)
        print(f"Dataset ready: {len(self.X)} in-range frames.")
        self._report_distribution()

    def _report_distribution(self):
        """Print what the human actually did - the ceiling on what BC can clone. If,
        say, block is ~0% here, the recording has no blocking to learn from and you
        need to record more of it, not tune the trainer."""
        move_frac = np.bincount(self.M, minlength=len(ACTIONS)) / max(len(self.M), 1)
        print("  human move mix: " + ", ".join(
            f"{i}:{f:.0%}" for i, f in enumerate(move_frac)))
        print(f"  human click {self.C.mean():.0%} | block {self.B.mean():.0%}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.M[i], self.C[i], self.B[i]


def class_weights(labels, n, cap):
    """Inverse-frequency class weights (most-common class -> 1.0, rarer -> up to cap).
    Without this, the rare-but-decisive actions (block, jump, s-tap, the retreats) get
    drowned by the common sprint-in+click and the clone learns 'never press them' -
    the exact collapse we're trying to prevent. Capped so a 0.5%-frequency action
    doesn't get a 200x boost and turn into a mash."""
    freq = np.bincount(labels, minlength=n).astype(np.float32) + 1.0
    w = np.clip(freq.max() / freq, 1.0, cap)
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def train():
    if not os.path.isfile(CSV_PATH):
        raise SystemExit(f"No {CSV_PATH}. Record play first: python server.py (press 'O').")

    ds = BCDataset(CSV_PATH)
    if len(ds) < 500:
        print(f"WARNING: only {len(ds)} frames - clone will be weak. Record more play.")

    # Validation split: a few contiguous BLOCKS spread evenly across the whole
    # recording, NOT the single newest tail. Two failure modes to avoid: a random
    # split leaks (consecutive ticks are near-identical, so val bleeds into train and
    # the score is meaninglessly high); a pure newest-tail makes the most recent
    # session - often a different opponent/day - the ENTIRE val set, so "best val loss"
    # fires on that distribution shift and saves a barely-trained epoch (recording MORE
    # data then paradoxically hurts the checkpoint). Blocks keep val leak-free (adjacent
    # frames are correlated only at the few block seams) while representing the whole
    # distribution, so "best val" reflects genuine fit.
    n = len(ds)
    n_blocks = 20
    val_blocks = {3, 9, 15}   # ~15%, spread across the timeline
    blocks = np.array_split(np.arange(n), n_blocks)
    val_idx = np.concatenate([blocks[b] for b in sorted(val_blocks)])
    train_idx = np.concatenate([blocks[b] for b in range(n_blocks) if b not in val_blocks])
    train_loader = DataLoader(Subset(ds, train_idx.tolist()), batch_size=256, shuffle=True)
    val_loader = DataLoader(Subset(ds, val_idx.tolist()), batch_size=512)
    print(f"Train {len(train_idx)} | val ({len(val_idx)}, blocks spread across the "
          f"recording) | device {DEVICE}")

    model = ActorCritic().to(DEVICE)
    # Seed the feature trunk from the existing movement BC (a working floor) so the
    # clone converges faster; every head is then trained here on top of it.
    warm_start_from_bc(model)

    w_move = class_weights(ds.M[train_idx], ACTION_DIMS["move"], cap=8.0)
    w_bin = {name: class_weights(lab[train_idx], 2, cap=5.0)
             for name, lab in (("click", ds.C), ("block", ds.B))}
    ce_move = nn.CrossEntropyLoss(weight=w_move)
    ce_click = nn.CrossEntropyLoss(weight=w_bin["click"])
    ce_block = nn.CrossEntropyLoss(weight=w_bin["block"])

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    epochs = 60
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def head_logits(obs):
        # _heads returns (move, click, block, aim_mean, value); we clone the first
        # three. aim_mean/value are left to their init (aim floor / PPO learns V).
        move_l, click_l, block_l, _, _ = model._heads(obs)
        return move_l, click_l, block_l

    best_val = float("inf")
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for X, M, C, B in train_loader:
            X = X.to(DEVICE); M = M.to(DEVICE); C = C.to(DEVICE); B = B.to(DEVICE)
            opt.zero_grad()
            ml, cl, bl = head_logits(X)
            loss = ce_move(ml, M) + ce_click(cl, C) + ce_block(bl, B)
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()

        model.eval()
        vloss, n = 0.0, 0
        acc = np.zeros(3)
        with torch.no_grad():
            for X, M, C, B in val_loader:
                X = X.to(DEVICE); M = M.to(DEVICE); C = C.to(DEVICE); B = B.to(DEVICE)
                ml, cl, bl = head_logits(X)
                vloss += (ce_move(ml, M) + ce_click(cl, C) + ce_block(bl, B)).item()
                bs = len(X)
                acc += bs * np.array([
                    (ml.argmax(-1) == M).float().mean().item(),
                    (cl.argmax(-1) == C).float().mean().item(),
                    (bl.argmax(-1) == B).float().mean().item()])
                n += bs
        vloss /= len(val_loader)
        acc /= max(n, 1)

        tag = ""
        if vloss < best_val:
            best_val = vloss
            torch.save(model.state_dict(), OUT_PATH)
            tag = " <- best, saved"
        if ep % 5 == 0 or ep == epochs - 1 or tag:
            print(f"epoch [{ep+1}/{epochs}] train {tot/len(train_loader):.3f} | "
                  f"val {vloss:.3f} | acc move {acc[0]:.0%} click {acc[1]:.0%} "
                  f"block {acc[2]:.0%}{tag}")

    print(f"\nBest val loss {best_val:.3f}. Saved cloned policy -> {OUT_PATH}")
    print("Now run:  python train_selfplay.py   (it starts from this policy when there "
          "is no pvp_selfplay_latest.pth to resume).")


if __name__ == "__main__":
    train()
