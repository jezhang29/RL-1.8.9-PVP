"""A/B measurement: does render framerate leak hits?

Hypothesis (see the camera-smoothing hack in MLMod.java): vanilla's attack raycast
(mc.objectMouseOver) is recomputed every RENDER frame from the INTERPOLATED look
vector, and the mod deliberately restores prevRotation to the pre-delta angle so the
camera sweeps smoothly. That sweep means a click serviced mid-interpolation raycasts
from an angle that trails the true tick angle - a whiff at 60 fps that lands at 20 fps.

Game logic is tick-locked at 20 Hz regardless of fps, so a FROZEN policy has the same
swing opportunities at any framerate; only hit REGISTRATION can differ. So run this
once with Minecraft capped at 60 fps and once at 20 fps and compare:

  hits / in-reach-swing   <- the money metric. If 20 fps > 60 fps, raycast lag is real.
  hits / 1000 ticks       <- same story, not conditioned on reach.
  swings / 1000 ticks     <- sanity check: should MATCH across fps (same policy).

  python measure_fps.py 60fps            # cap MC at 60, run, note the numbers
  python measure_fps.py 20fps            # cap MC at 20 (or vsync), run, compare
  python measure_fps.py 20fps 6000       # optional: longer run = tighter numbers

Each run appends a row to fps_measure.csv so the two sit side by side. Start the two
clients exactly as for training (learner first, then opponent).
"""
import sys
import csv
import os

import numpy as np
import torch

from pvp_env import PvPEnv
from combat import REACH
from train_selfplay import (ActorCritic, warm_start_from_bc, SelfPlayHarness,
                            RESET_COMMANDS, LEARNER_PORT, OPPONENT_PORT, DEVICE)

CHECKPOINT = "pvp_selfplay_latest.pth"
MEASURE_CSV = "fps_measure.csv"


def load_frozen(path):
    model = ActorCritic().to(DEVICE)
    warm_start_from_bc(model)          # harmless if the checkpoint fully overrides it
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 4000

    learner_env = PvPEnv(port=LEARNER_PORT, reset_commands=RESET_COMMANDS)
    opp_env = PvPEnv(port=OPPONENT_PORT, reset_commands=[])
    print(f"[{tag}] frozen A/B over {ticks} ticks (~{ticks/20/60:.1f} min). "
          f"Start the LEARNER client, then the OPPONENT client.")
    learner_env.connect()
    opp_env.connect()
    harness = SelfPlayHarness(learner_env, opp_env)

    model = load_frozen(CHECKPOINT)
    harness.set_opponent(model)        # both sides = the same frozen policy (mirror)
    print(f"[{tag}] loaded {CHECKPOINT}. Cap Minecraft's framerate NOW, then let it run.")

    swings = in_reach_swings = hits = rounds = kills = 0
    dealt = taken = 0.0
    combos = []
    stales = []   # mod-reported control-loop lag; 1 = healthy, ~2 = spin phase

    obs = harness.reset()
    for t in range(ticks):
        # Sample (not greedy): matches how the policy actually plays and how training
        # measures it. Over thousands of ticks the swing RATE converges to the same
        # expectation at any fps, which is exactly what makes the hit-rate comparable.
        action, _, _ = model.act(obs)
        dist = harness.learner.last_state.get("target_dist", -1.0) or -1.0
        if action["click"]:
            swings += 1
            if 0.0 < dist <= REACH:
                in_reach_swings += 1
        obs, _, done, info = harness.step(action)
        if info.get("dealt", 0.0) > 0.0:
            hits += 1
            combos.append(info.get("combo", 0))
        st = info.get("staleness", -1)
        if st is not None and st > 0:
            stales.append(st)
        dealt += info.get("dealt", 0.0)
        taken += info.get("taken", 0.0)
        if done:
            rounds += 1
            if info.get("result") == "kill":
                kills += 1
            obs = harness.reset()
        if (t + 1) % 500 == 0:
            lat = float(np.mean(stales[-500:])) if stales else float("nan")
            print(f"  [{tag}] {t+1}/{ticks} ticks | swings {swings} hits {hits} "
                  f"| hit/in-reach {hits/max(in_reach_swings,1):.1%} | lat {lat:.2f}")

    harness.close()

    per_k = 1000.0 / max(ticks, 1)
    hit_per_reach = hits / max(in_reach_swings, 1)
    mean_combo = float(np.mean(combos)) if combos else 0.0
    row = {
        "tag": tag, "ticks": ticks, "rounds": rounds, "kills": kills,
        "swings": swings, "in_reach_swings": in_reach_swings, "hits": hits,
        "hit_per_reach_swing": round(hit_per_reach, 4),
        "hits_per_1k": round(hits * per_k, 2),
        "swings_per_1k": round(swings * per_k, 2),
        "mean_combo": round(mean_combo, 3),
        "dealt": round(dealt, 1), "taken": round(taken, 1),
    }
    header = list(row.keys())
    new_file = not os.path.exists(MEASURE_CSV)
    with open(MEASURE_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new_file:
            w.writeheader()
        w.writerow(row)

    print("\n" + "=" * 60)
    print(f"  RESULT [{tag}]")
    print(f"  hits / in-reach swing : {hit_per_reach:6.1%}   <- compare across fps")
    print(f"  hits / 1000 ticks     : {hits*per_k:6.1f}")
    print(f"  swings / 1000 ticks   : {swings*per_k:6.1f}   (should MATCH across fps)")
    print(f"  mean combo            : {mean_combo:6.2f}")
    lat_all = float(np.mean(stales)) if stales else float("nan")
    print(f"  control-loop lat      : {lat_all:6.2f}   (1.0 healthy, ~2.0 = spin phase)")
    print(f"  dmg dealt / taken     : +{dealt:.0f} / -{taken:.0f}   ({rounds} rounds, {kills} kills)")
    print(f"  appended to {MEASURE_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
