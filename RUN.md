# Running the PvP bot

Two brains share one mod + one combat module ([combat.py](combat.py)):

- **BC bot** (`ai_server.py`): learned **movement**; aim + attack + W-tap + blockhit are
  computed rules (`combat.act`). This is the imitation-trained baseline.
- **Self-play bot** (`train_selfplay.py` → `ai_server.py selfplay`): learns the
  **situational judgment** — **movement** (engage/disengage/strafe/s-tap), **attack**,
  a **defensive-block posture**, and an **aim residual** on top of the computed angle
  (`combat.act_policy`). The mechanical, event-timed techniques stay **computed**, like
  aim, because their correct timing is a known function of an event: the base aim
  geometry, the **post-hit blockhit sprint-reset**, and the **crit-jump** (fired only in
  the victim's falling i-frame window). This is the one that can *surpass* hardcoded
  timings on the parts a policy can actually beat rules at.

**In-game keys:** `P` toggles the AI on/off · `O` toggles data recording.

> **After the latest mod rebuild, reinstall the jar** (`build/libs/ml-1.8.9-pvp-1.0.jar`)
> — it adds camera smoothing so AI aim renders at your real framerate instead of looking
> 20 fps. Body movement stays 20 TPS (Minecraft's tick rate; unavoidable).

---

## 0. Build the mod (do this after any change to `MLMod.java`)

VS Code's Java errors are false positives (Gradle 4.5) — the terminal build is truth.

```bash
cd ml-1.8.9-pvp
JAVA_HOME=$(/usr/libexec/java_home -v 1.8) ./gradlew build
```

Install `ml-1.8.9-pvp/build/libs/ml-1.8.9-pvp-1.0.jar` into your mods folder.

---

## 1. Run the current (behavioral-cloning) bot

```bash
python ai_server.py          # loads pvp_model_v2.pth, listens on :9999
```

Launch Minecraft, join a world, press **P**. Beyond 16 blocks it idles by design
(no training data out there — it just sprints to close the gap).

---

## 2. Record play and clone it into the policy (imitation-first)

At 20 steps/sec real time, self-play exploration is sample-starved and keeps settling
into degenerate corners (constant hopping, zero blocking). The highest-leverage fix is
to **clone good human play** into the policy first, then let RL fine-tune it. One minute
of you playing well is worth thousands of random exploration steps.

```bash
python server.py             # writes pvp_dataset_v3.csv while you play
```

Press **O** in-game to start/stop recording. Play **good** PvP — this is the ceiling on
what the clone can learn, so demonstrate the **situational judgment** you want: spacing,
when to commit vs disengage, **defensive blocking** to trade against a cornered opponent,
strafe, escape when low. Include chases from 10-16 blocks. You do **not** need to nail the
mechanical timings — the post-hit blockhit sprint-reset, W-tap and crit-jump are scripted
reflexes now, computed from the hit event, so the clone only needs your *decisions*. To
fight the AI on 9998 and record yourself on 9999, or vice-versa, use **P** to disable AI
on whichever client you're playing. Then:

```bash
python train_bc.py           # clones move/click/block from your play
                             #   -> pvp_selfplay_bc.pth  (a self-play starting policy)
```

It prints your actual move mix and click/block rates (if block reads ~0%, you didn't
record enough blocking — record more, don't tune the trainer) and per-head val accuracy.
`train_selfplay.py` then **starts from this policy** whenever there's no live run to
resume (see §3). Aim and jump are not cloned — aim stays the computed geometry floor, and
crit-jump is a scripted reflex.

> **Legacy path:** `python train_model.py` still trains the old movement-only
> `pvp_model_v2.pth`, which warm-starts only the self-play *trunk*. `train_bc.py`
> supersedes it (it clones the decision heads too); keep `pvp_model_v2.pth` around only
> as the trunk fallback.

---

## 3. Self-play RL (two clients) — "keep improving itself"

**Setup**
0. **Build the arena.** A flat ~20×20 platform ringed by a **2-block-high wall** (bots
   can't jump that) beats an open 40×40: no fall-damage deaths (which score as fake
   "kills"), and early random policies actually bump into each other instead of
   wandering for a minute. One-time world prep (in chat, as the host):
   ```
   /gamerule naturalRegeneration false   ← fights are decisive; the round-reset heals anyway
   /gamerule keepInventory true          ← swords survive death/respawn
   /gamerule doDaylightCycle false       ← optional: consistent lighting
   ```
   Give both accounts a sword. Update the `/tp` coordinates in `RESET_COMMANDS` to two
   spots ~6 blocks apart inside the arena.
1. Host a single-player world, **Open to LAN with cheats ON** (so `/tp` and `/effect` work).
2. Join it with **two** instances, both running the mod jar.
3. **Point the opponent client at port 9998.** Both clients run the same jar (default
   port 9999), so the SECOND client must be launched with the JVM arg
   `-Dmlpvp.port=9998`. The learner client needs no arg.
   - *Prism / MultiMC:* opponent instance → Edit → Settings → Java → JVM arguments →
     add `-Dmlpvp.port=9998`.
   - *Vanilla launcher:* Installations → opponent installation → More Options →
     JVM Arguments → append ` -Dmlpvp.port=9998`.
   Each client prints `Connected to Python AI Server on port 9999/9998!` — check it.
4. Edit the player names in `RESET_COMMANDS` near the top of
   [train_selfplay.py](train_selfplay.py) to match your two accounts.

**Run**
```bash
python train_selfplay.py     # learner on :9999, opponent on :9998
```
Connect the **learner** client first (grabs 9999), then the **opponent** (9998). A client
that starts before the trainer opens its port just retries every 5s. Leave it running
(overnight for a first result). It logs win-rate and average reward per update and
saves `pvp_selfplay_best.pth` / `pvp_selfplay_latest.pth`.

**Round lifecycle (automatic, no babysitting):** a round ends when someone's health
hits 0 (learner dies −10 / kills +10), or at 60 s (`max_ticks=1200`, timeout — this
also recovers a bot that wandered off the arena). The mod **auto-respawns** a dead
bot (no clicking the death screen), then `RESET_COMMANDS` teleports both back and
heals the survivor.

**Reward** (constants at the top of [pvp_env.py](pvp_env.py)): damage dealt − damage
taken − a time penalty, plus style shaping that is small next to a ~6-damage hit:

| Shaping | Constant(s) | Teaches |
|---|---|---|
| Range potential toward 2.8 blocks (charged for face-hugging too) | `IDEAL_RANGE`, `RANGE_COEF` | perfect spacing |
| Per-tick cost for blocking **beyond** `BLOCK_FREE_RANGE` (in-range blocking is free — reduction is server-side, so it must be held proactively through exchanges) | `BLOCK_COST`, `BLOCK_FREE_RANGE` | no momentum-killing block spam on approach |
| One-time signed bonus for first blood | `FIRST_HIT_BONUS` | win the entry |
| Escalating bonus for hits ≤1.25 s apart | `COMBO_BONUS`, `COMBO_WINDOW` | hold combos |
| Bonus ∝ target launch speed after our hit | `KB_COEF`, `KB_NORM` | sprint-reset (W-tap) |

**Startup priority:** the trainer loads, in order, (1) `pvp_selfplay_latest.pth` to
resume a live run, else (2) `pvp_selfplay_bc.pth` — the imitation-cloned policy from §2,
so RL fine-tunes a bot that already blocks like a human — else (3) the trunk-only BC
floor. So to start a **fresh run from your cloned play**, delete/rename
`pvp_selfplay_latest.pth` and it picks up `pvp_selfplay_bc.pth`. Stopping and restarting
mid-run always resumes from `latest`. Old checkpoints from before the jump head was
removed still load — `load_adapted` drops their `jump_head` weights automatically.

**What to expect early:** the first hour is supposed to look stupid — near-uniform
random movement, ~50% click spam, ~50% block spam. Aim starts ≈ the computed angle
(the residual head is zero-initialized). The policy updates **live** every 2048 ticks
(~2 min of play, the `upd N` log line) — no restart needed; the opponent client only
improves in steps (a snapshot of the learner joins its pool every 25 updates). Judge
progress by the **trend over hours** of `winrate` and `avg reward`, not by any single
round: reward is per-tick damage differential averaged over dozens of rounds per
update, so a lucky flail doesn't move it.

**Reading the log — are both bots improving?** Only one network trains; the opponent
plays frozen snapshots of it (up to ~250 updates old), so the opponent improves in
steps as snapshots rotate in. The log line splits winrate by snapshot age:
`winrate 58% (old 75% new 52%)`. Healthy learning = **old well above 50%** (the
present beats the past) while **new hovers near 50%** (a mirror can't beat itself).
Overall `avg reward` plateauing is normal and NOT stagnation — the opposition keeps
getting better under you.

**To fight the trained result:**

```bash
python ai_server.py selfplay                 # loads pvp_selfplay_best.pth
python ai_server.py selfplay pvp_selfplay_latest.pth   # or a specific checkpoint
```

Then join a world and press **P** — same as the BC bot, but now attack/block/W-tap/aim
are the policy's, not the rules'.

---

## Tuning combat feel ([combat.py](combat.py))

**Shared / self-play (`act_policy`):**

| Constant | Default | What it does |
|---|---|---|
| `AIM_GAIN` | 0.6 | Fraction of the *computed* aim error closed per tick (the learned residual rides on top) |
| `AIM_LEAD_TICKS` | 2.0 | Aim at where both fighters **will** be, cancelling the mod→Python→mod latency (fixes trailing/overshooting a strafing or jumping target; `tgt_vy` telemetry needs the jar from Jul 15+) |
| `AIM_RES_MAX` | 4.0 | Max magnitude (deg) of the learned aim residual — how far RL may bend the crosshair off the geometry |
| `BLOCK_MIN_TICKS` | 3 | Min hold once the policy chooses a **defensive** block — server-side reduction never registers 1-tick blips |
| `BLOCK_TICKS` | 3 | Length of the **scripted post-hit blockhit** pulse (sprint-reset) fired the tick the learner lands a hit |
| `CRIT_HURT_LO/HI` | 5 / 8 | Target hurtTime window the **scripted crit-jump** fires in (learner only) |
| `CRIT_RANGE` | 4.5 | Max distance to still chase a crit down |
| `CRIT_GAP` | 12 | Ticks between crit jumps |

In self-play the policy owns **click cadence** and the **defensive-block posture**, so
`CPS_FAST_PROB` and `SWING_*` don't apply — the per-tick rate (≤20 CPS) is the only cap.
But the **post-hit blockhit sprint-reset and crit-jump are scripted reflexes** now (the
Jul 17 re-hybridization), so `BLOCK_TICKS` and the `CRIT_*` constants above DO apply to
the learner, and there is no learned jump head. Early in training aim ≈ the computed
angle (residual starts near 0) and clicking is ~random; both sharpen as reward accrues.

**BC bot only (`act`, the rule combat):**

| Constant | Default | What it does |
|---|---|---|
| `CPS_FAST_PROB` | 0.75 | Click speed. `CPS = 20 / (2 - this)`: 0.75→16, 0.9→18, 1.0→20 |
| `SWING_RANGE` | 5.0 | Blocks out at which it starts swinging during approach |
| `SWING_YAW` | 35.0 | How far off-aim it'll still swing while closing |
| `WTAP_TICKS` | 1 | Ticks W+sprint drop after a hit (knockback reset) |
| `BLOCK_TICKS` | 3 | Ticks it sword-blocks after landing a hit |
