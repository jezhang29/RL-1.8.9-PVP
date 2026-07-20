"""Turn selfplay_metrics.csv into a self-contained progress graph.

No matplotlib, no install: reads the CSV the trainer appends to and writes
selfplay_progress.html with the data embedded and a tiny vanilla-JS SVG renderer.
Double-click the HTML to view; re-run this any time to refresh.

  python plot_progress.py                 # reads selfplay_metrics.csv
  python plot_progress.py other.csv out.html

WHICH CURVE MATTERS: in self-play both fighters share the policy, so raw winrate
sits near 50% by construction - it is NOT the improvement signal. The headline panel
is "winrate vs OLDER snapshots": the present self should beat its past selves >50%.
Net damage per round and combo length are the other ground-truth style/skill readouts.
"""
import sys
import csv
import json


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            def num(k):
                v = r.get(k, "")
                if v is None or v == "" or v == "nan":
                    return None
                try:
                    return float(v)
                except ValueError:
                    return None
            rows.append({k: num(k) for k in r})
    return rows


HTML = """<!doctype html>
<meta charset="utf-8">
<title>Self-play progress</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 14px/1.4 -apple-system, system-ui, sans-serif; margin: 0; padding: 24px;
         background:#fafafa; color:#1a1a1a; }
  @media (prefers-color-scheme: dark) { body { background:#15171a; color:#e6e6e6; } }
  h1 { font-size: 18px; margin: 0 0 4px; }
  .sub { opacity:.6; margin-bottom: 20px; }
  .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(420px,1fr)); gap:18px; }
  .card { background:canvas; border:1px solid color-mix(in srgb, currentColor 15%, transparent);
          border-radius:10px; padding:14px 14px 8px; }
  .card h2 { font-size:14px; margin:0 0 2px; }
  .card .note { font-size:12px; opacity:.6; margin:0 0 6px; }
  .legend { font-size:12px; margin-top:6px; display:flex; gap:14px; flex-wrap:wrap; }
  .legend span { display:inline-flex; align-items:center; gap:5px; }
  .swatch { width:12px; height:3px; border-radius:2px; display:inline-block; }
  svg { width:100%; height:200px; display:block; overflow:visible; }
  .controls { margin: 8px 0 18px; font-size:13px; }
</style>
<h1>Self-play progress</h1>
<div class="sub" id="sub"></div>
<div class="controls">
  smoothing window <input id="win" type="range" min="1" max="60" value="15"><span id="winv">15</span>
</div>
<div class="grid" id="grid"></div>
<script>
const DATA = __DATA__;
// The trainer logs one row per opponent/rollout, so `update` repeats many times.
// Collapse to a single point per update before plotting, otherwise the x-axis is
// non-monotonic and every series zig-zags back and forth into an illegible mess.
function aggregate(rows) {
  const byU = new Map();
  for (const r of rows) {
    if (!byU.has(r.update)) byU.set(r.update, []);
    byU.get(r.update).push(r);
  }
  const SUM = new Set(['dealt', 'taken', 'hits', 'timeouts', 'rounds', 'ties']);
  const out = [];
  for (const u of [...byU.keys()].sort((a, b) => a - b)) {
    const grp = byU.get(u);
    const rec = { update: u };
    const keys = new Set();
    for (const r of grp) for (const k of Object.keys(r)) keys.add(k);
    for (const k of keys) {
      if (k === 'update') continue;
      if (k === 'max_combo' || k === 'max_combo_taken') {
        let m = null;
        for (const r of grp) if (r[k] != null) m = m == null ? r[k] : Math.max(m, r[k]);
        rec[k] = m;
      } else if (SUM.has(k)) {
        let s = 0, any = false;
        for (const r of grp) if (r[k] != null) { s += r[k]; any = true; }
        rec[k] = any ? s : null;
      } else {
        // rounds-weighted mean over non-null values
        let s = 0, w = 0;
        for (const r of grp) {
          if (r[k] == null) continue;
          const wt = r.rounds != null && r.rounds > 0 ? r.rounds : 1;
          s += r[k] * wt; w += wt;
        }
        rec[k] = w ? s / w : null;
      }
    }
    out.push(rec);
  }
  return out;
}
const ROWS = aggregate(DATA);
const X = ROWS.map(d => d.update);

function rollmean(ys, w) {
  const out = ys.map((_, i) => {
    let s = 0, n = 0;
    for (let j = Math.max(0, i - w + 1); j <= i; j++) {
      if (ys[j] != null) { s += ys[j]; n++; }
    }
    return n ? s / n : null;
  });
  return out;
}

function chart(el, title, note, series, opts) {
  opts = opts || {};
  const W = 440, H = 200, PADL = 44, PADR = 12, PADT = 10, PADB = 22;
  const win = +document.getElementById('win').value;
  // optional log-y transform, for series with huge dynamic range (e.g. value loss)
  const T = opts.log ? (v => Math.log10(Math.max(v, 1e-9))) : (v => v);
  const Tinv = opts.log ? (t => Math.pow(10, t)) : (t => t);
  // autoscale over all (smoothed) series, in transformed space
  let lo = Infinity, hi = -Infinity;
  const smoothed = series.map(s => {
    const raw = ROWS.map(s.f);
    const sm = rollmean(raw, win);
    for (const v of sm) if (v != null) { const t = T(v); lo = Math.min(lo, t); hi = Math.max(hi, t); }
    return { ...s, raw, sm };
  });
  if (opts.y0 != null) lo = Math.min(lo, T(opts.y0));
  if (opts.y1 != null) hi = Math.max(hi, T(opts.y1));
  if (!isFinite(lo)) { lo = 0; hi = 1; }
  if (lo === hi) { hi = lo + 1; }
  const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
  const xmin = X[0], xmax = X[X.length - 1] || 1;
  const sx = u => PADL + (W - PADL - PADR) * ((u - xmin) / Math.max(1, xmax - xmin));
  const syT = t => PADT + (H - PADT - PADB) * (1 - (t - lo) / (hi - lo));
  const sy = v => syT(T(v));
  const fmtNum = v => (Math.abs(v) >= 1e6 ? (v / 1e6).toFixed(1) + 'M'
                     : Math.abs(v) >= 1e3 ? (v / 1e3).toFixed(1) + 'k'
                     : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2));
  const fmt = opts.pct ? (v => (v * 100).toFixed(0) + '%') : fmtNum;

  let svg = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">`;
  // gridlines + y labels
  for (let i = 0; i <= 4; i++) {
    const t = lo + (hi - lo) * i / 4, y = syT(t);
    svg += `<line x1="${PADL}" y1="${y}" x2="${W - PADR}" y2="${y}" stroke="currentColor" stroke-opacity=".08"/>`;
    svg += `<text x="${PADL - 6}" y="${y + 3}" text-anchor="end" font-size="10" fill="currentColor" fill-opacity=".55">${fmt(Tinv(t))}</text>`;
  }
  if (opts.ref != null) {
    const y = sy(opts.ref);
    svg += `<line x1="${PADL}" y1="${y}" x2="${W - PADR}" y2="${y}" stroke="currentColor" stroke-opacity=".35" stroke-dasharray="4 4"/>`;
  }
  // x labels (first / last update)
  svg += `<text x="${PADL}" y="${H - 6}" font-size="10" fill="currentColor" fill-opacity=".55">upd ${xmin}</text>`;
  svg += `<text x="${W - PADR}" y="${H - 6}" text-anchor="end" font-size="10" fill="currentColor" fill-opacity=".55">${xmax}</text>`;

  const path = (vals, w, op, dash) => {
    let d = '', pen = false;
    vals.forEach((v, i) => {
      if (v == null) { pen = false; return; }
      d += (pen ? 'L' : 'M') + sx(X[i]).toFixed(1) + ' ' + sy(v).toFixed(1) + ' ';
      pen = true;
    });
    return `<path d="${d}" fill="none" stroke="${op.color}" stroke-width="${w}" stroke-opacity="${op.o}" ${dash ? 'stroke-dasharray="4 3"' : ''} stroke-linejoin="round"/>`;
  };
  for (const s of smoothed) {
    if (opts.showRaw !== false) svg += path(s.raw, 1, { color: s.color, o: .18 }, false);
    svg += path(s.sm, 2, { color: s.color, o: 1 }, s.dashed);
  }
  svg += `</svg>`;

  const legend = smoothed.map(s =>
    `<span><span class="swatch" style="background:${s.color}"></span>${s.name}</span>`).join('');
  el.innerHTML = `<h2>${title}</h2><div class="note">${note}</div>${svg}<div class="legend">${legend}</div>`;
}

const CHARTS = [
  { title: 'Winrate vs OLDER snapshots  ← the real improvement signal',
    note: 'Present self beating its past selves. Above the 50% line = genuinely getting stronger. Raw winrate (mirror) hugs 50% by design.',
    series: [
      { name: 'vs old half', color: '#e0483e', f: d => d.wr_old },
      { name: 'overall', color: '#8a8a8a', f: d => d.winrate },
      { name: 'vs new half', color: '#3d8bff', f: d => d.wr_new },
    ], opts: { pct: true, y0: 0, y1: 1, ref: 0.5 } },
  { title: 'Net damage per round',
    note: '(dealt − taken) / rounds. Positive and rising = winning exchanges, not just trading.',
    series: [ { name: 'net dmg/round', color: '#2ea36b', f: d => d.rounds ? (d.dealt - d.taken) / d.rounds : null } ],
    opts: { ref: 0 } },
  { title: 'Combo length — dealt vs taken',
    note: 'Clean-hit chains landed (readout of the combo bonus — mean should drift up) vs unanswered chains TAKEN (readout of the Jul 18 combo-taken penalty — mean should drift DOWN as the bot learns to escape or answer back).',
    series: [
      { name: 'mean combo', color: '#d08b1f', f: d => d.mean_combo },
      { name: 'max combo', color: '#c0c0c0', f: d => d.max_combo, dashed: true },
      { name: 'mean taken', color: '#e0483e', f: d => d.mean_combo_taken },
      { name: 'max taken', color: '#8a5a5a', f: d => d.max_combo_taken, dashed: true },
    ] },
  { title: 'Exchange quality',
    note: 'first blood = share of rounds we land the opening hit. trade = share of our hits landed within 0.5s of taking one (trading, not comboing). block-eff = share of hits taken with block registered (damage halved). idle = uptake of the hitselect-release move (Part 1).',
    series: [
      { name: 'first blood %', color: '#2ea36b', f: d => d.first_blood },
      { name: 'trade %', color: '#e0483e', f: d => d.trade_frac },
      { name: 'block-eff %', color: '#9b59b6', f: d => d.blk_eff },
      { name: 'idle-move %', color: '#3d8bff', f: d => d.idle_frac },
    ], opts: { pct: true, y0: 0, y1: 1, ref: 0.5 } },
  { title: 'Style dials',
    note: 'Fraction of ticks holding block / swinging / jumping, in-range clicks that miss, and the share of the single most-used movement key-combo (11% = uniform over 9 moves, near 100% = movement collapse).',
    series: [
      { name: 'block %', color: '#9b59b6', f: d => d.blk },
      { name: 'click %', color: '#16a085', f: d => d.clk },
      { name: 'jump %', color: '#3d8bff', f: d => d.jmp },
      { name: 'whiff %', color: '#e0483e', f: d => d.whiff },
      { name: 'top-move share', color: '#8a8a8a', f: d => d.move_top_frac, dashed: true },
    ], opts: { pct: true, y0: 0 } },
  { title: 'Winrate vs scripted styles',
    note: 'Per-personality winrate. A style pinned at 100% is FARMED — it produces no learning signal anymore. The boxer (added Jul 17) is the current pressure; it should start well under 50% and climb slowly.',
    series: [
      { name: 'turtle', color: '#2ea36b', f: d => d.wr_turtle },
      { name: 'rusher', color: '#e0483e', f: d => d.wr_rusher },
      { name: 'kiter', color: '#3d8bff', f: d => d.wr_kiter },
      { name: 'boxer', color: '#d08b1f', f: d => d.wr_boxer },
      { name: 'trader', color: '#9b59b6', f: d => d.wr_trader },
    ], opts: { pct: true, y0: 0, y1: 1, ref: 0.5 } },
  { title: 'Exploration — policy entropy (nats)',
    note: '"Is it still trying new things?" Uniform ceilings: move ln(9)≈2.20, click/block/jump ln(2)≈0.69. Drifting down = converging (fine if winrate rises); a head near 0 has stopped exploring entirely.',
    series: [
      { name: 'move', color: '#e0483e', f: d => d.ent_move },
      { name: 'click', color: '#16a085', f: d => d.ent_click },
      { name: 'block', color: '#9b59b6', f: d => d.ent_block },
      { name: 'jump', color: '#3d8bff', f: d => d.ent_jump },
    ], opts: { y0: 0 } },
  { title: 'Aim: geometric bias vs learned lead (deg)',
    note: 'aim_bias = where the target ACTUALLY is relative to the crosshair at combat range (negative = crosshair sits right of the target). res_yaw = the mean yaw offset the policy adds. Healthy learning = res_yaw drifting to mirror the bias (canceling it); both flat = the residual is parked and misses stay priced in.',
    series: [
      { name: 'aim bias', color: '#e0483e', f: d => d.aim_bias },
      { name: 'learned lead (res_yaw)', color: '#3d8bff', f: d => d.res_yaw },
    ], opts: { ref: 0 } },
  { title: 'Average reward',
    note: 'Per-episode return. Noisy at 20 Hz; watch the smoothed trend, not single points.',
    series: [ { name: 'avg reward', color: '#e67e22', f: d => d.avg_reward } ] },
  { title: 'Control-loop staleness',
    note: 'Ticks between the state Python acted on and the action landing. Healthy = 1 (dashed line). Sitting near 2 = phase-locked session — the aim-spin bug.',
    series: [ { name: 'staleness', color: '#e0483e', f: d => d.staleness } ],
    opts: { ref: 1, y0: 0.8, y1: 2.2 } },
  { title: 'Policy loss',
    note: 'PPO policy-gradient loss — it lives near 0 and wobbles a bit each update as the policy shifts. Not a "lower is better" number; you just want it small and not exploding.',
    series: [ { name: 'policy', color: '#3d8bff', f: d => d.ploss } ],
    opts: { ref: 0 } },
  { title: 'Value loss (log scale)',
    note: 'How wrong the critic is at predicting round outcomes. Healthy is ~5-15. A spike past ~300 is a training detonation, NOT something that recovers on its own — the trainer now aborts before saving when it sees one (the Jul 17 blow-up).',
    series: [ { name: 'value', color: '#e0483e', f: d => d.vloss } ],
    opts: { log: true } },
];

function render() {
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  for (const c of CHARTS) {
    const card = document.createElement('div');
    card.className = 'card';
    grid.appendChild(card);
    chart(card, c.title, c.note, c.series, c.opts);
  }
  const last = ROWS[ROWS.length - 1] || {};
  document.getElementById('sub').textContent =
    `${ROWS.length} updates logged · latest update ${last.update ?? '?'}`;
}
const winEl = document.getElementById('win');
winEl.addEventListener('input', () => { document.getElementById('winv').textContent = winEl.value; render(); });
render();
</script>
"""


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "selfplay_metrics.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "selfplay_progress.html"
    rows = load_rows(src)
    if not rows:
        print(f"No data in {src} yet - run the trainer for a few updates first.")
        return
    html = HTML.replace("__DATA__", json.dumps(rows))
    with open(out, "w") as f:
        f.write(html)
    print(f"Wrote {out} from {len(rows)} updates. Open it in a browser (double-click).")


if __name__ == "__main__":
    main()
