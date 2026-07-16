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
const X = DATA.map(d => d.update);

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
  // autoscale over all (smoothed) series
  let lo = Infinity, hi = -Infinity;
  const smoothed = series.map(s => {
    const raw = DATA.map(s.f);
    const sm = rollmean(raw, win);
    for (const v of sm) if (v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    return { ...s, raw, sm };
  });
  if (opts.y0 != null) lo = Math.min(lo, opts.y0);
  if (opts.y1 != null) hi = Math.max(hi, opts.y1);
  if (!isFinite(lo)) { lo = 0; hi = 1; }
  if (lo === hi) { hi = lo + 1; }
  const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
  const xmin = X[0], xmax = X[X.length - 1] || 1;
  const sx = u => PADL + (W - PADL - PADR) * ((u - xmin) / Math.max(1, xmax - xmin));
  const sy = v => PADT + (H - PADT - PADB) * (1 - (v - lo) / (hi - lo));
  const fmt = opts.pct ? (v => (v * 100).toFixed(0) + '%') : (v => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2)));

  let svg = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">`;
  // gridlines + y labels
  for (let i = 0; i <= 4; i++) {
    const v = lo + (hi - lo) * i / 4, y = sy(v);
    svg += `<line x1="${PADL}" y1="${y}" x2="${W - PADR}" y2="${y}" stroke="currentColor" stroke-opacity=".08"/>`;
    svg += `<text x="${PADL - 6}" y="${y + 3}" text-anchor="end" font-size="10" fill="currentColor" fill-opacity=".55">${fmt(v)}</text>`;
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
  { title: 'Combo length',
    note: 'Clean-hit chain length per rollout. This is the readout of the clean-combo shaping — mean should drift up as chains hold.',
    series: [
      { name: 'mean combo', color: '#d08b1f', f: d => d.mean_combo },
      { name: 'max combo', color: '#c0c0c0', f: d => d.max_combo, dashed: true },
    ] },
  { title: 'Style dials',
    note: 'Fraction of ticks holding block / swinging — what the shaping is trying to move.',
    series: [
      { name: 'block %', color: '#9b59b6', f: d => d.blk },
      { name: 'click %', color: '#16a085', f: d => d.clk },
    ], opts: { pct: true, y0: 0 } },
  { title: 'Average reward',
    note: 'Per-episode return. Noisy at 20 Hz; watch the smoothed trend, not single points.',
    series: [ { name: 'avg reward', color: '#e67e22', f: d => d.avg_reward } ] },
  { title: 'Control-loop staleness',
    note: 'Ticks between the state Python acted on and the action landing. Healthy = 1 (dashed line). Sitting near 2 = phase-locked session — the aim-spin bug.',
    series: [ { name: 'staleness', color: '#e0483e', f: d => d.staleness } ],
    opts: { ref: 1, y0: 0.8, y1: 2.2 } },
  { title: 'Losses',
    note: 'Policy and value loss. Spikes = the policy moved; sustained blow-up = instability.',
    series: [
      { name: 'policy', color: '#3d8bff', f: d => d.ploss },
      { name: 'value', color: '#e0483e', f: d => d.vloss, dashed: true },
    ] },
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
  const last = DATA[DATA.length - 1] || {};
  document.getElementById('sub').textContent =
    `${DATA.length} updates logged · latest update ${last.update ?? '?'}`;
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
