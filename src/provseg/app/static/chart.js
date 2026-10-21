// Small dependency-free line chart: one line per series over a few x positions,
// 95% CI bands, direct end labels, a legend, and a crosshair tooltip that lists every series.
// Series colours are assigned in fixed palette order by series *name*, so a series keeps
// its colour across charts and filters.

const NS = "http://www.w3.org/2000/svg";
const SLOTS = 6;

export function colourIndex(name, order) {
  const i = order.indexOf(name);
  return (i < 0 ? order.length : i) % SLOTS;
}

function el(tag, attrs = {}, parent) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}

/**
 * @param {HTMLElement} host
 * @param {{xs:number[], series:{name:string, points:{x:number, mean:number|null, low:number|null, high:number|null}[]}[],
 *          order:string[], xLabel:string, yLabel:string, pct?:boolean, tooltip:HTMLElement}} spec
 */
export function lineChart(host, spec) {
  const { xs, series, order, xLabel, pct = true, tooltip } = spec;
  host.replaceChildren();
  host.classList.add("chart");
  const W = 560, H = 260, m = { t: 12, r: 96, b: 38, l: 44 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": spec.yLabel }, host);
  const vals = series.flatMap((s) => s.points.flatMap((p) => [p.low, p.high, p.mean])).filter((v) => v != null);
  let lo = Math.min(0, ...vals), hi = Math.max(...vals, pct ? 0.1 : 1);
  if (pct) { lo = 0; hi = Math.min(1, Math.ceil(hi * 4) / 4 || 1); }
  const x = (v) => m.l + (xs.indexOf(v) / Math.max(1, xs.length - 1)) * (W - m.l - m.r);
  const y = (v) => H - m.b - ((v - lo) / (hi - lo || 1)) * (H - m.t - m.b);
  const fmt = (v) => (v == null ? "n/a" : pct ? `${(100 * v).toFixed(1)}%` : v.toFixed(2));

  const grid = el("g", { class: "grid" }, svg);
  const axis = el("g", { class: "axis" }, svg);
  for (let i = 0; i <= 4; i++) {
    const v = lo + ((hi - lo) * i) / 4;
    el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }, grid);
    const t = el("text", { x: m.l - 8, y: y(v) + 4, "text-anchor": "end" }, axis);
    t.textContent = pct ? `${Math.round(100 * v)}` : v.toFixed(1);
  }
  xs.forEach((v) => {
    const t = el("text", { x: x(v), y: H - m.b + 16, "text-anchor": "middle" }, axis);
    t.textContent = String(v);
  });
  const xl = el("text", { x: (m.l + W - m.r) / 2, y: H - 4, "text-anchor": "middle", class: "lbl" }, svg);
  xl.textContent = xLabel;

  const ends = [];
  for (const s of series) {
    const c = `var(--series-${colourIndex(s.name, order) + 1})`;
    const pts = s.points.filter((p) => p.mean != null);
    if (!pts.length) continue;
    const banded = pts.filter((p) => p.low != null && p.high != null);
    if (banded.length > 1) {
      const d = banded.map((p, i) => `${i ? "L" : "M"}${x(p.x)},${y(p.high)}`).join("") +
        banded.slice().reverse().map((p) => `L${x(p.x)},${y(p.low)}`).join("") + "Z";
      el("path", { d, class: "band", fill: c }, svg);
    }
    el("path", { d: pts.map((p, i) => `${i ? "L" : "M"}${x(p.x)},${y(p.mean)}`).join(""), class: "series-line", stroke: c }, svg);
    pts.forEach((p) => el("circle", { cx: x(p.x), cy: y(p.mean), r: 4, class: "dot", fill: c }, svg));
    const last = pts[pts.length - 1];
    ends.push({ name: s.name, y: y(last.mean), x: x(last.x) });
  }
  // Direct labels at line ends, nudged apart so they never collide.
  ends.sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 13) ends[i].y = ends[i - 1].y + 13;
  if (series.length <= 4) {
    ends.forEach((e) => {
      const t = el("text", { x: e.x + 10, y: e.y + 4, class: "end-label" }, svg);
      t.textContent = e.name;
    });
  }

  const legend = document.createElement("div");
  legend.className = "legend";
  for (const s of series) {
    const span = document.createElement("span");
    const key = document.createElement("i");
    key.className = "key";
    key.style.background = `var(--series-${colourIndex(s.name, order) + 1})`;
    span.append(key, document.createTextNode(s.name));
    legend.append(span);
  }
  host.append(legend);

  // Crosshair: snaps to the nearest x position, tooltip lists every series there.
  const cross = el("line", { class: "crosshair", y1: m.t, y2: H - m.b, visibility: "hidden" }, svg);
  const hit = el("rect", { x: m.l - 12, y: 0, width: W - m.l - m.r + 24, height: H, fill: "transparent" }, svg);
  const show = (evt) => {
    const box = svg.getBoundingClientRect();
    const px = ((evt.clientX - box.left) / box.width) * W;
    const xv = xs.reduce((best, v) => (Math.abs(x(v) - px) < Math.abs(x(best) - px) ? v : best), xs[0]);
    cross.setAttribute("x1", x(xv));
    cross.setAttribute("x2", x(xv));
    cross.setAttribute("visibility", "visible");
    tooltip.replaceChildren();
    const head = document.createElement("div");
    head.className = "tt-head";
    head.textContent = `${xLabel}: ${xv}`;
    tooltip.append(head);
    for (const s of series) {
      const p = s.points.find((q) => q.x === xv);
      const row = document.createElement("div");
      row.className = "tt-row";
      const key = document.createElement("i");
      key.style.background = `var(--series-${colourIndex(s.name, order) + 1})`;
      const b = document.createElement("b");
      b.textContent = fmt(p?.mean ?? null);
      const name = document.createElement("span");
      name.textContent = s.name;
      row.append(key, b, name);
      tooltip.append(row);
    }
    tooltip.hidden = false;
    const tx = Math.min(window.innerWidth - 220, evt.clientX + 14);
    tooltip.style.left = `${tx}px`;
    tooltip.style.top = `${evt.clientY + 14}px`;
  };
  hit.addEventListener("pointermove", show);
  hit.addEventListener("pointerleave", () => { tooltip.hidden = true; cross.setAttribute("visibility", "hidden"); });
}
