import { lineChart } from "./chart.js";

// ---------- tiny DOM helper: strings are always inserted as text, never as HTML ----------
function h(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") n.className = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) n.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  return n;
}
const icon = (name) => h("i", { class: `ph ph-${name}`, "aria-hidden": "true" });
const pct = (v) => (v == null ? "n/a" : `${(100 * v).toFixed(1)}`);
const app = document.getElementById("app");
const tooltip = document.getElementById("tooltip");

// ---------- theme ----------
const root = document.documentElement;
try { const t = localStorage.getItem("pe-theme"); if (t) root.dataset.theme = t; } catch {}
document.getElementById("theme-toggle").addEventListener("click", () => {
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("pe-theme", root.dataset.theme); } catch {}
});

// ---------- data ----------
const cache = new Map();
async function api(path) {
  if (cache.has(path)) return cache.get(path);
  const p = fetch(`/api${path}`).then(async (r) => {
    if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 200)}`);
    return r.json();
  });
  cache.set(path, p);
  p.catch(() => cache.delete(path));
  return p;
}

const state = { runs: [], run: null };

// ---------- routing: #/  #/questions  #/questions/<qid>?noise=10 ----------
function parseHash() {
  const [path, query] = location.hash.replace(/^#/, "").split("?");
  const parts = path.split("/").filter(Boolean).map(decodeURIComponent);
  const params = new URLSearchParams(query || "");
  return { view: parts[0] || "overview", qid: parts[1] || null, params };
}

async function route() {
  closeDrawer();
  const { view, qid, params } = parseHash();
  document.querySelectorAll(".tabs a").forEach((a) => (a.dataset.view === view ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
  if (!state.run) {
    app.replaceChildren(emptyNoRuns());
    return;
  }
  try {
    if (view === "questions") await renderQuestions(qid, Number(params.get("noise") || 0));
    else await renderOverview();
  } catch (e) {
    app.replaceChildren(h("div", { class: "empty error" }, icon("warning-circle"), h("b", {}, "Could not load this view"), h("span", {}, String(e.message || e))));
  }
}

function emptyNoRuns() {
  return h("div", { class: "empty" }, icon("folder-dashed"), h("b", {}, "No experiment runs found"),
    h("span", {}, "Build a run first. The explorer reads everything from the runs directory."),
    h("pre", {}, "uv run provseg prepare -c configs/experiments/main.yaml\nuv run provseg build -c configs/experiments/main.yaml\nuv run provseg serve"));
}

// ---------- overview ----------
function seriesFrom(table, keyField, metric, xs) {
  const names = [...new Set(table.map((r) => r[keyField]))];
  return names.map((name) => ({
    name,
    points: xs.map((x) => {
      const r = table.find((t) => t[keyField] === name && t.noise_level === x);
      const c = r?.[metric];
      return { x, mean: c?.mean ?? null, low: c?.low ?? null, high: c?.high ?? null };
    }),
  }));
}

function chartPanel(title, unit, draw) {
  const host = h("div");
  const panel = h("div", { class: "panel" }, h("div", { class: "chart-title" }, h("h3", {}, title), h("span", { class: "unit" }, unit)), host);
  requestAnimationFrame(() => draw(host));
  return panel;
}

function ciCell(c) {
  if (!c || c.mean == null) return h("td", { class: "num" }, "n/a");
  return h("td", { class: "num" }, pct(c.mean), h("span", { class: "ci" }, `${pct(c.low)}-${pct(c.high)}`));
}

async function renderOverview() {
  const run = state.run;
  app.replaceChildren(h("div", { class: "skeleton", style: "height:120px" }));
  const s = await api(`/runs/${encodeURIComponent(run.name)}/summary`);
  const cfg = s.manifest.config;
  const st = s.manifest.sample_stats || {};
  const xs = cfg.noise.levels;
  const head = h("div", { class: "page-head" },
    h("h1", {}, run.name),
    h("p", {}, cfg.description || "Experiment run."),
    h("div", { class: "meta-row" },
      h("span", {}, "Base model ", h("b", {}, run.base_model || cfg.base_model)),
      h("span", {}, "Context budget ", h("b", {}, `${cfg.context_budget} tokens`)),
      h("span", {}, "Noise ", h("b", {}, `${cfg.noise.mode}, ${xs.join(" / ")} chunks`)),
      h("span", {}, "Prompt version ", h("b", {}, s.manifest.prompt_version)),
      h("span", {}, "Code ", h("b", { class: "mono" }, s.manifest.git_commit ? s.manifest.git_commit.slice(0, 10) : "uncommitted"))));
  const stats = h("div", { class: "stats" },
    [["Questions", st.questions], ["Answerable", st.answerable], ["Unanswerable", st.unanswerable], ["Papers", st.papers], ["Instances", st.instances],
     ["Evidence mapped", st.evidence_mapping ? `${st.evidence_mapping.matched}/${st.evidence_mapping.total}` : null]]
      .map(([k, v]) => h("div", { class: "stat" }, h("div", { class: "v" }, v ?? "n/a"), h("div", { class: "k" }, k))));

  const sections = [head, stats];

  const feats = await api(`/runs/${encodeURIComponent(run.name)}/featured`).catch(() => []);
  if (feats.length) {
    sections.push(h("section", { class: "section" },
      h("h2", {}, "Start here"),
      h("p", { class: "lede" }, "Questions picked by a fixed rule from the retrieval diagnostic below, not by hand. Open one to see where its evidence sits among the distractors, and test any claim against the source passages."),
      h("div", { class: "featured" }, feats.map((f) => h("a", { class: "feat", href: `#/questions/${encodeURIComponent(f.qid)}?noise=${f.unanswerable ? 0 : 50}` },
        h("div", { class: "q" }, f.question),
        h("div", { class: "t" }, f.title),
        h("div", { class: "why" }, icon(f.unanswerable ? "prohibit" : "chart-line-down"), f.reason))))));
  }

  // Retrieval diagnostic (no LLM involved)
  if (s.retrieval) {
    const t = s.retrieval.table;
    const order = [...new Set(t.map((r) => r.method))];
    sections.push(h("section", { class: "section" },
      h("h2", {}, "Evidence that reaches a fixed context budget"),
      h("p", { class: "lede" }, `Retrieval only, no model calls. For each answerable question: the share of gold evidence chunks that fit into the ${s.retrieval.context_budget}-token budget, and how much of that budget distractors take, as distractor chunks are injected.`),
      h("div", { class: "grid-2" },
        chartPanel("Gold evidence recall within budget", "% of gold chunks, 95% CI", (host) =>
          lineChart(host, { xs, series: seriesFrom(t, "method", "gold_recall_at_budget", xs), order, xLabel: "Distractor chunks injected", yLabel: "Gold recall", tooltip })),
        chartPanel("Budget taken by distractors", "% of context tokens", (host) =>
          lineChart(host, { xs, series: seriesFrom(t, "method", "distractor_token_share", xs), order, xLabel: "Distractor chunks injected", yLabel: "Distractor share", tooltip }))),
      h("div", { class: "table-wrap", style: "margin-top:16px" }, h("table", {},
        h("thead", {}, h("tr", {}, h("th", {}, "Method"), h("th", { class: "num" }, "Noise"), h("th", { class: "num" }, "n"), h("th", { class: "num" }, "Gold recall @ budget"), h("th", { class: "num" }, "All gold in context"), h("th", { class: "num" }, "Distractor share"), h("th", { class: "num" }, "MRR first gold"))),
        h("tbody", {}, t.map((r) => h("tr", {}, h("td", {}, r.method), h("td", { class: "num" }, r.noise_level), h("td", { class: "num" }, r.n), ciCell(r.gold_recall_at_budget), ciCell(r.all_gold_in_context), ciCell(r.distractor_token_share), ciCell(r.mrr_first_gold))))))));
  }

  // Main results
  if (s.metrics) {
    const t = s.metrics.table;
    const order = cfg.systems.map((x) => x.name);
    const metrics = [["answer_f1", "Answer F1", "answerable questions"], ["citation_precision", "Citation precision", "judge-verified citations"], ["unsupported_claim_rate", "Unsupported claims", "lower is better"]];
    sections.push(h("section", { class: "section" },
      h("h2", {}, "Answer quality and citation faithfulness under noise"),
      h("p", { class: "lede" }, "Same questions, base model, prompt and budget for every system. Means with 95% bootstrap CIs; citation support is decided by the separate evaluation judge, never by the in-system verifier."),
      h("div", { class: "grid-3" }, metrics.map(([k, title, unit]) => chartPanel(title, unit, (host) => lineChart(host, { xs, series: seriesFrom(t, "system", k, xs), order, xLabel: "Distractor chunks", yLabel: title, tooltip })))),
      h("div", { class: "table-wrap", style: "margin-top:16px" }, h("table", {},
        h("thead", {}, h("tr", {}, ["System", "Noise", "n", "Answer F1", "Citation prec.", "Unsupported", "Evidence F1", "Distractor cites", "Abstain F1", "Calls", "Cost $"].map((c, i) => h("th", { class: i ? "num" : "" }, c)))),
        h("tbody", {}, t.map((r) => h("tr", {}, h("td", {}, r.system), h("td", { class: "num" }, r.noise_level), h("td", { class: "num" }, r.n),
          ciCell(r.answer_f1), ciCell(r.citation_precision), ciCell(r.unsupported_claim_rate), ciCell(r.evidence_f1), ciCell(r.distractor_citation_rate),
          h("td", { class: "num" }, pct(r.abstain_f1)), h("td", { class: "num" }, r.calls?.mean?.toFixed(1) ?? "n/a"), h("td", { class: "num" }, r.cost_usd?.mean?.toFixed(4) ?? "n/a"))))))));
  } else {
    const counts = Object.entries(s.prediction_counts || {});
    sections.push(h("section", { class: "section" }, h("h2", {}, "Model results"),
      h("div", { class: "empty" }, icon("hourglass-medium"),
        h("b", {}, counts.length ? `Predictions exist for ${counts.length} system/noise pairs but have not been evaluated yet.` : "No model predictions in this run yet."),
        h("span", {}, "These steps call the configured model provider and need an API key. Estimate the cost first."),
        h("pre", {}, `uv run provseg estimate -c configs/experiments/${run.name.replace(/_v\d+$/, "")}.yaml\nuv run provseg predict  -c <config>\nuv run provseg judge    -c <config>\nuv run provseg evaluate -c <config>`))));
  }
  app.replaceChildren(...sections);
}

// ---------- questions ----------
let qFilter = { text: "", kind: "all" };

async function renderQuestions(qid, noise) {
  const run = state.run;
  const qs = await api(`/runs/${encodeURIComponent(run.name)}/questions`);
  if (!qid && qs.length) qid = qs[0].qid;
  const list = h("div", { class: "qitems", role: "list" });
  const count = h("div", { class: "qcount" });
  const search = h("input", { type: "search", placeholder: "Filter questions or papers", "aria-label": "Filter questions or papers", value: qFilter.text });
  const seg = h("div", { class: "seg", role: "group", "aria-label": "Question type" },
    ["all", "answerable", "unanswerable"].map((k) => h("button", { type: "button", "aria-pressed": String(qFilter.kind === k), onclick: () => { qFilter.kind = k; seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b.textContent.toLowerCase() === k))); fill(); } }, k[0].toUpperCase() + k.slice(1))));
  const fill = () => {
    const t = qFilter.text.toLowerCase();
    const shown = qs.filter((q) => (qFilter.kind === "all" || (qFilter.kind === "unanswerable") === q.unanswerable) && (!t || q.question.toLowerCase().includes(t) || q.title.toLowerCase().includes(t)));
    list.replaceChildren(...shown.map((q) => h("button", { class: "qitem", role: "listitem", "aria-current": String(q.qid === qid), onclick: () => (location.hash = `#/questions/${encodeURIComponent(q.qid)}?noise=${noise}`) },
      h("div", { class: "q" }, q.question), h("div", { class: "t" }, q.unanswerable ? "Unanswerable · " : "", q.title))));
    count.textContent = `${shown.length} of ${qs.length} questions`;
  };
  search.addEventListener("input", () => { qFilter.text = search.value; fill(); });
  fill();
  const detail = h("div", {}, h("div", { class: "skeleton", style: "height:200px" }));
  app.replaceChildren(h("div", { class: "browser" }, h("div", { class: "qlist" }, h("div", { class: "qlist-tools" }, search, seg), list, count), detail));
  if (!qid) { detail.replaceChildren(h("div", { class: "empty" }, "This run has no questions.")); return; }
  const d = await api(`/runs/${encodeURIComponent(run.name)}/questions/${encodeURIComponent(qid)}?noise=${noise}`);
  detail.replaceChildren(...questionDetail(d));
}

function verdictBadge(v) {
  const map = { entailed: ["good", "check-circle", "Verified"], not_enough_evidence: ["warn", "question", "Not enough evidence"], contradicted: ["bad", "x-circle", "Contradicted"] };
  const m = map[v];
  return m ? h("span", { class: `badge ${m[0]}`, title: "In-system verifier" }, icon(m[1]), m[2]) : null;
}
function judgeBadge(label) {
  const map = { supported: ["good", "seal-check", "Judge: supported"], partially_supported: ["warn", "circle-half", "Judge: partial"], not_supported: ["bad", "seal-warning", "Judge: not supported"], contradicted: ["bad", "x-circle", "Judge: contradicted"], error: ["neutral", "warning", "Judge error"] };
  const m = map[label];
  return m ? h("span", { class: `badge ${m[0]}`, title: "Evaluation judge" }, icon(m[1]), m[2]) : null;
}

function questionDetail(d) {
  const handleOf = Object.fromEntries(d.chunks.map((c) => [c.chunk_id, c.handle]));
  const chunkOf = Object.fromEntries(d.chunks.map((c) => [c.chunk_id, c]));
  const head = h("div", { class: "qhead" },
    h("h1", {}, d.question),
    h("div", { class: "paper" }, d.title, " ", h("span", { class: "mono" }, `(${d.doc_id})`)),
    h("div", { class: "gold" }, h("span", {}, "Gold answers"), [...new Set(d.references.map((r) => r.answer))].map((a) => h("span", { class: "chip gold-ans" }, a))));
  const noiseSeg = h("div", { class: "seg", role: "group", "aria-label": "Distractor chunks injected" },
    d.noise_levels.map((k) => h("button", { type: "button", "aria-pressed": String(k === d.noise_level), onclick: () => (location.hash = `#/questions/${encodeURIComponent(d.qid)}?noise=${k}`) }, String(k))));
  const toolbar = h("div", { class: "toolbar" }, h("span", {}, h("span", { class: "label" }, "Distractor chunks"), noiseSeg),
    h("span", { class: "label" }, `${d.chunks.length} chunks in this document, ${d.chunks.filter((c) => c.is_distractor).length} injected`));

  // Document strip: what each system put into its final context.
  const rows = [h("div", { class: "strip-row" }, h("div", { class: "name" }, "Document"), cells(d.chunks, new Set(), d))];
  for (const p of d.predictions) {
    const inCtx = new Set(p.context.flatMap((c) => c.chunk_ids));
    rows.push(h("div", { class: "strip-row" }, h("div", { class: "name", title: p.system }, p.system), cells(d.chunks, inCtx, d, p)));
  }
  const cc = d.computed_contexts;
  const ran = new Set(d.predictions.map((p) => p.system));
  const computed = cc ? [["rag_bm25", "BM25 top-k"], ["direct", "First chunks"]].filter(([k]) => !ran.has(k)) : [];
  for (const [k, label] of computed) {
    rows.push(h("div", { class: "strip-row" }, h("div", { class: "name", title: `${k}: built with the system's own ranking and token budget, no model call` }, label), cells(d.chunks, new Set(cc[k].chunk_ids), d)));
  }
  const share = (c) => (c.tokens ? Math.round((100 * c.distractor_tokens) / c.tokens) : 0);
  const fitLine = computed.length && cc.rag_bm25.gold_total
    ? h("p", { class: "fit" }, `Within the ${cc.budget}-token budget, BM25 fits ${cc.rag_bm25.gold_in_context} of ${cc.rag_bm25.gold_total} gold chunks (distractors take ${share(cc.rag_bm25)}% of it); taking the first chunks fits ${cc.direct.gold_in_context}. The computed rows use each system's own ranking and budget, with no model call.`)
    : null;
  const strip = h("div", { class: "panel" }, h("div", { class: "docstrip" }, rows),
    h("div", { class: "strip-legend" },
      h("span", {}, h("span", { class: "cell gold" }), "gold evidence"),
      h("span", {}, h("span", { class: "cell distractor" }), "injected distractor"),
      h("span", {}, h("span", { class: "cell in" }), "in the system's final context")), fitLine);

  const cards = d.predictions.length
    ? d.predictions.map((p) => answerCard(p, d, handleOf, chunkOf))
    : [h("div", { class: "empty" }, icon("hourglass-medium"), h("b", {}, "No model answers for this question at this noise level yet."), h("span", {}, "System answers with their citations and verifier decisions appear here once the experiment has been run with a model provider. Everything above is computed without a model."))];
  return [head, toolbar, strip, claimChecker(d), h("div", { class: "answers" }, cards)];
}

function cells(chunks, inCtx, d, pred) {
  return h("div", { class: "cells" }, chunks.map((c) => h("button", {
    type: "button",
    class: `cell${c.is_distractor ? " distractor" : ""}${c.is_gold ? " gold" : ""}${inCtx.has(c.chunk_id) ? " in" : ""}`,
    title: `${c.handle} ${c.heading}${c.is_distractor ? " (distractor)" : ""}${c.is_gold ? " (gold)" : ""}`,
    "aria-label": `Open ${c.handle}`,
    onclick: () => openSource(c.chunk_id, { d, pred }),
  })));
}

function answerCard(p, d, handleOf, chunkOf) {
  const judged = p.judgments != null;
  const judgeFor = (idx, scope) => (p.judgments || []).find((j) => j.statement_idx === idx && j.scope === scope)?.label;
  const stmtBlock = (st, idx, dropped) => {
    const scope = st.citations.length === 1 ? st.citations[0] : "joint";
    const jl = judged ? judgeFor(idx, scope) : null;
    return h("div", { class: `stmt${dropped ? " dropped" : ""}` },
      h("p", {}, st.text),
      st.original_text ? h("div", { class: "revised-from" }, "Revised from: ", st.original_text) : null,
      h("div", { class: "cites" },
        st.citations.map((cid) => h("button", { type: "button", class: `cite${chunkOf[cid]?.is_distractor ? " distractor" : ""}`, title: cid, onclick: () => openSource(cid, { d, pred: p, statement: st, judge: judgeFor(idx, cid) }) }, handleOf[cid] || cid)),
        st.invalid_citations.map((raw) => h("span", { class: "cite invalid", title: "Cited a passage that was not in the final context" }, raw)),
        !st.citations.length && !st.invalid_citations.length ? h("span", { class: "badge bad" }, icon("link-break"), "No citation") : null,
        p.kind === "provseg" && st.verdict !== "unverified" ? verdictBadge(st.verdict) : null,
        jl ? judgeBadge(jl) : null,
        dropped ? h("span", { class: "badge neutral" }, icon("trash"), "Removed by verifier") : null));
  };
  const u = p.usage || {};
  const notes = p.trace?.notes || [];
  return h("article", { class: "answer-card" },
    h("header", {}, h("span", { class: "sys" }, p.system), h("span", { class: "kind" }, p.kind)),
    p.error ? h("div", { class: "callout warn" }, icon("warning"), `Run error: ${p.error}`) : null,
    h("div", { class: `answer-text${p.abstained ? " abstain" : ""}` }, p.abstained ? "Abstained: the context does not establish an answer." : p.answer || "(empty)"),
    p.statements.map((st, i) => stmtBlock(st, i, false)),
    p.dropped_statements.length ? h("details", { class: "more" }, h("summary", {}, `${p.dropped_statements.length} statement(s) removed by verification`), p.dropped_statements.map((st, i) => stmtBlock(st, `d${i}`, true))) : null,
    notes.length ? h("details", { class: "more" }, h("summary", {}, `Notes from ${notes.length} chunks (quotes not found in their source are underlined)`), notes.filter((n) => n.relevance !== "none").map((n) =>
      h("div", { class: "note" }, h("div", {}, h("b", { class: "mono" }, n.handles.join(", ")), ` relevance: ${n.relevance}`),
        n.evidence.map((q) => h("div", {}, h("span", { class: q.verified ? "q-ok" : "q-bad" }, `"${q.text}"`))),
        n.reasoning ? h("div", { style: "color:var(--text-2)" }, n.reasoning) : null))) : null,
    h("div", { class: "usage" }, h("span", {}, `${p.context_tokens} ctx tokens`), h("span", {}, `${u.calls ?? 0} calls`), h("span", {}, `${(u.input_tokens ?? 0) + (u.output_tokens ?? 0)} tokens`), h("span", {}, `$${(u.cost_usd ?? 0).toFixed(4)}`)),
    !judged && p.statements.length ? h("div", { class: "callout info" }, icon("info"), "Not judged yet: run `provseg judge` to score citation support.") : null);
}

// ---------- live claim check (local NLI verifier, no model provider) ----------
const MAX_SOURCES = 4;
const pct0 = (v) => `${Math.round(100 * v)}%`;

function claimChecker(d) {
  const byId = Object.fromEntries(d.chunks.map((c) => [c.chunk_id, c]));
  const answers = [...new Set(d.references.map((r) => r.answer).filter((a) => a && a !== "Unanswerable" && a.length <= 600))];
  const evidence = [...new Set(d.references.flatMap((r) => r.highlighted_evidence).filter((e) => e && e.length >= 12 && e.length <= 600))];
  const presets = [...answers.slice(0, 3).map((t) => ["Annotator answer", t]), ...evidence.slice(0, 3).map((t) => ["Highlighted evidence", t])];
  const bm = d.computed_contexts?.rag_bm25;
  const topBm = bm ? Object.entries(bm.scores).sort((a, b) => b[1] - a[1]).map(([id]) => id) : [];
  let selected = d.gold_chunk_ids.length ? d.gold_chunk_ids.filter((id) => byId[id]).slice(0, MAX_SOURCES) : topBm.slice(0, 1);

  const claim = h("textarea", { id: "claim-input", rows: "3", maxlength: "600", "aria-describedby": "claim-help" });
  claim.value = evidence[0] ?? answers[0] ?? "";
  const presetRow = h("div", { class: "presets" }, presets.map(([kind, text]) =>
    h("button", { type: "button", class: "preset", title: text, onclick: () => { claim.value = text; claim.focus(); } }, h("span", { class: "k" }, kind), h("span", { class: "v" }, text))));
  const chosen = h("div", { class: "chosen", "aria-live": "polite" });
  const add = h("select", { "aria-label": "Add a source passage" });
  const result = h("div", { class: "check-result", "aria-live": "polite" });
  const verifyBtn = h("button", { type: "button", class: "btn primary" }, icon("seal-check"), "Verify");
  const quoteBtn = h("button", { type: "button", class: "btn" }, icon("text-aa"), "Find as quote");

  const tag = (c) => `${c.handle}  ${c.heading}${c.is_gold ? "  (gold)" : ""}${c.is_distractor ? "  (distractor)" : ""}`;
  const redraw = () => {
    chosen.replaceChildren(...selected.map((id) => h("span", { class: `pick${byId[id].is_distractor ? " distractor" : ""}` },
      h("button", { type: "button", class: "cite", title: id, onclick: () => openSource(id, { d }) }, byId[id].handle),
      h("span", { class: "h" }, byId[id].heading),
      byId[id].is_gold ? h("span", { class: "badge warn" }, "gold") : null,
      byId[id].is_distractor ? h("span", { class: "badge neutral" }, "distractor") : null,
      h("button", { type: "button", class: "x", "aria-label": `Remove ${byId[id].handle}`, onclick: () => { selected = selected.filter((x) => x !== id); redraw(); } }, icon("x")))));
    if (!selected.length) chosen.append(h("span", { class: "hint" }, "No passage chosen yet."));
    add.replaceChildren(h("option", { value: "" }, selected.length >= MAX_SOURCES ? `At most ${MAX_SOURCES} passages` : "Add a passage"),
      ...d.chunks.filter((c) => !selected.includes(c.chunk_id)).map((c) => h("option", { value: c.chunk_id }, tag(c))));
    add.disabled = selected.length >= MAX_SOURCES;
  };
  add.addEventListener("change", () => { if (add.value) { selected = [...selected, add.value].slice(0, MAX_SOURCES); redraw(); } });
  redraw();

  async function run(nli) {
    const text = claim.value.trim();
    if (!text) { result.replaceChildren(h("div", { class: "callout warn" }, icon("warning"), "Write a claim first.")); return; }
    if (!selected.length) { result.replaceChildren(h("div", { class: "callout warn" }, icon("warning"), "Choose at least one source passage.")); return; }
    verifyBtn.disabled = quoteBtn.disabled = true;
    result.replaceChildren(h("div", { class: "pending" }, h("div", { class: "skeleton", style: "height:14px;width:60%" }), h("div", { class: "skeleton", style: "height:14px;width:85%" }),
      h("span", {}, nli ? "Running the NLI verifier on CPU. This usually takes 10 to 30 seconds, longer on the first check while the model loads." : "Searching the passages.")));
    try {
      const r = await fetch(`/api/runs/${encodeURIComponent(state.run.name)}/check`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ qid: d.qid, noise: d.noise_level, claim: text, chunk_ids: selected, nli }) });
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `HTTP ${r.status}`);
      result.replaceChildren(...checkResult(await r.json(), d, byId));
    } catch (e) {
      result.replaceChildren(h("div", { class: "callout warn" }, icon("warning-circle"), `The check failed: ${e.message || e}`));
    } finally {
      verifyBtn.disabled = !state.liveNli;
      quoteBtn.disabled = false;
    }
  }
  verifyBtn.addEventListener("click", () => run(true));
  quoteBtn.addEventListener("click", () => run(false));
  verifyBtn.disabled = !state.liveNli;

  return h("section", { class: "panel checker", "aria-labelledby": "checker-title" },
    h("h2", { id: "checker-title" }, "Check a claim against the source"),
    h("p", { class: "lede" }, "This is Provenance-Segment's verification step, run live with no model provider: exact quote anchoring, then the NLI verifier on the original text of the chosen passages (best of 1- and 3-sentence windows; supported at entailment 0.5 or above)."),
    h("label", { for: "claim-input" }, "Claim"),
    claim,
    h("div", { id: "claim-help", class: "help" }, "Start from an annotator's answer or evidence, then edit it: change a number, negate it, or check it against a distractor instead."),
    presets.length ? presetRow : null,
    h("div", { class: "label-row" }, h("span", {}, "Source passages"), add),
    chosen,
    h("div", { class: "actions" }, verifyBtn, quoteBtn, state.liveNli ? null : h("span", { class: "hint" }, "The NLI verifier is off on this host; quote search still works.")),
    result);
}

function checkResult(r, d, byId) {
  const head = h("div", { class: "result-head" },
    r.nli ? h("span", {}, verdictBadge(r.nli.label), h("span", { class: "score" }, `entailment ${pct0(r.nli.entail)}`),
      r.nli.joint && r.nli.joint.entail >= r.nli.entail ? h("span", { class: "hint" }, "decided by the cited passages together") : null)
      : h("span", { class: "score" }, "Quote search only"),
    r.nli ? h("span", { class: "hint mono" }, r.nli.model) : null);
  const rows = r.sources.map((s) => {
    const c = byId[s.chunk_id];
    const ranges = [];
    if (s.quote.found) ranges.push([s.quote.start, s.quote.end]);
    else if (s.nli?.window_start != null) ranges.push([s.nli.window_start, s.nli.window_end]);
    const note = s.quote.found ? "The claim occurs verbatim at the highlighted offsets." : s.nli?.window_start != null ? "Highlighted: the window the verifier scored highest." : null;
    return h("div", { class: "src-row" },
      h("button", { type: "button", class: `cite${c?.is_distractor ? " distractor" : ""}`, title: "Open the passage", onclick: () => openSource(s.chunk_id, { d, ranges, rangeNote: note }) }, s.handle || s.chunk_id),
      h("div", { class: "src-body" },
        h("div", { class: "src-tags" },
          s.quote.found ? h("span", { class: "badge good" }, icon("quotes"), `verbatim at characters ${s.quote.start} to ${s.quote.end}${s.quote.elided ? ", with an elision" : ""}`) : h("span", { class: "badge neutral" }, icon("quotes"), "not verbatim"),
          s.nli ? h("span", { class: "score", title: `Highest contradiction over this passage's windows: ${pct0(s.nli.contradict)}` }, `entailment ${pct0(s.nli.entail)}`) : null,
          c?.is_distractor ? h("span", { class: "badge neutral" }, "distractor") : null),
        s.nli?.window ? h("blockquote", {}, s.nli.window) : null));
  });
  return [head, ...rows];
}

// ---------- source drawer ----------
const drawer = document.getElementById("drawer");
function closeDrawer() { drawer.classList.remove("open"); drawer.setAttribute("aria-hidden", "true"); }
document.getElementById("drawer-close").addEventListener("click", closeDrawer);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });

function renderSource(text, ranges, overlapStart, overlapEnd) {
  // Split the chunk at highlight and overlap boundaries, then style heading lines.
  const cuts = new Set([0, text.length, overlapStart, overlapEnd]);
  ranges.forEach(([a, b]) => { cuts.add(a); cuts.add(b); });
  const pts = [...cuts].filter((x) => x >= 0 && x <= text.length).sort((a, b) => a - b);
  const out = h("div", { class: "source" });
  for (let i = 0; i < pts.length - 1; i++) {
    const [a, b] = [pts[i], pts[i + 1]];
    if (a === b) continue;
    const seg = text.slice(a, b);
    const marked = ranges.some(([x, y]) => a >= x && b <= y);
    const overlap = a >= overlapStart && b <= overlapEnd;
    const parts = seg.split(/(§ [^\n]*)/);
    for (const part of parts) {
      if (!part) continue;
      let node = part.startsWith("§ ") ? h("span", { class: "sec" }, part.slice(2)) : document.createTextNode(part);
      if (marked) node = h("mark", {}, node);
      else if (overlap) node = h("span", { class: "overlap", title: "Overlap carried from the previous chunk" }, node);
      out.append(node);
    }
  }
  return out;
}

async function openSource(chunkId, ctx) {
  const c = await api(`/runs/${encodeURIComponent(state.run.name)}/chunks/${encodeURIComponent(chunkId)}`);
  const meta = ctx.d.chunks.find((x) => x.chunk_id === chunkId);
  document.getElementById("drawer-kicker").textContent = `${meta?.handle ?? ""}  ${chunkId}`;
  document.getElementById("drawer-title").textContent = c.section_heading;
  const ranges = [];
  for (const n of ctx.pred?.trace?.notes || []) for (const q of n.evidence) if (q.verified && q.handle === meta?.handle && q.start != null) ranges.push([q.start, q.end]);
  for (const n of ctx.pred?.trace?.final_notes || []) for (const q of n.evidence) if (q.verified && q.handle === meta?.handle && q.start != null) ranges.push([q.start, q.end]);
  for (const r of ctx.ranges || []) ranges.push(r);
  const headingEnd = c.text.indexOf("\n\n") + 2;
  const body = [
    c.is_distractor ? h("div", { class: "callout warn" }, icon("warning"), `Injected distractor from another paper: ${c.title}`) : h("div", { class: "callout info" }, icon("file-text"), c.title),
    meta?.is_gold ? h("div", { class: "callout info" }, icon("star"), "Contains gold evidence for this question.") : null,
  ];
  if (ctx.statement) {
    body.push(h("div", { class: "judgement" }, h("b", {}, "Cited for"), h("div", {}, ctx.statement.text),
      h("div", { class: "cites" }, verdictBadge(ctx.statement.verdict), ctx.judge ? judgeBadge(ctx.judge) : null)));
  }
  if (ctx.rangeNote) body.push(h("div", { style: "color:var(--text-3);font-size:12px" }, ctx.rangeNote));
  else if (ranges.length) body.push(h("div", { style: "color:var(--text-3);font-size:12px" }, `${ranges.length} quote(s) recorded by this system's notes are highlighted at their exact offsets.`));
  body.push(renderSource(c.text, ranges, c.overlap_chars ? headingEnd : -1, c.overlap_chars ? headingEnd + c.overlap_chars : -1));
  document.getElementById("drawer-body").replaceChildren(...body.filter(Boolean));
  drawer.classList.add("open");
  drawer.setAttribute("aria-hidden", "false");
  document.getElementById("drawer-close").focus();
}

// ---------- boot ----------
async function boot() {
  const sel = document.getElementById("run-select");
  try {
    state.runs = await api("/runs");
  } catch (e) {
    app.replaceChildren(h("div", { class: "empty error" }, icon("plugs"), h("b", {}, "The explorer API is not reachable"), h("span", {}, String(e.message || e))));
    return;
  }
  state.liveNli = await api("/health").then((x) => !!x.live_nli).catch(() => false);
  let saved = null;
  try { saved = localStorage.getItem("pe-run"); } catch {}
  state.run = state.runs.find((r) => r.name === saved) || state.runs.find((r) => r.has_metrics) || state.runs[0] || null;
  sel.replaceChildren(...state.runs.map((r) => h("option", { value: r.name, selected: r === state.run }, r.name)));
  sel.disabled = state.runs.length < 2;
  sel.addEventListener("change", () => {
    state.run = state.runs.find((r) => r.name === sel.value);
    try { localStorage.setItem("pe-run", state.run.name); } catch {}
    route();
  });
  window.addEventListener("hashchange", route);
  route();
}
boot();
