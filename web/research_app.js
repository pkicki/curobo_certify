/* Research report app. Evidence levels are always labelled:
 *  EXACT  = exact rational computation (serialized model) - in-browser BigInt or recorded exact runs
 *  NATIVE = native-reported certificate, conditional on documented FP32 semantics A1-A5
 *  SAMPLED= finite native samples / numerical diagnostics, never a certificate
 *  BLOCKED / NOT RUN as recorded. No number on this page is synthesized. */
(function () {
'use strict';
const D = window.REPORT_DATA, P = window.BernsteinProof, M = window.ModelCore;
const $ = (s, r = document) => r.querySelector(s);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const L = {exact: '<span class="pill L-exact">EXACT</span>', native: '<span class="pill L-native">NATIVE (A1–A5)</span>',
  sampled: '<span class="pill L-sampled">SAMPLED</span>', blocked: '<span class="pill L-blocked">BLOCKED</span>',
  notrun: '<span class="pill L-notrun">NOT RUN</span>',
  audited: '<span class="pill L-audit">AUDITED (every float32 t)</span>'};
const fmt = (v, d = 3) => v === null || v === undefined || Number.isNaN(v) ? '—' : (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : (+v).toFixed(d));
const med = a => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor((s.length - 1) / 2)] : NaN; };
const pct = (a, q) => { const s = [...a].sort((x, y) => x - y); if (!s.length) return NaN; const i = (s.length - 1) * q / 100, lo = Math.floor(i); return s[lo] + (s[Math.min(lo + 1, s.length - 1)] - s[lo]) * (i - lo); };
$('#legend').innerHTML = `${L.exact} ${L.native} ${L.audited} ${L.sampled} ${L.blocked} ${L.notrun}`;

/* ---------------- SVG helpers ---------------- */
const tick = v => v === 0 ? '0' : Math.abs(v) >= 1e4 || Math.abs(v) < 1e-2 ? Number(v).toExponential(1) : String(Number(v.toPrecision(3)));
function frame(W, H, xr, yr, opts = {}) {
  const l = 58, r = 14, t = 14, b = 34;
  const X = x => l + (x - xr[0]) / (xr[1] - xr[0] || 1) * (W - l - r);
  const Y = y => t + (yr[1] - y) / (yr[1] - yr[0] || 1) * (H - t - b);
  let s = '';
  for (let i = 0; i <= 4; i++) {
    const yv = yr[0] + (yr[1] - yr[0]) * i / 4, xv = xr[0] + (xr[1] - xr[0]) * i / 4;
    s += `<line x1="${l}" x2="${W - r}" y1="${Y(yv)}" y2="${Y(yv)}" stroke="#22344d" stroke-dasharray="2 5"/>`;
    s += `<text x="${l - 6}" y="${Y(yv) + 4}" text-anchor="end">${tick(opts.ylog ? 10 ** yv : yv)}</text>`;
    s += `<text x="${X(xv)}" y="${H - 12}" text-anchor="middle">${tick(opts.xlog ? 10 ** xv : xv)}</text>`;
  }
  if (opts.xlabel) s += `<text x="${(l + W - r) / 2}" y="${H - 1}" text-anchor="middle">${esc(opts.xlabel)}</text>`;
  return {X, Y, s, W, H};
}
function svg(W, H, inner, label) { return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(label || '')}">${inner}</svg>`; }
const poly = (pts, stroke, w = 2, dash = '') => `<polyline fill="none" stroke="${stroke}" stroke-width="${w}" ${dash ? `stroke-dasharray="${dash}"` : ''} points="${pts.map(p => p.join(',')).join(' ')}"/>`;
const COLORS = ['#6be0d9', '#ffcb7f', '#c9a7ff', '#ff899f', '#8bacff', '#77e2b8', '#f2a6e0', '#d6e685', '#9fd3ff'];

/* ---------------- tabs ---------------- */
const TABS = [['overview', 'Answers'], ['margins', 'Planning margins'], ['timelaw', 'Native time law'], ['explorer', 'Trajectory explorer'], ['population', 'Real-path population'],
  ['adaptive', 'Adaptive vs uniform'], ['latency', 'Latency'], ['native', 'Native linkage'], ['correctness', 'Correctness'], ['provenance', 'Provenance']];
const sections = {};
for (const [id, title] of TABS) {
  const b = document.createElement('button'); b.textContent = title; b.setAttribute('role', 'tab'); b.dataset.tab = id;
  b.onclick = () => show(id); $('#tabs').appendChild(b);
  const sec = document.createElement('section'); sec.id = 'tab-' + id; sec.className = 'hidden'; $('#main').appendChild(sec); sections[id] = sec;
}
let rendered = {};
function show(id) {
  for (const b of $('#tabs').children) b.setAttribute('aria-selected', b.dataset.tab === id);
  for (const k in sections) sections[k].classList.toggle('hidden', k !== id);
  if (!rendered[id]) { rendered[id] = true; RENDER[id](sections[id]); }
  try { localStorage.setItem('report-tab', id); } catch (e) { /* storage unavailable */ }
}

/* ---------------- overview ---------------- */
function overview(el) {
  const A = D.data.answers;
  const nr = D.data.native_reported, tz = D.data.tightness;
  let h = '<div class="card"><h2>Headline evidence</h2><div class="grid">';
  if (nr) {
    const B = D.data.bench_native_all;
    const g = B ? B.rows.filter(r => r.contract === 'native' && r.method === 'A' && !r.path.includes('margin_arm')) : [];
    h += `<div class="kpi"><small>Native-reported K3 certificates ${L.native}</small><b>${g.length ? `${g.filter(r => r.final === 'CERTIFIED').length} / ${g.length}` : `${nr.certified} / ${nr.denominator}`}</b><small>default-margin native outputs (compiled + verifier); Python reference ${nr.certified}/${nr.denominator}</small></div>`;
  }
  if (tz) h += `<div class="kpi"><small>Real paths analysed exactly ${L.exact}</small><b>${tz.paths}</b><small>physical: ${tz.physical_paths_needing_subdivision} need subdivision, ${tz.physical_paths_with_genuine_violation} genuinely violate</small></div>`;
  const c = D.data.campaigns && D.data.campaigns.O3;
  if (c) h += `<div class="kpi"><small>Compiled producer+verifier vs exact reference ${L.exact}</small><b>${c.breaches} breaches</b><small>${c.cases.toLocaleString()} seeded adversarial cases (-O3)</small></div>`;
  const f = D.data.bench_summary && D.data.bench_summary.fresh && D.data.bench_summary.fresh[0];
  if (f) h += `<div class="kpi"><small>Fresh request added latency p50 / p95 (${esc(f.gpu)})</small><b>${fmt(f.added_ms.p50, 2)} / ${fmt(f.added_ms.p95, 2)} ms</b><small>targets ≤1 / ≤3 ms: ${f.targets.added_p50_le_1ms ? 'met' : 'NOT met'} / ${f.targets.added_p95_le_3ms ? 'met' : 'NOT met'}</small></div>`;
  else h += `<div class="kpi"><small>Fresh request latency</small><b>${L.notrun}</b></div>`;
  h += '</div></div>';
  const MA = D.data.margin_answers;
  if (MA) {
    h += `<div class="card"><h2>Campaign 2026-11-17: planning margins, native time law, latency</h2>`;
    for (const q of MA.questions) h += `<div class="answer"><b>${esc(q.q)}</b><p>${q.a}</p></div>`;
    h += `<div class="answer"><b>Decisions</b><p>${MA.decision}</p></div></div>`;
  }
  h += '<div class="card"><h2>Answers to the ten research questions (campaign 2026-11-16)</h2>';
  if (!A) h += `<p>${L.notrun} answers file missing.</p>`;
  else for (const q of A.questions) h += `<div class="answer"><b>${esc(q.q)}</b><p>${q.a}</p></div>`;
  if (A) h += `<div class="answer"><b>Decision</b><p>${A.decision}</p></div>`;
  h += '</div>';
  el.innerHTML = h;
}

/* ---------------- explorer ---------------- */
const modelCache = {};
function getModel(id) {
  if (!modelCache[id]) { const p = D.paths.find(x => x.id === id); modelCache[id] = {p, m: M.build(p, D.table)}; }
  return modelCache[id];
}
function f32bits(x) { const dv = new DataView(new ArrayBuffer(4)); dv.setFloat32(0, x, true); return dv.getUint32(0, true); }
function contractLimits(p, contract) {
  if (contract === 'physical') return {exact: p.limits, float: p.limits_float};
  const exact = {}, flt = {};
  for (const k in p.limits_float) {
    const sc = Math.fround(p.planning_scale || 0.95);  // this output's source-hash-backed TimeOpt scale
    flt[k] = p.limits_float[k].map(v => Math.fround(Math.fround(v) * sc));  // float32 product, as analysed
    exact[k] = flt[k].map(v => P.repr(M.f32(f32bits(v))));
  }
  return {exact, float: flt};
}
const ST = {path: null, contract: 'physical', family: 'jerk', joint: 0, span: 0, depth: 12, nodes: 4096, focus: -1, proof: null};
function explorer(el) {
  el.innerHTML = `<div class="card"><h2>Real native TimeOpt output explorer ${L.exact}</h2>
  <p class="sub">Select a frozen real output. The model is rebuilt <b>exactly</b> from its float32 controls (BigInt rationals); the selected residual's Bernstein hulls, proof tree, peak bracket and witnesses are exact for the serialized model. Native samples are ${L.sampled} only. Whole-path status comes from the recorded exact analysis.</p>
  <div class="controls">
   <div><label for="xp">Trajectory</label><select id="xp"></select></div>
   <div><label for="xc">Contract</label><select id="xc"><option value="physical">physical limits</option><option value="planning">solver planning-scaled (path's own scale)</option></select></div>
   <div><label for="xf">Constraint family</label><select id="xf">${M.FAMILIES.map(f => `<option>${f}</option>`).join('')}</select></div>
   <div><label for="xj">Joint</label><select id="xj"></select></div>
   <div><label for="xs">Common span <span id="xsv"></span></label><input id="xs" type="range" min="0" max="46" value="0"></div>
   <div><label for="xd">Max depth</label><input id="xd" type="number" min="0" max="20" value="12" style="width:70px"></div>
   <div><button class="btn" id="xworst">Jump to hardest residual</button></div>
  </div><div id="xstatus" style="margin-top:10px"></div></div>
  <div class="two"><div class="card"><h3>Physical quantity along the path (selected joint)</h3><div id="xphys"></div>
   <p class="sub" style="font-size:12px">Curve: exact model evaluated in floating point for display. Dots: native S32768 samples ${L.sampled} (subsampled, per-joint extrema kept). Shaded band: selected common span.</p></div>
   <div class="card"><h3>Residual on selected span r(u) ≤ 0, u∈[0,1]</h3><div id="xres"></div><div id="xbr" class="mono" style="font-size:12px"></div></div></div>
  <div class="card"><h3>Adaptive proof tree (exact, breadth-first) — click a node</h3><div id="xtree" class="tree"></div><div id="xinsp" class="mono" style="font-size:12px"></div></div>`;
  const sel = $('#xp', el);
  sel.innerHTML = D.paths.map(p => `<option value="${esc(p.id)}">${esc(p.id)} · scale ${fmt(p.planning_scale || 0.95, 3)} · ${esc(p.scene || '')}</option>`).join('');
  const go = () => { readControls(); runExplorer(); };
  for (const id of ['#xp', '#xc', '#xf', '#xj', '#xd']) $(id, el).addEventListener('change', go);
  $('#xs', el).addEventListener('input', go);
  $('#xworst', el).onclick = () => { jumpWorst(); };
  ST.path = D.paths[0].id; syncControls(); runExplorer();
}
function syncControls() {
  const {m} = getModel(ST.path);
  $('#xp').value = ST.path; $('#xc').value = ST.contract; $('#xf').value = ST.family;
  $('#xj').innerHTML = Array.from({length: m.D}, (_, j) => `<option value="${j}">${j}</option>`).join('');
  $('#xj').value = String(ST.joint); $('#xj').disabled = ST.family === 'positive_x';
  $('#xs').max = String(m.spans.length - 1); $('#xs').value = String(Math.min(ST.span, m.spans.length - 1)); $('#xd').value = String(ST.depth);
}
function readControls() {
  const changedPath = $('#xp').value !== ST.path;
  ST.path = $('#xp').value; ST.contract = $('#xc').value; ST.family = $('#xf').value;
  ST.joint = +$('#xj').value || 0; ST.span = changedPath ? 0 : +$('#xs').value; ST.depth = Math.max(0, Math.min(20, +$('#xd').value || 0));
  ST.focus = -1; syncControls();
}
function tightRow(id) {
  for (const src of [D.data.tightness, D.data.margin_tightness]) {
    const r = src && src.rows.find(x => x.path === id);
    if (r) return r;
  }
  return null;
}
function jumpWorst() {
  const tz = tightRow(ST.path);
  const rows = tz ? tz.contracts[ST.contract].nontrivial_residuals : [];
  if (!rows.length) { $('#xstatus').innerHTML += ' <span class="pill L-notrun">no residual needs subdivision under this contract</span>'; return; }
  const w = rows.reduce((a, b) => (b.peak_lower > a.peak_lower ? b : a));
  ST.family = w.family; ST.joint = Math.max(0, w.joint); ST.span = w.span; ST.focus = -1; syncControls(); runExplorer();
}
function runExplorer() {
  const {p, m} = getModel(ST.path);
  const lim = contractLimits(p, ST.contract);
  const [Ls, Rs] = m.spans[ST.span];
  $('#xsv').textContent = `${ST.span} · s∈[${P.repr(Ls)}, ${P.repr(Rs)}]`;
  const tz = tightRow(ST.path);
  const cc = tz ? tz.contracts[ST.contract] : null;
  $('#xstatus').innerHTML = cc ? `Whole-path exact residual classes (${ST.contract}) ${L.exact}: ${Object.entries(cc.class_counts).map(([k, v]) => `<span class="pill ${k === 'GENUINE_VIOLATION' ? 'L-blocked' : k === 'H0_CERTIFIED' ? 'L-exact' : 'L-sampled'}">${k} ${v}</span>`).join(' ')} · native duration ${fmt(p.duration, 4)} s · sampled max utilization v/a/j ${L.sampled} ${['velocity', 'acceleration', 'jerk'].map(k => fmt(cc.sampled_max_utilization_S32768[k], 4)).join(' / ')}` : `${L.notrun} no recorded whole-path analysis`;
  drawPhysical(p, m, lim);
  // exact residual + proof
  const r = M.residual(m, ST.span, ST.family, ST.joint, lim.exact);
  const b = M.bernstein(r.power);
  const strict = ST.family === 'positive_x';
  ST.proof = P.build(b, {depth: ST.depth, nodes: ST.nodes, strict, strictLeft: strict && ST.span > 0, strictRight: strict && ST.span < m.spans.length - 1});
  drawResidual(b, r);
}
function drawPhysical(p, m, lim) {
  const W = 760, H = 300, fam = ST.family, j = ST.joint;
  const key = fam === 'velocity' ? 'v' : fam.startsWith('acc') ? 'a' : fam === 'jerk' ? 'j' : 'x';
  const xs = [], ys = [];
  for (let i = 0; i <= 600; i++) { const s = i / 600; xs.push(s); ys.push(M.physical(m, j, s)[key]); }
  let pts = xs.map((s, i) => [s, ys[i]]);
  let limv = null;
  if (key !== 'x') limv = lim.float[{v: 'velocity', a: 'acceleration', j: 'jerk'}[key]][j];
  const nat = [];
  if (key !== 'x') {
    const pr = M.decode(p.samples.progress).values, vaj = M.decode(p.samples.vaj).values, n = p.samples.count, Dn = p.shape[1];
    const fi = {v: 0, a: 1, j: 2}[key];
    for (let i = 0; i < n; i++) nat.push([pr[i], vaj[(fi * n + i) * Dn + j]]);
  }
  const all = ys.concat(nat.map(q => q[1])).concat(limv !== null ? [limv, -limv] : [0]);
  const lo = Math.min(...all), hi = Math.max(...all), pad = 0.08 * (hi - lo || 1);
  const F = frame(W, H, [0, 1], [lo - pad, hi + pad], {xlabel: 'path progress s'});
  const [Ls, Rs] = m.spans[ST.span];
  let s = F.s + `<rect x="${F.X(P.num(Ls))}" y="14" width="${Math.max(2, F.X(P.num(Rs)) - F.X(P.num(Ls)))}" height="${H - 48}" fill="#6be0d922"/>`;
  if (limv !== null) for (const v of [limv, -limv]) s += `<line x1="58" x2="${W - 14}" y1="${F.Y(v)}" y2="${F.Y(v)}" stroke="#ff899f" stroke-dasharray="7 5" stroke-width="1.5"/>`;
  s += poly(pts.map(([a, c]) => [F.X(a), F.Y(c)]), '#6be0d9', 2);
  for (const [a, c] of nat) s += `<circle cx="${F.X(a)}" cy="${F.Y(c)}" r="1.8" fill="#ffcb7f" opacity=".75"/>`;
  $('#xphys').innerHTML = svg(W, H, s, 'physical quantity along path') + `<div class="legend"><span>— model ${key} (joint ${j})</span><span style="color:#ffcb7f">• native samples (${esc(p.samples.label.split('x = ')[1] || '')})</span><span style="color:#ff899f">- - ${ST.contract} limit ${limv !== null ? fmt(limv, 4) : ''}</span></div>`;
}
function drawResidual(b, r) {
  const pr = ST.proof, W = 760, H = 300;
  const vals = []; for (let i = 0; i <= 300; i++) vals.push([i / 300, P.sample(b, i / 300)]);
  const shown = pr.nodes.filter(n => ['PROVEN', 'VIOLATED', 'UNKNOWN_DEPTH', 'PENDING'].includes(n.state));
  const ys = vals.map(v => v[1]).concat(b.map(P.num)).concat([0]);
  for (const n of shown) ys.push(P.num(n.lower), P.num(n.upper));
  const finite = ys.filter(Number.isFinite), lo = Math.min(...finite), hi = Math.max(...finite), pad = 0.1 * (hi - lo || 1);
  const F = frame(W, H, [0, 1], [lo - pad, hi + pad], {xlabel: 'local span parameter u'});
  let s = F.s + `<line x1="58" x2="${W - 14}" y1="${F.Y(0)}" y2="${F.Y(0)}" stroke="#ff899f" stroke-dasharray="7 5"/>`;
  const root = pr.nodes[0];
  s += `<rect x="${F.X(0)}" y="${F.Y(P.num(root.upper))}" width="${F.X(1) - F.X(0)}" height="${Math.max(1, F.Y(P.num(root.lower)) - F.Y(P.num(root.upper)))}" fill="none" stroke="#c9a7ff" stroke-dasharray="4 3"><title>initial (depth-0) Bernstein hull</title></rect>`;
  for (const n of shown) {
    const col = n.state === 'PROVEN' ? '#77e2b8' : n.state === 'VIOLATED' ? '#ff899f' : '#ffcb7f';
    s += `<rect x="${F.X(P.num(n.l))}" y="${F.Y(P.num(n.upper))}" width="${Math.max(1, F.X(P.num(n.r)) - F.X(P.num(n.l)))}" height="${Math.max(1, F.Y(P.num(n.lower)) - F.Y(P.num(n.upper)))}" fill="${col}30" stroke="${col}"><title>${esc(n.path || 'root')} [${esc(P.repr(n.lower))}, ${esc(P.repr(n.upper))}]</title></rect>`;
  }
  const focus = ST.focus >= 0 ? pr.nodes[ST.focus] : root;
  const nn = focus.b.length - 1;
  s += poly(focus.b.map((c, i) => [F.X(P.num(focus.l) + (P.num(focus.r) - P.num(focus.l)) * i / nn), F.Y(P.num(c))]), '#ffcb7f', 1.4, '5 3');
  s += poly(vals.map(([u, v]) => [F.X(u), F.Y(v)]), '#6be0d9', 2.2);
  if (pr.witness) s += `<circle cx="${F.X(P.num(pr.witness.at))}" cy="${F.Y(P.num(pr.witness.value))}" r="7" fill="#ff899f"><title>exact witness</title></circle>`;
  $('#xres').innerHTML = svg(W, H, s, 'residual with Bernstein hulls') + `<div class="legend"><span style="color:#c9a7ff">▭ initial hull</span><span style="color:#77e2b8">▭ proven leaf</span><span style="color:#ffcb7f">- - control polygon (focused node)</span><span style="color:#6be0d9">— residual (float display)</span></div>`;
  const br = P.peakBracket(b, 6), u = P.uniform(b, 0);
  let umin = '—';
  for (let k = 0; k <= 8; k++) { const uu = P.uniform(b, k); if (uu.status === 'CERTIFIED') { umin = `U${k}: ${uu.cells} cells`; break; } }
  const sp = r.sp, h = P.num(sp.h);
  $('#xbr').innerHTML = `<p><b>Status ${pr.status}</b> ${L.exact} · degree ${pr.degree} · nodes ${pr.nodes.length} · leaves ${pr.leaves.length} · replay ${pr.replay.pass ? 'PASS (shared arithmetic)' : '—'}</p>
   <p>Initial hull upper U0 = ${esc(P.repr(u.worstUpper))} ≈ ${fmt(P.num(u.worstUpper), 6)} · exact peak bracket [L*, U*] (64 cells) = [${fmt(P.num(br.lower), 6)}, ${fmt(P.num(br.upper), 6)}] · hull overestimate ∈ [${fmt(P.num(br.gapLower), 6)}, ${fmt(P.num(br.gapUpper), 6)}]</p>
   <p>Uniform comparison: smallest certifying level ${umin}; adaptive used ${pr.nodes.length} nodes / ${pr.leaves.length} leaves.</p>
   ${pr.witness ? `<p style="color:#ff899f">Exact witness at u = ${esc(P.repr(pr.witness.at))} (s ≈ ${fmt(P.num(sp.L) + h * P.num(pr.witness.at), 6)}), r = ${esc(P.repr(pr.witness.value))} ≈ ${fmt(P.num(pr.witness.value), 6)} &gt; 0: a genuine violation of the ${esc(ST.contract)} ${esc(ST.family)} limit by the exact serialized model.</p>` : ''}`;
  $('#xtree').innerHTML = pr.nodes.slice(0, 400).map((n, i) => `<button class="node ${n.state}${i === ST.focus ? ' sel' : ''}" data-i="${i}">${esc(n.path || 'root')}</button>`).join('') + (pr.nodes.length > 400 ? ` … ${pr.nodes.length - 400} more` : '');
  for (const btn of $('#xtree').querySelectorAll('button')) btn.onclick = () => { ST.focus = +btn.dataset.i; drawResidual(b, r); };
  const n = focus;
  $('#xinsp').innerHTML = `Node ${esc(n.path || 'root')} · ${n.state} · u∈[${esc(P.repr(n.l))}, ${esc(P.repr(n.r))}] (s∈[${fmt(P.num(sp.L) + h * P.num(n.l), 6)}, ${fmt(P.num(sp.L) + h * P.num(n.r), 6)}]) · exact coefficient bounds [${esc(P.repr(n.lower))}, ${esc(P.repr(n.upper))}]<br>coefficients: ${n.b.map(c => esc(P.repr(c))).join(', ')}`;
}

/* ---------------- planning margins ---------------- */
function margins(el) {
  const A = D.data.margin_analysis;
  if (!A) { el.innerHTML = `<div class="card">${L.notrun} margin analysis not available.</div>`; return; }
  const ms = Object.keys(A.summary_by_margin).sort((a, b) => +a - +b);
  let h = `<div class="card"><h2>Matched planning-margin sweep ${L.exact}</h2>
  <p class="sub">${A.denominators.source_queries} registered requests = ${A.denominators.pairs} frozen source queries × ${ms.length} TimeOpt planning scales (limit_scale = jerk_limit_scale; physical limits unchanged; constructed native config audited). Exact certificates are for the serialized model (compiled producer + independent verifier); native-reported certificates are ${L.native}; the physical-time column is the exhaustive time-law audit (see the next tab). Durations are native outputs.</p>
  <table><tr><th>margin</th><th>submitted</th><th>native success</th><th>failed</th><th>duration p50 [s]</th><th>exact model K3 certified</th><th>exact violated</th><th>needs subdivision</th><th>violates solver's own scaled contract</th><th>worst physical witness residual</th><th>native-reported certified</th><th>native: unknown (tightened)</th><th>time-law audited &amp; certified</th><th>native cert. p50 [ms] (1 thread)</th></tr>`;
  for (const m of ms) {
    const s = A.summary_by_margin[m];
    h += `<tr><td>${m}</td><td class="num">${s.attempted}</td><td class="num">${s.native_success}</td><td class="num">${s.native_failed}</td><td class="num">${fmt(s.duration_p50_s_success_only, 4)}</td><td class="num">${s.model_certified}</td><td class="num" style="color:${s.model_violated ? '#ff899f' : ''}">${s.model_violated}</td><td class="num">${s.physical_paths_needing_subdivision ?? '—'}</td><td class="num">${s.solver_contract_paths_genuinely_violating ?? '—'}</td><td class="num">${s.worst_physical_witness_residual == null ? '—' : fmt(s.worst_physical_witness_residual, 5)}</td><td class="num">${s.native_reported_certified}</td><td class="num">${s.native_reported_unknown_tightened}</td><td class="num">${s.native_time_linked_audited_certified ?? '—'}</td><td class="num">${fmt(s.native_reported_latency_p50_ms, 2)}</td></tr>`;
  }
  h += '</table></div>';
  // forest plot of paired gains
  const W = 760, H = 240, cmp = A.paired_to_0p95, keys = Object.keys(cmp).sort((a, b) => +a - +b);
  const vals = keys.flatMap(k => (cmp[k].duration_gain_pct_paired_success_only.ci95 || [])).concat([0]);
  const F = frame(W, H, [Math.min(...vals) - 0.5, Math.max(...vals) + 0.5], [0, keys.length + 1], {xlabel: 'paired duration change vs 0.95 [%] (positive = faster); median and 95% pair-bootstrap CI'});
  let s = F.s + `<line x1="${F.X(0)}" x2="${F.X(0)}" y1="14" y2="${H - 34}" stroke="#ff899f" stroke-dasharray="5 4"/>`;
  keys.forEach((k, i) => {
    const b = cmp[k].duration_gain_pct_paired_success_only, y = F.Y(keys.length - i);
    if (b.ci95) s += `<line x1="${F.X(b.ci95[0])}" x2="${F.X(b.ci95[1])}" y1="${y}" y2="${y}" stroke="#6be0d9" stroke-width="3"/>`;
    const right = F.X(b.median) > W * 0.6;
    if (b.median != null) s += `<circle cx="${F.X(b.median)}" cy="${y}" r="6" fill="#ffcb7f"/><text x="${F.X(b.median) + (right ? -10 : 10)}" y="${y - 10}" text-anchor="${right ? 'end' : 'start'}">${k}: ${fmt(b.median, 2)}% (n=${b.n})</text>`;
  });
  h += `<div class="card"><h2>Paired duration change relative to the pinned 0.95 margin</h2>${svg(W, H, s, 'forest plot')}<p class="sub">Missingness per arm: ${keys.map(k => `${k}: baseline-only ${cmp[k].missingness.baseline_only_success}, candidate-only ${cmp[k].missingness.candidate_only_success}, neither ${cmp[k].missingness.both_failed_or_not_run}`).join(' · ')}. Exploratory; no margin is selected as a policy from these data.</p></div>`;
  // per-pair duration lines
  const rows = A.rows.filter(r => r.native_duration_s != null);
  const pairs = [...new Set(rows.map(r => r.pair_id))];
  const F2 = frame(W, 300, [+ms[0], +ms[ms.length - 1]], [Math.min(...rows.map(r => r.native_duration_s)) * 0.97, Math.max(...rows.map(r => r.native_duration_s)) * 1.03], {xlabel: 'planning margin (TimeOpt limit scale)'});
  let s2 = F2.s;
  pairs.forEach((pid, i) => {
    const pts = rows.filter(r => r.pair_id === pid).sort((a, b) => +a.margin - +b.margin);
    s2 += poly(pts.map(r => [F2.X(+r.margin), F2.Y(r.native_duration_s)]), COLORS[i % COLORS.length], 1.2);
    for (const r of pts) {
      const st = r.exact_tightness ? (r.exact_tightness.physical.genuine_violations ? '#ff899f' : (r.exact_tightness.physical.split_certifiable ? '#ffcb7f' : '#77e2b8')) : '#9baecb';
      s2 += `<circle cx="${F2.X(+r.margin)}" cy="${F2.Y(r.native_duration_s)}" r="4" fill="${st}"><title>${esc(r.source_id)} T=${fmt(r.native_duration_s, 5)} s</title></circle>`;
    }
  });
  h += `<div class="card"><h2>Duration of each source query across margins</h2>${svg(W, 300, s2, 'duration by margin')}<div class="legend"><span style="color:#77e2b8">● physical K3 at depth 0</span><span style="color:#ffcb7f">● needs adaptive subdivision</span><span style="color:#ff899f">● genuine physical violation (exact witness)</span></div><p class="sub">Click-through: choose the source ID in the trajectory explorer to inspect its residuals, hulls and proof trees.</p></div>`;
  el.innerHTML = h;
}

/* ---------------- native time law ---------------- */
function timelaw(el) {
  const T = D.data.time_audit;
  if (!T) { el.innerHTML = `<div class="card">${L.notrun} exhaustive time-law audit not available.</div>`; return; }
  const ok = T.rows.filter(r => r.eps_s_certified_max != null);
  const eps = ok.map(r => r.eps_s_certified_max);
  let h = `<div class="card"><h2>Native time → progress inversion, every float32 requested time</h2>
  <p class="sub">For each frozen native output, every float32 requested time t in [0, T_native] (≈10<sup>9</sup> values per path) was run through the pinned upstream <code>progress_at_time</code> on ${esc(T.gpu)} and compared with a rigorous enclosure of the exact inverse s*(t) of the mathematical time law ∫ds/√x (2<sup>22</sup>-cell outward-rounded grid ${L.exact}). The same enumeration checked every reported native v/a/j against the physical limits. For these two results the native values are observed on the GPU, so no FP32 rounding assumption is needed; they rely on the harness executing the same compiled upstream code as production (bitwise reproduction of saved production samples) and on the rigorous CPU time-law enclosure.</p>
  <p class="sub">Physical-time linkage: a native sample at time t differs from the exact continuous trajectory's state at the same t by at most E(same-progress, ${L.native}) + L·ε_s (inversion shift, shown below). Jerk is one-sided at geometry knots: near a knot (within ε_s in progress) the native jerk equals the exact trajectory's jerk on the other side of the knot.</p>
  <div class="grid"><div class="kpi"><small>paths audited</small><b>${ok.length}</b></div><div class="kpi"><small>max certified |s_native − s*|</small><b>${fmt(Math.max(...eps), 3)}</b></div><div class="kpi"><small>median certified |s_native − s*|</small><b>${fmt(med(eps), 3)}</b></div><div class="kpi"><small>reported values within physical limits at every float32 time</small><b>${ok.filter(r => r.exhaustive_reported_status === 'CERTIFIED_EXHAUSTIVE').length} / ${ok.length}</b></div></div>
  <table><tr><th>path</th><th>float32 times</th><th>certified ε_s</th><th>CPU cross-check</th><th>inversion shift L·ε_s / v_lim</th><th>… / a_lim</th><th>… / j_lim (off-knot)</th><th>max reported |v|/v_lim</th><th>|a|/a_lim</th><th>|j|/j_lim</th><th>status</th></tr>`;
  for (const r of ok) {
    const R = r.exhaustive_reported;
    h += `<tr><td>${esc(r.path)}</td><td class="num">${r.times_enumerated.toLocaleString()}</td><td class="num">${fmt(r.eps_s_certified_max, 3)}</td><td>${r.crosscheck_consistent ? 'consistent' : 'INCONSISTENT'}</td><td class="num">${fmt(r.relative_deviation_vs_limits.v, 2)}</td><td class="num">${fmt(r.relative_deviation_vs_limits.a, 2)}</td><td class="num">${fmt(r.relative_deviation_vs_limits.j, 2)}</td>${['velocity', 'acceleration', 'jerk'].map(k => `<td class="num" style="color:${R[k].max_utilization > 1 ? '#ff899f' : ''}">${Number(R[k].max_utilization).toFixed(7)}</td>`).join('')}<td>${r.exhaustive_reported_status === 'CERTIFIED_EXHAUSTIVE' ? L.audited : '<span class="pill L-blocked">EXCEEDS LIMIT</span>'}</td></tr>`;
  }
  el.innerHTML = h + '</table></div>';
}

/* ---------------- population ---------------- */
function population(el) {
  const tz = D.data.tightness;
  if (!tz) { el.innerHTML = `<div class="card">${L.notrun}</div>`; return; }
  const meta = Object.fromEntries(D.paths.map(p => [p.id, p]));
  const rows = tz.rows.map(r => {
    const p = meta[r.path] || {};
    const cnt = c => r.contracts[c].class_counts;
    return {id: r.path, cond: p.condition, scene: p.scene, mag: p.magnitude, T: r.native_duration_s,
      ps: cnt('physical').SPLIT_CERTIFIABLE || 0, pv: cnt('physical').GENUINE_VIOLATION || 0,
      qs: cnt('planning').SPLIT_CERTIFIABLE || 0, qv: cnt('planning').GENUINE_VIOLATION || 0,
      uj: r.contracts.physical.sampled_max_utilization_S32768.jerk, ua: r.contracts.physical.sampled_max_utilization_S32768.acceleration,
      uv: r.contracts.physical.sampled_max_utilization_S32768.velocity};
  });
  const cond = c => rows.filter(r => r.cond === c);
  const summ = (rs, k) => `${rs.filter(r => r[k] > 0).length}/${rs.length}`;
  let h = `<div class="card"><h2>Exact residual classes on ${rows.length} real native outputs ${L.exact}</h2>
  <p class="sub">Each of ${tz.rows[0] ? (tz.rows[0].common_spans * 29) : '—'} residuals per path is classified exactly: H0_CERTIFIED (initial hull ≤ 0), SPLIT_CERTIFIABLE (initial hull inconclusive but exact peak bracket ≤ 0), GENUINE_VIOLATION (exact positive witness), UNRESOLVED. Sampled utilization is ${L.sampled}.</p>
  <table><tr><th>population</th><th>paths</th><th>physical: need split</th><th>physical: genuine violation</th><th>planning 0.95×: need split</th><th>planning 0.95×: violation</th></tr>`;
  for (const c of ['default_margin_0p95', 'margin_arm_1p00']) {
    const rs = cond(c);
    h += `<tr><td>${c}</td><td class="num">${rs.length}</td><td class="num">${summ(rs, 'ps')}</td><td class="num">${summ(rs, 'pv')}</td><td class="num">${summ(rs, 'qs')}</td><td class="num">${summ(rs, 'qv')}${c === 'margin_arm_1p00' ? ' (not this condition\'s contract)' : ''}</td></tr>`;
  }
  h += `</table></div><div class="card"><h2>Per path (click to explore)</h2><table id="poptab"><tr>${['path', 'condition', 'scene', 'magnitude', 'T [s]', 'phys split', 'phys viol', 'plan split', 'plan viol', 'max |v|/v', 'max |a|/a', 'max |j|/j'].map((t, i) => `<th data-k="${i}">${t}</th>`).join('')}</tr>`;
  for (const r of rows) h += `<tr class="click" data-id="${esc(r.id)}"><td>${esc(r.id)}</td><td>${esc(r.cond)}</td><td>${esc(r.scene)}</td><td>${esc(r.mag)}</td><td class="num">${fmt(r.T, 3)}</td><td class="num">${r.ps}</td><td class="num" style="color:${r.pv ? '#ff899f' : ''}">${r.pv}</td><td class="num">${r.qs}</td><td class="num" style="color:${r.qv ? '#ff899f' : ''}">${r.qv}</td><td class="num">${fmt(r.uv, 4)}</td><td class="num">${fmt(r.ua, 4)}</td><td class="num" style="color:${r.uj > 1 ? '#ff899f' : ''}">${fmt(r.uj, 4)}</td></tr>`;
  h += '</table></div>';
  // utilization scatter
  const W = 760, H = 300, F = frame(W, H, [0.85, 1.01], [0, 1], {xlabel: 'sampled max |jerk|/physical limit (SAMPLED)'});
  let s = F.s;
  rows.forEach((r, i) => { s += `<circle cx="${F.X(Math.max(0.85, Math.min(1.01, r.uj)))}" cy="${F.Y((i + 0.5) / rows.length)}" r="4" fill="${r.pv ? '#ff899f' : r.ps ? '#ffcb7f' : '#77e2b8'}"><title>${esc(r.id)}</title></circle>`; });
  s += `<line x1="${F.X(1)}" x2="${F.X(1)}" y1="14" y2="${H - 34}" stroke="#ff899f" stroke-dasharray="6 4"/>`;
  h += `<div class="card"><h2>Jerk utilization vs exact class</h2>${svg(W, H, s, 'utilization')}<div class="legend"><span style="color:#77e2b8">● H0 certified</span><span style="color:#ffcb7f">● needs subdivision</span><span style="color:#ff899f">● genuine physical violation</span></div></div>`;
  el.innerHTML = h;
  for (const tr of el.querySelectorAll('tr.click')) tr.onclick = () => { ST.path = tr.dataset.id; ST.contract = 'physical'; show('explorer'); syncControls(); jumpWorst(); runExplorer(); };
}

/* ---------------- adaptive vs uniform ---------------- */
function adaptive(el) {
  const B = D.data.bench_native_all;
  if (!B) { el.innerHTML = `<div class="card">${L.notrun} frozen benchmark file not available.</div>`; return; }
  const methods = ['H0', 'U1', 'U2', 'U4', 'U6', 'A', 'A-nw'];
  const contracts = [...new Set(B.rows.map(r => r.contract))];
  let h = `<div class="card"><h2>Matched methods on identical frozen outputs ${L.exact}</h2><p class="sub">Compiled producer + independent verifier (native contract adds the compiled FP32 bound). Outcome counts are exact certificates/verdicts; times are medians of ${B.repeats} repeats after ${B.warmups} warmups (${esc(B.cpu || '')}).</p>`;
  for (const c of contracts) {
    for (const cond of ['default', 'margin']) {
      const rows = B.rows.filter(r => r.contract === c && (cond === 'margin') === r.path.includes('margin_arm'));
      if (!rows.length) continue;
      h += `<h3>${esc(c)} contract · ${cond === 'margin' ? 'margin-arm 1.00 outputs' : 'default-margin outputs'} (${new Set(rows.map(r => r.path)).size} paths)</h3><table><tr><th>method</th><th>CERTIFIED</th><th>VIOLATED</th><th>UNKNOWN</th><th>mean nodes</th><th>mean leaves</th><th>p50 ms</th><th>p95 ms</th></tr>`;
      for (const m of methods) {
        const rs = rows.filter(r => r.method === m); if (!rs.length) continue;
        const t = rs.map(r => med(r.repeats_ms_total));
        const cnt = k => rs.filter(r => k(r.final)).length;
        h += `<tr><td>${m}</td><td class="num">${cnt(s => s === 'CERTIFIED')}</td><td class="num">${cnt(s => s.startsWith('VIOLATED'))}</td><td class="num">${cnt(s => s.startsWith('UNKNOWN') || s === 'REJECTED_BY_VERIFIER')}</td><td class="num">${fmt(rs.reduce((a, r) => a + r.nodes, 0) / rs.length, 0)}</td><td class="num">${fmt(rs.reduce((a, r) => a + r.leaves, 0) / rs.length, 0)}</td><td class="num">${fmt(pct(t, 50), 2)}</td><td class="num">${fmt(pct(t, 95), 2)}</td></tr>`;
      }
      h += '</table>';
    }
  }
  // scatter: nodes vs time
  const pts = B.rows.filter(r => methods.includes(r.method)).map(r => [Math.log10(Math.max(1, r.nodes)), Math.log10(Math.max(1e-3, med(r.repeats_ms_total))), r.method]);
  if (pts.length) {
    const W = 760, H = 320, F = frame(W, H, [Math.min(...pts.map(p => p[0])), Math.max(...pts.map(p => p[0]))], [Math.min(...pts.map(p => p[1])), Math.max(...pts.map(p => p[1]))], {xlabel: 'log10 processed nodes', xlog: true, ylog: true});
    let s = F.s; pts.forEach(p => { s += `<circle cx="${F.X(p[0])}" cy="${F.Y(p[1])}" r="3" fill="${COLORS[methods.indexOf(p[2])]}" opacity=".8"/>`; });
    h += `<h3>Certificate cost vs proof size (all paths, contracts)</h3>${svg(W, H, s, 'nodes vs time')}<div class="legend">${methods.map((m, i) => `<span style="color:${COLORS[i]}">● ${m}</span>`).join('')}</div>`;
  }
  el.innerHTML = h + '</div>';
}

/* ---------------- latency ---------------- */
function ecdf(series, W = 760, H = 300, xlabel = 'ms') {
  const all = series.flatMap(s => s.v).filter(Number.isFinite);
  if (!all.length) return '';
  const lo = Math.log10(Math.max(1e-3, Math.min(...all))), hi = Math.log10(Math.max(...all));
  const F = frame(W, H, [lo, hi], [0, 1], {xlabel: xlabel + ' (log scale)', xlog: true});
  let s = F.s;
  series.forEach((ser, i) => { const v = [...ser.v].sort((a, b) => a - b); s += poly(v.map((x, k) => [F.X(Math.log10(Math.max(1e-3, x))), F.Y((k + 1) / v.length)]), COLORS[i % COLORS.length], 2); });
  return svg(W, H, s, 'ECDF') + `<div class="legend">${series.map((ser, i) => `<span style="color:${COLORS[i % COLORS.length]}">— ${esc(ser.name)}</span>`).join('')}</div>`;
}
function latency(el) {
  const S = D.data.bench_summary;
  let h = '<div class="card"><h2>Frozen-output complete certificate latency</h2>';
  const B = D.data.bench_native_all;
  if (B) {
    for (const c of [...new Set(B.rows.map(r => r.contract))]) {
      const ser = ['H0', 'U2', 'U6', 'A', 'A-nw'].map(m => ({name: `${m}`, v: B.rows.filter(r => r.contract === c && r.method === m).map(r => med(r.repeats_ms_total))}));
      h += `<h3>${esc(c)} contract — per-path median total ms (producer + verifier${c === 'native' ? ' + native bound' : ''})</h3>` + ecdf(ser);
    }
  } else h += L.notrun;
  if (S && S.frozen && S.frozen.length) {
    h += '<h3>Cluster-bootstrap summary (10,000 resamples)</h3><table><tr><th>contract | method | condition</th><th>paths</th><th>clusters</th><th>p50</th><th>p50 95% CI</th><th>p95</th><th>p95 95% CI</th><th>p99</th><th>max</th></tr>';
    for (const f of S.frozen) for (const [k, g] of Object.entries(f.groups)) {
      const l = g.latency_ms;
      if (!/\|(H0|U1|U2|U4|U6|A|A-nw)\|/.test(k)) continue;
      h += `<tr><td>${esc(k)}</td><td class="num">${l.paths}</td><td class="num">${l.clusters}</td><td class="num">${fmt(l.p50, 2)}</td><td class="num">[${fmt(l.p50_ci95[0], 2)}, ${fmt(l.p50_ci95[1], 2)}]</td><td class="num">${fmt(l.p95, 2)}</td><td class="num">[${fmt(l.p95_ci95[0], 2)}, ${fmt(l.p95_ci95[1], 2)}]</td><td class="num">${fmt(l.p99, 2)}</td><td class="num">${fmt(l.p100, 2)}</td></tr>`;
    }
    h += '</table>';
  }
  const S1 = D.data.l1_summary;
  if (S1) {
    h += `<h3>L1 optimisation: native bound + producer + verifier on identical frozen outputs (${Object.values(S1.configs)[0].complete_ms.n} paths)</h3><table><tr><th>configuration</th><th>complete p50 [ms]</th><th>95% CI</th><th>p95</th><th>p99</th><th>native bound p50</th><th>paired Δ vs pieces=16, threads=1</th><th>outcomes</th></tr>`;
    for (const [k, v] of Object.entries(S1.configs)) h += `<tr><td>${esc(k)}</td><td class="num">${fmt(v.complete_ms.p50, 2)}</td><td class="num">[${fmt(v.complete_p50_ci95[0], 2)}, ${fmt(v.complete_p50_ci95[1], 2)}]</td><td class="num">${fmt(v.complete_ms.p95, 2)}</td><td class="num">${fmt(v.complete_ms.p99, 2)}</td><td class="num">${fmt(v.native_bound_ms.p50, 2)}</td><td class="num">${v.paired_diff_vs_baseline_ms ? `${fmt(v.paired_diff_vs_baseline_ms.median, 2)} [${fmt(v.paired_diff_vs_baseline_ms.ci95[0], 2)}, ${fmt(v.paired_diff_vs_baseline_ms.ci95[1], 2)}]` : '—'}</td><td>${esc(JSON.stringify(v.statuses))}</td></tr>`;
    h += `</table><p class="sub">Thread-count mismatches in status or bound hash: ${S1.correctness.thread_count_mismatches.length}. Fewer pieces give looser (still rigorous) bounds and fewer native-reported certificates.</p>`;
    if (S1.fresh) {
      const F = S1.fresh;
      h += `<h3>Fresh paired requests on ${esc(F.gpu)} (${F.queries} queries): native only vs + certificate with 1 and 8 threads</h3><table><tr><th>arm</th><th>added p50 [ms]</th><th>95% CI</th><th>p95</th><th>pooled p99</th><th>pooled max</th><th>p50≤1 ms</th><th>p95≤3 ms</th></tr>`;
      for (const [arm, v] of Object.entries(F.arms)) h += `<tr><td>${arm === 'B1' ? '1 thread' : arm === 'B8' ? '8 threads' : esc(arm)}</td><td class="num">${fmt(v.added_ms.p50, 2)}</td><td class="num">[${fmt(v.added_p50_ci95[0], 2)}, ${fmt(v.added_p50_ci95[1], 2)}]</td><td class="num">${fmt(v.added_ms.p95, 2)}</td><td class="num">${fmt(v.pooled_requests.p99, 2)}</td><td class="num">${fmt(v.pooled_requests.max, 2)}</td><td>${F.targets[arm].p50_le_1ms ? 'MET' : 'NOT MET'}</td><td>${F.targets[arm].p95_le_3ms ? 'MET' : 'NOT MET'}</td></tr>`;
      h += `</table><p class="sub">Paired difference (8 threads − 1 thread) per query: median ${fmt(F.paired_last_minus_first_arm_ms.median, 2)} ms [${fmt(F.paired_last_minus_first_arm_ms.ci95[0], 2)}, ${fmt(F.paired_last_minus_first_arm_ms.ci95[1], 2)}]. Native request p50 ${fmt(F.native_ms.p50, 2)} ms. Nondeterministic native outputs across repeats: ${F.nondeterministic_queries.map(esc).join(', ') || 'none'}.</p>`;
    }
  }
  h += '</div><div class="card"><h2>Fresh native requests: added latency of certification (campaign 2026-11-16, single-threaded)</h2>';
  for (const key of ['fresh_3090', 'fresh_a100']) {
    const F = D.data[key];
    if (!F) { h += `<p>${key}: ${L.notrun}</p>`; continue; }
    const fs = S && S.fresh ? S.fresh.find(x => x.gpu === F.gpu) : null;
    h += `<h3>${esc(F.gpu)} · ${esc(F.contract)} contract · ${F.rows.length} queries × ${F.repeats} paired AB/BA repeats</h3>`;
    if (fs) h += `<p>added p50 ${fmt(fs.added_ms.p50, 2)} ms [${fmt(fs.added_ms.p50_ci95[0], 2)}, ${fmt(fs.added_ms.p50_ci95[1], 2)}], p95 ${fmt(fs.added_ms.p95, 2)} ms · native request p50 ${fmt(fs.native_ms.p50, 2)} ms · targets p50≤1 ms <b>${fs.targets.added_p50_le_1ms ? 'MET' : 'NOT MET'}</b>, p95≤3 ms <b>${fs.targets.added_p95_le_3ms ? 'MET' : 'NOT MET'}</b> · stage medians ${Object.entries(fs.stage_median_ms).map(([k, v]) => `${k} ${fmt(v, 2)}`).join(', ')}</p>`;
    h += `<p class="sub" style="font-size:12px">Denominator: all ${F.rows.length} submitted default-condition queries; native-failure queries (no output to certify) contribute ≈0 added latency and appear at the left end of the paired curve (clamped to 1e-3 ms on the log axis).</p>`;
    h += ecdf([{name: 'native only', v: F.rows.flatMap(r => r.native_ms)}, {name: 'native + certificate', v: F.rows.flatMap(r => r.total_ms)}, {name: 'added (paired)', v: F.rows.flatMap(r => r.added_ms).map(x => Math.max(x, 1e-3))}]);
  }
  el.innerHTML = h + '</div>';
}

/* ---------------- native linkage ---------------- */
function native(el) {
  const NR = D.data.native_reported, FZ = D.data.falsify, TC = D.data.time_consistency;
  let h = `<div class="card"><h2>Native FP32 linkage</h2><p class="sub">Every reported native value is a deterministic FP32 function G(s) of the final progress s∈[0,1+4u]. A forward error analysis of the pinned kernel (approximate division ≤2 ulp, sqrtf ≤3 ulp, FTZ, optional FMA, measured FP32 timing table, knot-side offsets) bounds |G(s) − model(s)| per span and joint; the exact model is then certified against limits minus that bound ${L.native}. Time-law consistency s(t) is only a ${L.sampled} diagnostic → complete equivalence ${L.blocked}.</p>`;
  if (NR) {
    h += `<table><tr><th>path</th><th>status</th><th>max E_v / v_lim</th><th>max E_a / a_lim</th><th>max E_j / j_lim</th>${FZ ? '<th>production bitwise repro mismatches</th><th>trace rows (bitwise)</th><th>max |native−model| / bound</th>' : ''}${TC ? '<th>max |s_n−s*| (ulp)</th><th>jerk Δ at same t</th>' : ''}</tr>`;
    for (const r of NR.rows) {
      const fz = FZ && FZ.rows.find(x => x.path === r.path), tc = TC && TC.rows.find(x => x.path === r.path);
      const maxRatio = fz ? Math.max(...Object.values(fz.same_progress_sweep).flatMap(v => ['velocity', 'acceleration', 'jerk'].map(f => v[f].max_ratio_to_bound))) : null;
      h += `<tr><td>${esc(r.path)}</td><td>${r.status === 'CERTIFIED' ? L.native : esc(r.status)}</td><td class="num">${fmt(r.max_relative_bound.velocity, 7)}</td><td class="num">${fmt(r.max_relative_bound.acceleration, 6)}</td><td class="num">${fmt(r.max_relative_bound.jerk, 5)}</td>`;
      if (FZ) h += fz ? `<td class="num">${Object.values(fz.production_bitwise_reproduction).reduce((a, v) => a + v.bit_mismatches, 0)}</td><td class="num">${fz.captured_trace_jerk ? `${fz.captured_trace_jerk.rows - fz.captured_trace_jerk.bit_mismatches}/${fz.captured_trace_jerk.rows}` : '—'}</td><td class="num">${fmt(maxRatio, 3)}</td>` : '<td>—</td><td>—</td><td>—</td>';
      if (TC) h += tc ? `<td class="num">${fmt(tc.progress_error.max_abs_in_float32_ulps_of_s, 0)}</td><td class="num">${fmt(tc.families.jerk.max_total_abs, 4)} (same-s ${fmt(tc.families.jerk.max_same_progress_abs, 4)})</td>` : '<td>—</td><td>—</td>';
      h += '</tr>';
    }
    h += '</table>';
  } else h += L.notrun;
  const FU = D.data.fp_unit_audit, SA = D.data.sass_audit;
  if (FU) h += `<h3>Arithmetic assumptions checked on ${esc(FU.gpu)}</h3><p>Exhaustive over all ${FU.inputs.toLocaleString()} positive normal float32 inputs with cuRobo's NVRTC options: ${Object.entries(FU.results).map(([k, v]) => `${esc(k)} ≤ ${fmt(v.max_error_ulps, 3)} ulp`).join(', ')}. SASS of <code>sample_with_progress_kernel</code>: ${SA ? Object.entries(SA.functions).filter(([f]) => f.includes('sample_with_progress')).map(([, r]) => `MUFU ${esc(JSON.stringify(r.mufu))}, ${r.ffma_total} FFMA, ${r.ftz_modified} .FTZ-modified, FCHK ${r.fp_instructions.FCHK || 0}`).join('') : '—'}.</p>`;
  if (NR) h += `<h3>Assumptions</h3><ul>${NR.arithmetic_model.assumptions.map(a => `<li>${esc(a)}</li>`).join('')}</ul><p>Not claimed: ${NR.not_claimed.map(esc).join('; ')}.</p>`;
  el.innerHTML = h + '</div>';
}

/* ---------------- correctness ---------------- */
function correctness(el) {
  let h = `<div class="card"><h2>Compiled certifier soundness campaigns ${L.exact}</h2><p class="sub">Seeded adversarial cases (near-active limits, hidden violations, broken rest, zero-speed endpoints, interior zero speed, negative x dips, overflow, NaN/Inf, tiny budgets). Breach = false certificate, false witness, or verifier disagreement. Finite tests; not a universal proof.</p><table><tr><th>build</th><th>cases</th><th>breaches</th><th>reference-certified</th><th>compiled+verified certified</th><th>top outcomes (producer / final / exact reference)</th></tr>`;
  for (const [n, c] of Object.entries(Object.assign({}, D.data.campaigns || {}, Object.fromEntries(Object.entries(D.data.campaigns_mt4 || {}).map(([k, v]) => [k + ' (parallel build, 4 threads)', v]))))) {
    if (!c) { h += `<tr><td>${n}</td><td colspan="5">${L.notrun}</td></tr>`; continue; }
    h += `<tr><td>${n} ${esc(c.extra_flags || '')}</td><td class="num">${c.cases}</td><td class="num">${c.breaches}</td><td class="num">${c.reference_certified}</td><td class="num">${c.compiled_also_certified}</td><td>${c.confusion.slice(0, 6).map(x => `${esc(x.producer)}/${esc(x.final)}/${esc(x.reference)}: ${x.count}`).join('<br>')}</td></tr>`;
  }
  h += '</table></div>';
  const BR = D.data.browser;
  h += '<div class="card"><h2>Proof Lab browser acceptance</h2>';
  if (BR) h += `<table><tr><th>engine</th><th>version</th><th>checks passed</th><th>console errors</th><th>external requests</th></tr>${BR.engines.map(e => `<tr><td>${esc(e.engine)}</td><td>${esc(e.version)}</td><td class="num">${e.checks.filter(c => c.pass).length}/${e.checks.length}</td><td class="num">${e.console_errors.length}</td><td class="num">${e.external_requests.length}</td></tr>`).join('')}</table><p class="mono" style="font-size:11px">HTML sha256 ${esc(BR.html_sha256)}</p>`;
  else h += L.notrun;
  el.innerHTML = h + '</div>';
}

/* ---------------- provenance ---------------- */
function provenance(el) {
  const m = D.meta;
  let h = `<div class="card"><h2>Provenance</h2><p>Branch <span class="mono">${esc(m.branch)}</span>, built from commit <span class="mono">${esc(m.built_from_commit)}</span>; upstream <span class="mono">${esc(m.upstream)}</span>.</p><table><tr><th>source file</th><th>sha256</th></tr>`;
  for (const [k, v] of Object.entries(m.sources)) h += `<tr><td class="mono">${esc(k)}</td><td class="mono">${v === 'NOT_AVAILABLE' ? L.notrun : esc(v)}</td></tr>`;
  el.innerHTML = h + '</table></div>';
}

const RENDER = {overview, margins, timelaw, explorer, population, adaptive, latency, native, correctness, provenance};
let first = 'overview';
try { const t = localStorage.getItem('report-tab'); if (t && RENDER[t]) first = t; } catch (e) { /* ignore */ }
show(first);
})();
