const $ = (id) => document.getElementById(id);
const KEY = "qwen21_factory_";
const PERSIST = ["dataset-path", "gemma-path", "trigger-word", "name", "output-dir", "models-dir", "comfy-dir",
    "vram", "steps", "save-every", "rank", "alpha", "lr", "optimizer", "batch", "repeats", "dropout", "keep",
    "opt-args", "sample-every", "sample-w", "sample-h", "sample-seed", "sample-strengths", "sample-prompts", "cap-overwrite", "loss-win"];
const PERSIST_CHECK = ["sample-on", "resume", "shutdown", "loss-lr"];

/* Settings live in settings.json on the server (survives port changes / browser data wipes);
   localStorage is only a fallback and the source for a one-time migration. */
function collect() {
    const values = {}, checks = {};
    PERSIST.forEach((id) => (values[id] = $(id).value));
    PERSIST_CHECK.forEach((id) => (checks[id] = $(id).checked));
    return { values, checks, res: resolutions(), tab: currentTab };
}
function apply(st) {
    if (!st || !st.values) return false;
    PERSIST.forEach((id) => { if (st.values[id] !== undefined) $(id).value = st.values[id]; });
    PERSIST_CHECK.forEach((id) => { if (st.checks && st.checks[id] !== undefined) $(id).checked = st.checks[id]; });
    if (st.res) document.querySelectorAll(".res").forEach((c) => (c.checked = st.res.includes(+c.value)));
    return true;
}
let saveTimer = null;
function save() {
    const st = collect();
    try { localStorage.setItem(KEY + "state", JSON.stringify(st)); } catch (e) { /* storage may be blocked */ }
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => post("/api/settings", st).catch(() => {}), 500);
}
async function load() {
    let st = null;
    try { st = await api("/api/settings"); } catch (e) { /* server unreachable: fall back */ }
    if (!st || !st.values) {
        try { st = JSON.parse(localStorage.getItem(KEY + "state") || "null"); } catch (e) { st = null; }
    }
    if (!st) { // migrate the pre-server-settings localStorage layout
        st = { values: {}, checks: {}, res: null };
        PERSIST.forEach((id) => { const v = localStorage.getItem(KEY + id); if (v !== null) st.values[id] = v; });
        PERSIST_CHECK.forEach((id) => { const v = localStorage.getItem(KEY + id); if (v !== null) st.checks[id] = v === "1"; });
        const r = localStorage.getItem(KEY + "res");
        if (r) st.res = r.split(",").map(Number);
    }
    apply(st);
    return st;
}
function bindAutosave() {
    [...PERSIST, ...PERSIST_CHECK].forEach((id) => { $(id).addEventListener("input", save); $(id).addEventListener("change", save); });
    document.querySelectorAll(".res").forEach((c) => c.addEventListener("change", save));
}
let currentTab = "dataset";
const resolutions = () => [...document.querySelectorAll(".res")].filter((c) => c.checked).map((c) => +c.value);

/* ---------- tabs ---------- */
function showTab(t) {
    currentTab = t;
    if (typeof saveTimer !== "undefined") save();
    document.querySelectorAll(".tab-content").forEach((s) => (s.style.display = s.id === "tab-" + t ? "" : "none"));
    document.querySelectorAll(".nav-item").forEach((n) => n.classList.toggle("active", n.dataset.tab === t));
    if (t === "train") { loadOutputs(); fetchLoss(true); }
}
document.querySelectorAll(".nav-item").forEach((n) => (n.onclick = () => showTab(n.dataset.tab)));

async function api(url, opts) {
    const r = await fetch(url, opts);
    return r.json();
}
const post = (url, body) => api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

async function browseInto(id, kind) {
    const r = await api(kind === "folder" ? "/api/browse-folder" : "/api/browse-file");
    if (r.path) { $(id).value = r.path; save(); }
}

/* ---------- env ---------- */
async function refreshEnv() {
    const e = await api("/api/env?models_dir=" + encodeURIComponent($("models-dir").value));
    const b = (ok) => `<span class="badge ${ok ? "ok" : "ng"}">${ok ? "OK" : "NG"}</span>`;
    const g = e.gpu;
    $("env-box").innerHTML = `GPU: ${g.name}<br>VRAM free: ${g.free_mb} / ${g.total_mb} MB<br>` +
        `ai-toolkit ${b(e.aitk)}<br>DiT ${b(!!e.weights.dit)} TE ${b(!!e.weights.text_encoder)} VAE ${b(!!e.weights.vae)}<br>` +
        `Config ${b(e.hf_config)} Captioner ${b(e.captioner_env)}` +
        (e.ready ? "" : `<br><span style="color:#f87171">download_models.bat を実行 / run download_models.bat</span>`);
}

/* ---------- dataset / captions ---------- */
let saveTimers = {};
async function loadDataset(goNext) {
    save();
    const path = $("dataset-path").value.trim();
    if (!path) return;
    let r;
    try { r = await api("/api/dataset/images?path=" + encodeURIComponent(path)); } catch (e) { return; }
    if (!r.files) { $("dataset-summary").textContent = "フォルダが見つかりません / not found"; return; }
    const withCap = r.files.filter((f) => f.caption).length;
    $("dataset-summary").textContent = `${r.files.length} images, ${withCap} captioned`;
    const grid = $("caption-grid");
    grid.innerHTML = "";
    r.files.forEach((f) => {
        const d = document.createElement("div");
        d.className = "card image-caption-card";
        d.style.marginBottom = "0";
        const img = document.createElement("img");
        img.className = "caption-preview-img";
        img.loading = "lazy";
        img.src = "/api/image?path=" + encodeURIComponent(f.path);
        const ta = document.createElement("textarea");
        ta.className = "caption-textarea";
        ta.value = f.caption;
        ta.placeholder = f.name;
        const push = () => post("/api/dataset/update-caption", { path: f.path, caption: ta.value });
        ta.oninput = () => { clearTimeout(saveTimers[f.path]); saveTimers[f.path] = setTimeout(push, 800); };
        ta.onblur = () => { clearTimeout(saveTimers[f.path]); push(); };
        d.append(img, ta);
        grid.append(d);
    });
    window.imageCount = r.files.length;
    updateEpochHint();
    if (goNext && r.files.length) showTab("caption");
}

async function runCaptioner() {
    save();
    const r = await post("/api/run-captioner", {
        path: $("dataset-path").value.trim(), model_path: $("gemma-path").value.trim(),
        trigger_word: $("trigger-word").value, overwrite: $("cap-overwrite").value === "1",
    });
    if (r.status !== "started") alert(r.message);
}

/* ---------- training ---------- */
function updateEpochHint() {
    const n = window.imageCount || 0, steps = +$("steps").value, b = +$("batch").value || 1, rep = +$("repeats").value || 1;
    $("epoch-hint").textContent = n ? `参考: 約 ${(steps * b / (n * rep)).toFixed(1)} epoch 相当 (${n}枚 × repeat ${rep}, batch ${b}。バケット端数で多少ずれます)` : "";
}
["steps", "batch", "repeats"].forEach((id) => ($(id).oninput = updateEpochHint));
$("optimizer").onchange = () => {
    if ($("optimizer").value === "prodigy") $("lr").value = "1.0";
    else if ($("lr").value === "1.0") $("lr").value = "1e-4";
};

async function startTraining() {
    save();
    $("start-msg").textContent = "";
    const body = {
        path: $("dataset-path").value.trim(), output_dir: $("output-dir").value.trim(), name: $("name").value.trim(),
        models_dir: $("models-dir").value.trim(), vram: $("vram").value, steps: +$("steps").value,
        save_every: +$("save-every").value, keep_checkpoints: +$("keep").value, batch_size: +$("batch").value,
        num_repeats: +$("repeats").value, resolutions: resolutions(), lr: $("lr").value.trim(), rank: +$("rank").value,
        alpha: +$("alpha").value, optimizer_type: $("optimizer").value, optimizer_args: $("opt-args").value,
        caption_dropout: +$("dropout").value, trigger_word: $("trigger-word").value,
        sample_enabled: $("sample-on").checked, sample_every: +$("sample-every").value,
        sample_size: 768, sample_width: +$("sample-w").value, sample_height: +$("sample-h").value,
        sample_seed: +$("sample-seed").value, sample_strengths: $("sample-strengths").value, sample_prompts: $("sample-prompts").value,
        resume: $("resume").checked, comfy_loras_dir: $("comfy-dir").value.trim(), shutdown: $("shutdown").checked,
    };
    const r = await post("/api/start-training", body);
    if (r.status !== "started") $("start-msg").textContent = r.message;
}
async function stopJob() {
    if (!confirm("停止しますか? / Stop the running job?")) return;
    await post("/api/stop", {});
}

async function loadOutputs() {
    const o = $("output-dir").value.trim(), n = $("name").value.trim();
    if (!o || !n) return;
    const q = `output_dir=${encodeURIComponent(o)}&name=${encodeURIComponent(n)}`;
    const r = await api(`/api/outputs?${q}`);
    $("outputs").innerHTML = r.checkpoints.length
        ? r.checkpoints.map((c) => `${c.name} (${c.mb} MB)`).join("<br>") + `<br><code>${o}\${n}</code>` : "-";
    renderSamples(r.samples);
    try { drawNorms((await api(`/api/lora-norms?${q}`)).points); } catch (e) { drawNorms([]); }
}

function renderSamples(samples) {
    const host = $("samples");
    if (!samples.length) { host.innerHTML = '<p class="small">-</p>'; return; }
    let strengths = [];
    try { strengths = $("sample-strengths").value.split(/[,;]/).map((t) => t.trim()).filter(Boolean); } catch (e) { /* ignore */ }
    const per = Math.max(1, strengths.length); // images per prompt: columns are strengths
    const byStep = {};
    samples.forEach((s) => (byStep[s.step] = byStep[s.step] || []).push(s));
    host.innerHTML = Object.keys(byStep).map(Number).sort((a, b) => b - a).map((step) => {
        const imgs = byStep[step].map((s) => {
            const label = strengths.length ? `m=${strengths[s.idx % per] ?? "?"}` : `#${s.idx}`;
            return `<figure style="margin:0"><img src="/api/image?path=${encodeURIComponent(s.path)}" loading="lazy" title="${s.name}">` +
                `<figcaption class="small" style="text-align:center">${label}</figcaption></figure>`;
        }).join("");
        return `<div style="margin-bottom:1rem"><div class="small" style="margin-bottom:.3rem"><b>step ${step}</b></div>` +
            `<div class="thumbs" style="grid-template-columns:repeat(auto-fill,minmax(150px,1fr))">${imgs}</div></div>`;
    }).join("");
}

function drawNorms(pts) {
    const cv = $("norm-canvas"), dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
    if (!W) return;
    cv.width = W * dpr; cv.height = H * dpr;
    const g = cv.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, W, H);
    g.font = "11px sans-serif"; g.fillStyle = "#94a3b8";
    if (!pts.length) { g.fillText("データなし / no checkpoints", 60, 30); return; }
    const mL = 56, mR = 16, mT = 14, mB = 26, pw = W - mL - mR, ph = H - mT - mB;
    const x0 = Math.min(0, pts[0].step), x1 = Math.max(...pts.map((p) => p.step), 1);
    const ymax = Math.max(...pts.map((p) => p.total)) * 1.15 || 1;
    const X = (v) => mL + ((v - x0) / (x1 - x0)) * pw, Y = (v) => mT + ph - (v / ymax) * ph;
    g.strokeStyle = "rgba(255,255,255,0.08)"; g.textAlign = "right";
    for (let i = 0; i <= 4; i++) { const v = (ymax * i) / 4, y = Y(v); g.beginPath(); g.moveTo(mL, y); g.lineTo(mL + pw, y); g.stroke(); g.fillText(v.toFixed(1), mL - 6, y + 4); }
    g.strokeStyle = "#2dd4bf"; g.lineWidth = 2; g.beginPath();
    const all = [{ step: 0, total: 0 }, ...pts];
    all.forEach((p, i) => (i ? g.lineTo(X(p.step), Y(p.total)) : g.moveTo(X(p.step), Y(p.total)))); g.stroke();
    g.textAlign = "center";
    pts.forEach((p) => {
        g.fillStyle = "#2dd4bf"; g.beginPath(); g.arc(X(p.step), Y(p.total), 3.5, 0, 6.3); g.fill();
        g.fillStyle = "#94a3b8"; g.fillText(p.final ? `${p.step} (final)` : p.step, X(p.step), H - 8);
        g.fillText(p.total.toFixed(1), X(p.step), Y(p.total) - 8);
    });
    if (pts.length >= 2) {
        const a = pts[pts.length - 2], b = pts[pts.length - 1];
        const growth = ((b.total - a.total) / Math.max(a.total, 1e-9)) * 100;
        $("norm-info").textContent = `最新の伸び: ${a.step}→${b.step} step で ${growth >= 0 ? "+" : ""}${growth.toFixed(0)}% ` +
            `(直線的に増え続けているなら、まだベースから離れ続けています / 頭打ちなら収束)`;
    }
}

/* ---------- loss curve ---------- */
let L = { steps: [], loss: [], lr: [] };
const lossCv = $("loss-canvas");

function movavg(a, w) {
    if (w <= 1) return a.slice();
    const out = new Array(a.length); let sum = 0;
    for (let i = 0; i < a.length; i++) { sum += a[i]; if (i >= w) sum -= a[i - w]; out[i] = sum / Math.min(i + 1, w); }
    return out;
}

function drawLoss(hover) {
    const dpr = window.devicePixelRatio || 1;
    const W = lossCv.clientWidth, H = lossCv.clientHeight;
    if (!W) return;
    lossCv.width = W * dpr; lossCv.height = H * dpr;
    const g = lossCv.getContext("2d"); g.scale(dpr, dpr);
    g.clearRect(0, 0, W, H);
    const css = getComputedStyle(document.documentElement);
    const mL = 56, mR = $("loss-lr").checked ? 56 : 12, mT = 10, mB = 24, pw = W - mL - mR, ph = H - mT - mB;
    g.font = "11px sans-serif"; g.fillStyle = "#94a3b8"; g.strokeStyle = "rgba(255,255,255,0.08)";
    const n = L.steps.length;
    if (!n) { g.fillText("データなし / no data (loss_log.db)", mL + 10, mT + 20); return; }
    const x0 = L.steps[0], x1 = Math.max(L.steps[n - 1], x0 + 1);
    const sm = movavg(L.loss, +$("loss-win").value);
    const sorted = L.loss.slice().sort((a, b) => a - b);
    const ymax = sorted[Math.floor(sorted.length * 0.99)] * 1.1 || 1, ymin = 0;
    const X = (s) => mL + ((s - x0) / (x1 - x0)) * pw, Y = (v) => mT + ph - ((Math.min(v, ymax) - ymin) / (ymax - ymin)) * ph;
    for (let i = 0; i <= 4; i++) {
        const v = ymin + ((ymax - ymin) * i) / 4, y = Y(v);
        g.beginPath(); g.moveTo(mL, y); g.lineTo(mL + pw, y); g.stroke();
        g.textAlign = "right"; g.fillText(v.toExponential(1), mL - 6, y + 4);
    }
    g.textAlign = "center";
    for (let i = 0; i <= 5; i++) { const s = Math.round(x0 + ((x1 - x0) * i) / 5); g.fillText(s, X(s), H - 6); }
    // raw loss
    g.strokeStyle = "rgba(251,146,60,0.28)"; g.lineWidth = 1; g.beginPath();
    L.loss.forEach((v, i) => (i ? g.lineTo(X(L.steps[i]), Y(v)) : g.moveTo(X(L.steps[i]), Y(v)))); g.stroke();
    // smoothed loss
    g.strokeStyle = "#fb923c"; g.lineWidth = 2; g.beginPath();
    sm.forEach((v, i) => (i ? g.lineTo(X(L.steps[i]), Y(v)) : g.moveTo(X(L.steps[i]), Y(v)))); g.stroke();
    // lr (right axis)
    if ($("loss-lr").checked) {
        const lrs = L.lr.map((v) => (v == null ? 0 : v)), lmax = Math.max(...lrs, 1e-12) * 1.1;
        g.strokeStyle = "#60a5fa"; g.lineWidth = 1.5; g.beginPath();
        lrs.forEach((v, i) => { const y = mT + ph - (v / lmax) * ph; i ? g.lineTo(X(L.steps[i]), y) : g.moveTo(X(L.steps[i]), y); });
        g.stroke(); g.fillStyle = "#60a5fa"; g.textAlign = "left";
        for (let i = 0; i <= 4; i++) g.fillText(((lmax * i) / 4).toExponential(1), mL + pw + 6, mT + ph - (ph * i) / 4 + 4);
    }
    if (hover != null) {
        const i = Math.max(0, Math.min(n - 1, Math.round(((hover - mL) / pw) * (n - 1))));
        g.strokeStyle = "rgba(255,255,255,0.4)"; g.beginPath(); g.moveTo(X(L.steps[i]), mT); g.lineTo(X(L.steps[i]), mT + ph); g.stroke();
        $("loss-tip").textContent = `step ${L.steps[i]}  loss ${L.loss[i].toExponential(3)}  avg ${sm[i].toExponential(3)}` +
            (L.lr[i] != null ? `  lr ${L.lr[i].toExponential(2)}` : "");
    } else $("loss-tip").textContent = "";
    const last = L.loss.slice(-Math.min(50, n)), first = L.loss.slice(0, Math.min(50, n));
    const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
    $("loss-info").textContent = `${n} steps | 最初の50平均 ${mean(first).toExponential(2)} → 直近50平均 ${mean(last).toExponential(2)} | min ${sorted[0].toExponential(2)}`;
}
lossCv.addEventListener("mousemove", (e) => drawLoss(e.clientX - lossCv.getBoundingClientRect().left));
lossCv.addEventListener("mouseleave", () => drawLoss(null));
$("loss-win").onchange = $("loss-lr").onchange = () => drawLoss(null);
window.addEventListener("resize", () => drawLoss(null));

async function fetchLoss(full) {
    const o = $("output-dir").value.trim(), n = $("name").value.trim();
    if (!o || !n) return;
    const since = full || !L.steps.length ? -1 : L.steps[L.steps.length - 1];
    let r;
    try { r = await api(`/api/loss?output_dir=${encodeURIComponent(o)}&name=${encodeURIComponent(n)}&since=${since}`); } catch (e) { return; }
    if (!r.exists) { L = { steps: [], loss: [], lr: [] }; drawLoss(null); return; }
    if (since >= 0 && r.last_step >= 0 && r.last_step < since) return fetchLoss(true); // run was restarted / pruned
    if (since < 0) L = { steps: r.steps, loss: r.loss, lr: r.lr };
    else if (r.steps.length) { L.steps.push(...r.steps); L.loss.push(...r.loss); L.lr.push(...r.lr); }
    drawLoss(null);
}
const reloadLoss = () => fetchLoss(true);
setInterval(() => { if (lastStatus === "running" && $("tab-train").style.display !== "none") fetchLoss(false); }, 3000);

/* ---------- websocket ---------- */
const fmt = (s) => { s = Math.max(0, Math.round(s)); const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
    return (h ? h + "h " : "") + m + "m " + (s % 60) + "s"; };
let lastStatus = "idle", elapsedBase = 0, elapsedAt = 0;

function addLog(line) {
    const el = $("log");
    const stick = el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
    el.append(document.createTextNode(line + "\n"));
    while (el.childNodes.length > 2500) el.removeChild(el.firstChild);
    if (stick) el.scrollTop = el.scrollHeight;
}

function applyStatus(s) {
    const running = s.status === "running";
    $("start-btn").disabled = running;
    $("stop-btn").disabled = !running;
    $("cap-btn").disabled = running;
    const badge = $("state-badge");
    badge.textContent = s.kind ? `${s.kind}: ${s.status}` : s.status;
    badge.className = "badge " + (s.status === "finished" || running ? "ok" : "ng");
    $("st-phase").textContent = s.phase || "-";
    $("st-step").textContent = `${s.step} / ${s.total}`;
    $("st-loss").textContent = s.loss != null ? s.loss.toExponential(2) : "-";
    $("st-speed").textContent = s.speed || "-";
    const pct = s.total ? (s.step / s.total) * 100 : 0;
    if (s.kind === "caption") {
        $("cap-progress").style.display = "";
        $("cap-ptext").textContent = `Progress: ${s.step} / ${s.total}`;
        $("cap-pct").textContent = pct.toFixed(0) + "%";
        $("cap-bar").style.width = pct + "%";
    } else {
        $("train-bar").style.width = pct + "%";
    }
    elapsedBase = s.elapsed; elapsedAt = Date.now();
    if (s.status === "running" && s.step > 0 && s.phase === "training") {
        $("st-eta").textContent = fmt(((s.total - s.step) / s.step) * Math.max(1, s.elapsed));
    } else if (s.status !== "running") $("st-eta").textContent = "-";
    if (lastStatus !== "running" && s.status === "running" && s.kind === "train") { L = { steps: [], loss: [], lr: [] }; drawLoss(null); }
    if (lastStatus === "running" && s.status !== "running") {
        loadOutputs(); fetchLoss(true);
        if (s.kind === "caption" && s.status === "finished") loadDataset(false);
        if (s.message && s.status === "failed") $("start-msg").textContent = s.message;
        refreshEnv();
    }
    lastStatus = s.status;
}
setInterval(() => {
    if (lastStatus === "running") $("st-elapsed").textContent = fmt(elapsedBase + (Date.now() - elapsedAt) / 1000);
}, 1000);

function connect() {
    const ws = new WebSocket(`ws://${location.host}/ws`);
    ws.onmessage = (ev) => {
        const m = JSON.parse(ev.data);
        if (m.t === "history") { $("log").textContent = ""; m.lines.forEach(addLog); }
        else if (m.t === "log") addLog(m.line);
        else if (m.t === "status") applyStatus(m);
    };
    ws.onclose = () => setTimeout(connect, 2000);
    const ping = setInterval(() => { if (ws.readyState === 1) ws.send("ping"); else clearInterval(ping); }, 15000);
}

(async () => {
    const st = await load();
    bindAutosave();
    updateEpochHint();
    refreshEnv();
    setInterval(refreshEnv, 30000);
    connect();
    if ($("dataset-path").value) loadDataset(false);
    if (st && st.tab && st.tab !== "dataset") showTab(st.tab);
})();
