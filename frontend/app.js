const $ = (id) => document.getElementById(id);
const KEY = "qwen21_factory_";
const PERSIST = ["dataset-path", "gemma-path", "trigger-word", "name", "output-dir", "models-dir", "comfy-dir",
    "vram", "steps", "save-every", "rank", "alpha", "lr", "optimizer", "batch", "repeats", "dropout", "keep",
    "opt-args", "sample-every", "sample-size", "sample-prompts"];
const PERSIST_CHECK = ["sample-on", "resume", "shutdown"];

function load() {
    PERSIST.forEach((id) => { const v = localStorage.getItem(KEY + id); if (v !== null) $(id).value = v; });
    PERSIST_CHECK.forEach((id) => { const v = localStorage.getItem(KEY + id); if (v !== null) $(id).checked = v === "1"; });
    const r = localStorage.getItem(KEY + "res");
    if (r) document.querySelectorAll(".res").forEach((c) => (c.checked = r.split(",").includes(c.value)));
}
function save() {
    PERSIST.forEach((id) => localStorage.setItem(KEY + id, $(id).value));
    PERSIST_CHECK.forEach((id) => localStorage.setItem(KEY + id, $(id).checked ? "1" : "0"));
    localStorage.setItem(KEY + "res", resolutions().join(","));
}
const resolutions = () => [...document.querySelectorAll(".res")].filter((c) => c.checked).map((c) => +c.value);

/* ---------- tabs ---------- */
function showTab(t) {
    document.querySelectorAll(".tab-content").forEach((s) => (s.style.display = s.id === "tab-" + t ? "" : "none"));
    document.querySelectorAll(".nav-item").forEach((n) => n.classList.toggle("active", n.dataset.tab === t));
    if (t === "train") loadOutputs();
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
        sample_size: +$("sample-size").value, sample_prompts: $("sample-prompts").value,
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
    const r = await api(`/api/outputs?output_dir=${encodeURIComponent(o)}&name=${encodeURIComponent(n)}`);
    $("outputs").innerHTML = r.checkpoints.length
        ? r.checkpoints.map((c) => `${c.name} (${c.mb} MB)`).join("<br>") + `<br><code>${o}\\${n}</code>` : "-";
    $("samples").innerHTML = r.samples.map((s) => `<img src="/api/image?path=${encodeURIComponent(s.path)}" title="${s.name}">`).join("");
}

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
    if (lastStatus === "running" && s.status !== "running") {
        loadOutputs();
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

load();
refreshEnv();
setInterval(refreshEnv, 30000);
connect();
if ($("dataset-path").value) loadDataset(false);
