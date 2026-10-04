import asyncio
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.parse
from collections import deque
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import aitk_config
import progress

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.dirname(BACKEND_DIR)
AITK_DIR = os.path.join(BACKEND_DIR, "ai-toolkit")
DEFAULT_MODELS_DIR = os.path.join(APP_ROOT, "model")
HF_HOME = os.path.join(DEFAULT_MODELS_DIR, "hf_cache")
LOG_DIR = os.path.join(APP_ROOT, "logs")
CAPTIONER_PYTHON = os.path.join(APP_ROOT, "venv_captioner", "Scripts", "python.exe")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MIN_FREE_VRAM_MB = 6000

# Weight files ai-toolkit accepts for qwen_image_2 (convrot8 preferred).
WEIGHT_FILES = {
    "dit": ["qwen_image_2.1_int8_convrot.safetensors", "qwen_image_2.1_bf16.safetensors"],
    "text_encoder": ["qwen3vl_8b_int8_convrot.safetensors", "qwen3vl_8b_bf16.safetensors"],
    "vae": ["qwen_image_2.1_vae_bf16.safetensors"],
}

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(_app):
    global loop
    loop = asyncio.get_running_loop()
    yield


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------- job state
class Job:
    def __init__(self):
        self.kind: Optional[str] = None  # "train" | "caption"
        self.status = "idle"  # idle | running | finished | failed | stopped
        self.name = ""
        self.process: Optional[subprocess.Popen] = None
        self.stop_requested = False
        self.step = 0
        self.total = 0
        self.phase = ""
        self.loss: Optional[float] = None
        self.lr: Optional[float] = None
        self.speed = ""
        self.started = 0.0
        self.message = ""
        self.outputs: list = []

    def snapshot(self):
        return {
            "kind": self.kind, "status": self.status, "name": self.name,
            "step": self.step, "total": self.total, "phase": self.phase,
            "loss": self.loss, "lr": self.lr, "speed": self.speed,
            "elapsed": int(time.time() - self.started) if self.started and self.status == "running" else 0,
            "message": self.message, "outputs": self.outputs,
        }


job = Job()
log_buffer: deque = deque(maxlen=3000)
clients: list[WebSocket] = []
loop: Optional[asyncio.AbstractEventLoop] = None
log_file = None


def is_running() -> bool:
    return job.status == "running"


async def send_all(msg: dict):
    text = json.dumps(msg, ensure_ascii=False)
    for ws in list(clients):
        try:
            await ws.send_text(text)
        except Exception:
            if ws in clients:
                clients.remove(ws)


def emit(msg: dict):
    """Thread-safe broadcast."""
    if loop is not None:
        asyncio.run_coroutine_threadsafe(send_all(msg), loop)


def log(line: str):
    line = line.rstrip("\r\n")
    log_buffer.append(line)
    try:
        print(line, flush=True)
    except Exception:  # console encoding / closed stdout must never kill a reader thread
        pass
    if log_file:
        try:
            log_file.write(line + "\n")
            log_file.flush()
        except Exception:
            pass
    emit({"t": "log", "line": line})


def push_status():
    emit({"t": "status", **job.snapshot()})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.append(ws)
    try:
        await ws.send_text(json.dumps({"t": "history", "lines": list(log_buffer)}, ensure_ascii=False))
        await ws.send_text(json.dumps({"t": "status", **job.snapshot()}))
        while True:
            await ws.receive_text()  # keeps the connection open; client sends pings
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        if ws in clients:
            clients.remove(ws)


# ---------------------------------------------------------------- helpers
def gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=10).strip().split("\n")[0]
        name, total, used = [p.strip() for p in out.split(",")]
        return {"name": name, "total_mb": int(total), "used_mb": int(used), "free_mb": int(total) - int(used)}
    except Exception:
        return {"name": "NVIDIA GPU not detected", "total_mb": 0, "used_mb": 0, "free_mb": 0}


def find_weight(models_dir: str, category: str, names: list) -> Optional[str]:
    root = os.path.join(models_dir, {"dit": "diffusion_models", "text_encoder": "text_encoders", "vae": "vae"}[category])
    for n in names:
        for dp, _dn, files in os.walk(root):
            if n in files:
                return os.path.join(dp, n)
    return None


def hf_config_ready() -> bool:
    snaps = os.path.join(HF_HOME, "hub", "models--Qwen--Qwen-Image-2.1", "snapshots")
    if not os.path.isdir(snaps):
        return False
    for s in os.listdir(snaps):
        if os.path.exists(os.path.join(snaps, s, "processor", "tokenizer.json")) and \
           os.path.exists(os.path.join(snaps, s, "text_encoder", "config.json")):
            return True
    return False


def env_status(models_dir: str):
    weights = {k: find_weight(models_dir, k, v) for k, v in WEIGHT_FILES.items()}
    return {
        "aitk": os.path.exists(os.path.join(AITK_DIR, "run.py")),
        "weights": weights,
        "hf_config": hf_config_ready(),
        "captioner_env": os.path.exists(CAPTIONER_PYTHON),
        "models_dir": models_dir,
        "ready": all(weights.values()) and hf_config_ready() and os.path.exists(os.path.join(AITK_DIR, "run.py")),
    }


def kill_tree(proc: subprocess.Popen, force: bool):
    args = ["taskkill", "/PID", str(proc.pid), "/T"] + (["/F"] if force else [])
    subprocess.run(args, capture_output=True)


def pick_dialog(code: str) -> str:
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
        return out.stdout.strip()
    except Exception:
        return ""


# ---------------------------------------------------------------- API: env / browse
@app.get("/api/env")
async def api_env(models_dir: str = ""):
    return {"gpu": gpu_info(), **env_status(models_dir or DEFAULT_MODELS_DIR)}


@app.get("/api/browse-folder")
async def browse_folder():
    code = ("import tkinter as tk\nfrom tkinter import filedialog\nr=tk.Tk();r.withdraw();"
            "r.attributes('-topmost',True)\nprint(filedialog.askdirectory(title='Select Folder'))")
    path = await asyncio.get_running_loop().run_in_executor(None, pick_dialog, code)
    return {"path": path.replace("/", os.sep) if path else ""}


@app.get("/api/browse-file")
async def browse_file():
    code = ("import tkinter as tk\nfrom tkinter import filedialog\nr=tk.Tk();r.withdraw();"
            "r.attributes('-topmost',True)\nprint(filedialog.askopenfilename(title='Select File'))")
    path = await asyncio.get_running_loop().run_in_executor(None, pick_dialog, code)
    return {"path": path.replace("/", os.sep) if path else ""}


# ---------------------------------------------------------------- API: dataset
@app.get("/api/dataset/images")
async def list_images(path: str):
    if not os.path.isdir(path):
        raise HTTPException(status_code=404, detail="Folder not found")
    files = []
    for f in sorted(os.listdir(path)):
        base, ext = os.path.splitext(f)
        if ext.lower() in IMAGE_EXTENSIONS:
            txt = os.path.join(path, base + ".txt")
            caption = ""
            if os.path.exists(txt):
                with open(txt, "r", encoding="utf-8") as tf:
                    caption = tf.read().strip()
            files.append({"name": f, "path": os.path.join(path, f), "caption": caption})
    return {"files": files}


class CaptionUpdate(BaseModel):
    path: str
    caption: str


@app.post("/api/dataset/update-caption")
async def update_caption(u: CaptionUpdate):
    base, ext = os.path.splitext(u.path)
    if ext.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Not an image path")
    with open(base + ".txt", "w", encoding="utf-8") as f:
        f.write(u.caption)
    return {"status": "success"}


@app.get("/api/image")
async def get_image(path: str):
    if os.path.splitext(path)[1].lower() not in IMAGE_EXTENSIONS or not os.path.isfile(path):
        raise HTTPException(status_code=404)
    return FileResponse(path)


# ---------------------------------------------------------------- process runner
def spawn(cmd, env=None, cwd=None):
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=cwd,
                            creationflags=flags)


def reader_thread(proc: subprocess.Popen, kind: str, on_done):
    try:
        read = proc.stdout.read1 if hasattr(proc.stdout, "read1") else proc.stdout.read
        for seg in progress.iter_lines(read):
            line = progress.clean(seg)
            if not line:
                continue
            try:
                handle_line(kind, line)
            except Exception as e:  # keep draining the pipe no matter what
                try:
                    log(f"[app] line handling error: {e}")
                except Exception:
                    pass
    finally:
        proc.wait()
        on_done(proc.returncode)


_last_push = 0.0


def handle_line(kind: str, line: str):
    global _last_push
    if kind == "train" and progress.is_tqdm(line):
        p = progress.parse_tqdm(line, job.name)
        if not p:
            return
        job.phase = "training" if p["is_training"] else (p["desc"] or "caching")
        job.step, job.total = p["step"], p["total"]
        job.loss = p.get("loss", job.loss)
        job.lr = p.get("lr", job.lr)
        job.speed = p.get("speed", job.speed)
        now = time.time()
        if now - _last_push > 0.4 or p["step"] == p["total"]:
            _last_push = now
            push_status()
        if p["step"] == p["total"] or p["step"] % 50 == 0:
            log(line)
        return
    if kind == "caption":
        m = re.search(r"\[CAPTION_PROGRESS\]\s*(\d+)/(\d+)", line)
        if m:
            job.step, job.total, job.phase = int(m.group(1)), int(m.group(2)), "captioning"
            push_status()
            return
    log(line)
    if kind == "train" and job.phase in ("", "starting") and "Loading" in line:
        job.phase = "loading"
        push_status()


def finish(kind_label: str, rc: int, success_check=None):
    if job.stop_requested:
        job.status, job.message = "stopped", "Stopped by user"
    elif rc == 0 and (success_check is None or success_check()):
        job.status, job.message = "finished", "Completed"
    else:
        job.status = "failed"
        job.message = job.message or f"{kind_label} failed (exit code {rc})"
    job.phase = job.status
    log(f"--- {kind_label} {job.status} (exit code {rc}) ---")
    push_status()


# ---------------------------------------------------------------- API: training
class TrainingConfig(BaseModel):
    path: str
    output_dir: str
    name: str
    models_dir: str = ""
    vram: str = "low"
    steps: int = 2000
    save_every: int = 250
    keep_checkpoints: int = 4
    batch_size: int = 1
    num_repeats: int = 1
    resolutions: list[int] = [1024]
    lr: str = "1e-4"
    rank: int = 16
    alpha: int = 16
    optimizer_type: str = "adamw8bit"
    optimizer_args: str = ""
    caption_dropout: float = 0.05
    trigger_word: str = ""
    sample_enabled: bool = False
    sample_every: int = 250
    sample_size: int = 768
    sample_prompts: str = ""
    resume: bool = False
    comfy_loras_dir: str = ""
    shutdown: bool = False


def validate_training(c: TrainingConfig):
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", c.name or ""):
        return "Name may contain only letters, digits, '_', '-' and '.'."
    if not os.path.isdir(c.path):
        return "Dataset folder not found."
    imgs = [f for f in os.listdir(c.path) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
    if not imgs:
        return "No images found in the dataset folder."
    if not c.output_dir:
        return "Output folder is required."
    if c.vram not in aitk_config.VRAM_PRESETS:
        return "Invalid VRAM mode."
    if c.optimizer_type not in aitk_config.OPTIMIZERS:
        return "Invalid optimizer."
    if c.steps < 1 or c.save_every < 1 or not c.resolutions:
        return "Steps, save interval and resolutions must be set."
    try:
        float(c.lr)
    except ValueError:
        return "Learning rate is not a number."
    env = env_status(c.models_dir or DEFAULT_MODELS_DIR)
    if not env["aitk"]:
        return "ai-toolkit is missing. Run start.bat to set up the environment."
    missing = [k for k, v in env["weights"].items() if not v]
    if missing:
        return "Missing model files: " + ", ".join(missing) + ". Run download_models.bat."
    if not env["hf_config"]:
        return "Model config files are missing. Run download_models.bat."
    out_dir = os.path.join(c.output_dir, c.name)
    if os.path.isdir(out_dir) and any(f.endswith(".safetensors") for f in os.listdir(out_dir)) and not c.resume:
        return (f"'{out_dir}' already has checkpoints and ai-toolkit would resume from them. "
                "Change the name or tick 'Resume'.")
    return None


@app.post("/api/start-training")
async def start_training(c: TrainingConfig):
    global log_file
    if is_running():
        return {"status": "error", "message": "Another job is already running."}
    err = validate_training(c)
    if err:
        return {"status": "error", "message": err}
    g = gpu_info()
    if g["total_mb"] and g["free_mb"] < MIN_FREE_VRAM_MB:
        return {"status": "error",
                "message": f"Only {g['free_mb']} MB of VRAM is free ({g['name']}). "
                           "Close other GPU programs (ComfyUI, browsers, ...) and retry."}

    models_dir = c.models_dir or DEFAULT_MODELS_DIR
    os.makedirs(c.output_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    cfg_path = os.path.join(c.output_dir, f"{c.name}_job.yaml")
    aitk_config.write_config(c.model_dump(), cfg_path)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = open(os.path.join(LOG_DIR, f"train_{c.name}_{stamp}.log"), "w", encoding="utf-8")
    log_buffer.clear()
    job.__init__()
    job.kind, job.name, job.status, job.phase = "train", c.name, "running", "starting"
    job.total, job.started = c.steps, time.time()
    log(f"Config written to {cfg_path}")
    log(f"GPU: {g['name']}  free VRAM: {g['free_mb']} MB  preset: {c.vram}")

    env = os.environ.copy()
    env.update({"MODELS_PATH": models_dir, "HF_HOME": HF_HOME, "PYTHONUTF8": "1",
                "PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"})
    cmd = [sys.executable, "-u", os.path.join(AITK_DIR, "run.py"), cfg_path]
    job.process = spawn(cmd, env=env, cwd=AITK_DIR)
    t0 = job.started
    final_path = os.path.join(c.output_dir, c.name, f"{c.name}.safetensors")

    def success():
        return os.path.exists(final_path) and os.path.getmtime(final_path) >= t0 - 1

    def done(rc):
        global log_file
        finish("Training", rc, success)
        if job.status == "finished" and c.comfy_loras_dir:
            try:
                os.makedirs(c.comfy_loras_dir, exist_ok=True)
                dst = shutil.copy2(final_path, os.path.join(c.comfy_loras_dir, f"{c.name}.safetensors"))
                job.outputs.append(dst)
                log(f"Copied to ComfyUI loras: {dst}")
            except Exception as e:
                log(f"ComfyUI copy failed: {e}")
        if job.status == "finished":
            job.outputs.insert(0, final_path)
        push_status()
        if log_file:
            log_file.close()
            log_file = None
        if job.status == "finished" and c.shutdown:
            log("Shutting down in 60 seconds. Run 'shutdown /a' to cancel.")
            os.system("shutdown /s /t 60")

    threading.Thread(target=reader_thread, args=(job.process, "train", done), daemon=True).start()
    push_status()
    return {"status": "started"}


@app.post("/api/stop")
async def stop_job():
    if not is_running() or job.process is None:
        return {"status": "error", "message": "No job is running."}
    job.stop_requested = True
    log("Stop requested ...")
    proc = job.process
    kill_tree(proc, force=False)

    def escalate():
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            log("Process did not exit, forcing termination.")
            kill_tree(proc, force=True)

    threading.Thread(target=escalate, daemon=True).start()
    return {"status": "stopping"}


@app.get("/api/status")
async def api_status():
    return job.snapshot()


@app.get("/api/outputs")
async def list_outputs(output_dir: str, name: str):
    base = os.path.join(output_dir, name)
    ckpts, samples = [], []
    if os.path.isdir(base):
        for f in sorted(os.listdir(base)):
            if f.endswith(".safetensors"):
                p = os.path.join(base, f)
                ckpts.append({"name": f, "path": p, "mb": round(os.path.getsize(p) / 1048576, 1)})
        sdir = os.path.join(base, "samples")
        if os.path.isdir(sdir):
            for f in sorted(os.listdir(sdir), reverse=True):
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
                    samples.append({"name": f, "path": os.path.join(sdir, f)})
    return {"checkpoints": ckpts, "samples": samples[:24]}


# ---------------------------------------------------------------- API: saved settings
SETTINGS_FILE = os.path.join(APP_ROOT, "settings.json")


@app.get("/api/settings")
async def get_settings():
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


@app.post("/api/settings")
async def save_settings(data: dict):
    if len(json.dumps(data)) > 200_000:
        raise HTTPException(status_code=413, detail="Settings too large")
    tmp = SETTINGS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, SETTINGS_FILE)  # atomic: never leaves a half-written file
    return {"status": "saved"}


# ---------------------------------------------------------------- API: loss curve
@app.get("/api/loss")
async def api_loss(output_dir: str, name: str, since: int = -1):
    """Per-step loss / lr from ai-toolkit's loss_log.db (steps > `since`)."""
    db = os.path.join(output_dir, name, "loss_log.db")
    empty = {"exists": False, "steps": [], "loss": [], "lr": [], "last_step": -1}
    if not os.path.isfile(db):
        return empty
    uri = "file:" + urllib.parse.quote(db.replace("\\", "/"), safe="/:") + "?mode=ro"
    try:
        con = sqlite3.connect(uri, uri=True, timeout=5)
    except sqlite3.Error:
        return empty
    try:
        keys = [r[0] for r in con.execute("SELECT key FROM metric_keys WHERE key LIKE 'loss/%'")]
        if not keys:
            return {**empty, "exists": True}
        key = "loss/loss" if "loss/loss" in keys else sorted(keys)[0]
        rows = con.execute(
            "SELECT m.step, m.value_real, l.value_real FROM metrics m "
            "LEFT JOIN metrics l ON l.step = m.step AND l.key = 'learning_rate' "
            "WHERE m.key = ? AND m.step > ? ORDER BY m.step", (key, since)).fetchall()
        last = con.execute("SELECT MAX(step) FROM steps").fetchone()[0]
    except sqlite3.Error:
        return {**empty, "exists": True}
    finally:
        con.close()
    return {
        "exists": True, "key": key,
        "steps": [r[0] for r in rows], "loss": [r[1] for r in rows], "lr": [r[2] for r in rows],
        "last_step": last if last is not None else -1,
    }


# ---------------------------------------------------------------- API: captioner
class CaptionerConfig(BaseModel):
    path: str
    model_path: str
    trigger_word: str = ""
    overwrite: bool = False


@app.post("/api/run-captioner")
async def run_captioner(c: CaptionerConfig):
    global log_file
    if is_running():
        return {"status": "error", "message": "Another job is already running."}
    if not os.path.isdir(c.path):
        return {"status": "error", "message": "Dataset folder not found."}
    if not c.model_path or not os.path.exists(os.path.join(c.model_path, "config.json")):
        return {"status": "error", "message": "Gemma model folder is invalid (config.json not found)."}
    if not os.path.exists(CAPTIONER_PYTHON):
        return {"status": "error", "message": "Captioner environment (venv_captioner) is missing. Run start.bat."}

    log_buffer.clear()
    log_file = None
    job.__init__()
    job.kind, job.name, job.status, job.phase, job.started = "caption", "caption", "running", "loading", time.time()
    cmd = [CAPTIONER_PYTHON, "-u", os.path.join(BACKEND_DIR, "gemma_captioner.py"),
           f"--train_data_dir={c.path}", f"--model_path={c.model_path}"]
    if c.trigger_word.strip():
        cmd.append(f"--trigger_word={c.trigger_word.strip()}")
    if c.overwrite:
        cmd.append("--overwrite")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    job.process = spawn(cmd, env=env)
    threading.Thread(target=reader_thread, args=(job.process, "caption", lambda rc: finish("Captioning", rc)),
                     daemon=True).start()
    push_status()
    return {"status": "started"}


app.mount("/", StaticFiles(directory=os.path.join(APP_ROOT, "frontend"), html=True), name="frontend")

def pick_port(start: int) -> int:
    import socket
    for p in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:  # nothing is listening
                return p
        print(f"[INFO] Port {p} is in use (another instance may be running).")
    raise SystemExit("No free port found.")


if __name__ == "__main__":
    import uvicorn
    import webbrowser
    port = pick_port(int(os.environ.get("APP_PORT", "8002")))
    url = f"http://127.0.0.1:{port}"
    print(f"Open {url} in your browser.", flush=True)
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
