"""Environment setup for Qwen-Image 2.1 LoRA Factory (idempotent).

Verified combination (RTX 5090, Windows, Python 3.12):
  torch 2.13.0+cu130 / torchvision 0.28.0+cu130 / transformers 5.5.3 (ai-toolkit pins)
"""
import hashlib
import os
import shutil
import subprocess
import sys

BACKEND = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BACKEND)
AITK_DIR = os.path.join(BACKEND, "ai-toolkit")
AITK_URL = "https://github.com/ostris/ai-toolkit.git"
AITK_COMMIT = "ecee894ed2b1f3716d9d7326693061ec1a3105bb"
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
MARKER = os.path.join(ROOT, "venv", ".setup_ok")

USE_UV = shutil.which("uv") is not None


def run(cmd, cwd=None):
    print(">", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=cwd, shell=isinstance(cmd, str)) == 0


def pip(py, *args):
    if USE_UV:
        return run(["uv", "pip", "install", "--python", py, *args])
    return run([py, "-m", "pip", "install", *args])


def file_hash(*paths):
    h = hashlib.sha256()
    for p in paths:
        if os.path.exists(p):
            with open(p, "rb") as f:
                h.update(f.read())
    return h.hexdigest()


def ensure_aitk():
    if os.path.exists(os.path.join(AITK_DIR, "run.py")):
        print("[INFO] ai-toolkit found.")
        return True
    if shutil.which("git") is None:
        print("[ERROR] Git is not installed. Install it from https://git-scm.com/")
        return False
    if not run(["git", "clone", AITK_URL, AITK_DIR]):
        return False
    return run(["git", "checkout", AITK_COMMIT], cwd=AITK_DIR)


def main_env():
    py = sys.executable
    constraints = os.path.join(ROOT, "constraints.txt")
    reqs = [os.path.join(AITK_DIR, "requirements_base.txt"), os.path.join(BACKEND, "requirements.txt")]
    digest = file_hash(constraints, *reqs)
    if os.path.exists(MARKER) and open(MARKER).read().strip() == digest:
        print("[INFO] Main environment is up to date.")
        return True
    print("[SETUP] Installing PyTorch 2.13.0 (CUDA 13.0) ...")
    if not pip(py, "torch==2.13.0", "torchvision==0.28.0", "--index-url", TORCH_INDEX):
        return False
    print("[SETUP] Installing ai-toolkit requirements ...")
    if not pip(py, "-c", constraints, "-r", reqs[0], "-r", reqs[1]):
        return False
    # ai-toolkit imports torchaudio; no cu130 build exists for torch 2.13, 2.11 works for import.
    if not pip(py, "--no-deps", "torchaudio==2.11.0", "--index-url", TORCH_INDEX):
        return False
    with open(MARKER, "w") as f:
        f.write(digest)
    return True


def captioner_env():
    """Gemma needs transformers>=5.5 in its own venv. Failure is not fatal."""
    venv = os.path.join(ROOT, "venv_captioner")
    cap_py = os.path.join(venv, "Scripts", "python.exe")
    if os.path.exists(os.path.join(venv, ".setup_ok")):
        print("[INFO] Captioner environment is ready.")
        return True
    if not os.path.exists(cap_py):
        ok = run(["uv", "venv", venv, "--python", "3.12"]) if USE_UV else run([sys.executable, "-m", "venv", venv])
        if not ok:
            return False
    if not pip(cap_py, "torch", "torchvision", "--index-url", TORCH_INDEX):
        return False
    if not pip(cap_py, "-r", os.path.join(BACKEND, "requirements_captioner.txt")):
        return False
    open(os.path.join(venv, ".setup_ok"), "w").write("ok")
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("  Qwen-Image 2.1 LoRA Factory - Setup Check")
    print("=" * 50)
    if not ensure_aitk() or not main_env():
        print("[ERROR] Setup failed. See messages above.")
        sys.exit(1)
    if not captioner_env():
        print("[WARN] Captioner environment could not be prepared. Auto-captioning will be unavailable.")
    print("[OK] Setup complete.")
