# Qwen-Image 2.1 LoRA Factory

Local GUI (FastAPI + plain HTML/JS) for training Qwen-Image 2.1 LoRAs with
[ai-toolkit](https://github.com/ostris/ai-toolkit) (`qwen_image_2` arch), pinned to commit
`ecee894` in `backend/setup_check.py`. Modeled on the Krea2 LoRA Factory.

## Requirements
- Windows, NVIDIA GPU (verified on RTX 5090 32 GB), Git
- Python 3.12 or [uv](https://docs.astral.sh/uv/)
- ~20 GB disk for model files

## Quick start
1. `start.bat` – creates `venv`, clones ai-toolkit, installs the verified
   dependency set (`constraints.txt`: torch 2.13.0+cu130, transformers 5.5.3, ...),
   prepares `venv_captioner`, starts the server and opens http://127.0.0.1:8002.
2. `download_models.bat` – downloads only missing files into `model\`
   (ComfyUI layout: `diffusion_models\`, `text_encoders\`, `vae\`) after asking:
   - `qwen_image_2.1_int8_convrot.safetensors` (DiT, ~7.3 GB)
   - `qwen3vl_8b_int8_convrot.safetensors` (Qwen3-VL-8B, ~9.4 GB)
   - `qwen_image_2.1_vae_bf16.safetensors` (~0.7 GB)
   - small config/processor files of `Qwen/Qwen-Image-2.1` into `model\hf_cache`

   Existing files (e.g. in a ComfyUI install) can be reused: put them under `model\`
   (a hard link avoids duplication) or set "Models folder" in the GUI.
3. In the GUI: Dataset -> Captions -> Configuration -> Run.

<img width="776" height="592" alt="image" src="https://github.com/user-attachments/assets/0d461b22-f58c-4401-bb1a-7ff74f97c980" />

## Notes
- Training runs `python backend\ai-toolkit\run.py <name>_job.yaml`; the generated YAML is kept in the
  output folder. Output: `<Output>\<name>\<name>.safetensors` plus step checkpoints.
- ai-toolkit **resumes** from existing checkpoints in `<Output>\<name>`; the app refuses to start on a
  non-empty folder unless "Resume" is ticked.
- The app refuses to start if less than ~6 GB of VRAM is free (close ComfyUI etc. first).
- Only one job (training or captioning) runs at a time. Stop uses `taskkill /T` (forced after 8 s).
- Logs: `logs\train_*.log`.
- Settings are saved automatically to `settings.json` (not committed), so they survive port/browser changes.
- Run tab: **Loss curve** (per-step loss/lr), **LoRA weight norm** per checkpoint (how far the LoRA has moved
  the base model; a steady linear rise means it is still drifting), and **sample grids by step**.
- Samples use a fixed seed and an optional width/height (e.g. 576x1024 for portrait). "Strengths" (e.g. `0, 0.5, 1`)
  renders every prompt once per LoRA strength (`--m`); strength 0 is the base model, so base, LoRA and
  checkpoints can be compared fairly.

## Verification status
Verified: 20-step smoke test (512 px, rank 16, convrot8 + full layer offload, "low" mode),
checkpoint written with ComfyUI-style `diffusion_model.*` keys.
Not verified: "balanced"/"high" presets (untuned estimates), in-training sample generation,
Gemma captioning in this app, loading the LoRA in ComfyUI, 1024 px speed/VRAM.

## Layout
```
backend/  main.py aitk_config.py progress.py setup_check.py download_models.py gemma_captioner.py
frontend/ index.html app.js style.css
model/    diffusion_models text_encoders vae hf_cache   (git-ignored)
```
