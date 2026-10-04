"""Build an ai-toolkit job config (qwen_image_2 arch) from the GUI settings."""
import os

import yaml

# VRAM presets. "low" is the layout verified in the 20-step smoke test; the
# other two are untuned estimates that trade speed for memory.
VRAM_PRESETS = {
    "high": {"transformer_offload": 0.0, "te_offload": 1.0},
    "balanced": {"transformer_offload": 0.5, "te_offload": 1.0},
    "low": {"transformer_offload": 1.0, "te_offload": 1.0},
}

OPTIMIZERS = {
    # name -> (ai-toolkit optimizer name, default optimizer_params)
    "adamw8bit": ("adamw8bit", {}),
    "adamw": ("adamw", {}),
    "prodigy": ("prodigy", {"decouple": True, "weight_decay": 0.01,
                            "use_bias_correction": True, "safeguard_warmup": True}),
}


def fwd(path: str) -> str:
    return os.path.abspath(path).replace("\\", "/")


def parse_params(text: str) -> dict:
    """'a=1 b=True' -> {'a': 1, 'b': True}"""
    out = {}
    for tok in (text or "").split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        low = v.lower()
        if low in ("true", "false"):
            out[k] = low == "true"
            continue
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def parse_strengths(text: str) -> list:
    """'0, 0.5, 1' -> [0.0, 0.5, 1.0] (empty -> [])."""
    out = []
    for tok in (text or "").replace(";", ",").split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    return out


def build_config(c: dict) -> dict:
    """c: validated settings dict from the API (see TrainingConfig in main.py)."""
    preset = VRAM_PRESETS.get(c["vram"], VRAM_PRESETS["low"])
    opt_name, opt_defaults = OPTIMIZERS.get(c["optimizer_type"], OPTIMIZERS["adamw8bit"])
    opt_params = dict(opt_defaults)
    opt_params.update(parse_params(c.get("optimizer_args", "")))

    train = {
        "batch_size": c["batch_size"],
        "cache_text_embeddings": True,
        "steps": c["steps"],
        "gradient_accumulation": 1,
        "train_unet": True,
        "train_text_encoder": False,
        "gradient_checkpointing": True,
        "noise_scheduler": "flowmatch",
        "optimizer": opt_name,
        "lr": float(c["lr"]),
        "dtype": "bf16",
        "disable_sampling": not c["sample_enabled"],
    }
    if opt_params:
        train["optimizer_params"] = opt_params

    model = {
        "name_or_path": "Comfy-Org/Qwen-Image-2.1",
        "arch": "qwen_image_2",
        "quantize": True,
        "qtype": "convrot8",
        "quantize_te": True,
        "qtype_te": "convrot8",
    }
    if preset["transformer_offload"] > 0 or preset["te_offload"] > 0:
        model["layer_offloading"] = True
        model["layer_offloading_transformer_percent"] = preset["transformer_offload"]
        model["layer_offloading_text_encoder_percent"] = preset["te_offload"]

    prompts = [p.strip() for p in c.get("sample_prompts", "").splitlines() if p.strip()]
    if not prompts:
        prompts = ["a photo"]
    trigger = c.get("trigger_word", "").strip()
    if trigger:
        prompts = [p if trigger in p else f"{trigger}, {p}" for p in prompts]
    # Strength comparison: every prompt is rendered once per LoRA strength (--m) with the same seed,
    # so a row of images differs only by LoRA strength. Order: prompt-major, strength-minor.
    seed = int(c.get("sample_seed", 42))
    strengths = parse_strengths(c.get("sample_strengths", ""))
    if strengths:
        prompts = [f"{p} --m {m:g} --seed {seed}" for p in prompts for m in strengths]
    else:
        prompts = [f"{p} --seed {seed}" for p in prompts]

    process = {
        "type": "sd_trainer",
        "training_folder": fwd(c["output_dir"]),
        "device": "cuda:0",
        "network": {"type": "lora", "linear": c["rank"], "linear_alpha": c["alpha"]},
        "save": {
            "dtype": "float16",
            "save_every": c["save_every"],
            "max_step_saves_to_keep": c["keep_checkpoints"],
        },
        "datasets": [{
            "folder_path": fwd(c["path"]),
            "caption_ext": "txt",
            "caption_dropout_rate": c["caption_dropout"],
            "shuffle_tokens": False,
            "cache_latents_to_disk": True,
            "num_repeats": c["num_repeats"],
            "resolution": sorted(set(c["resolutions"])),
        }],
        "train": train,
        "model": model,
        "sample": {
            "sampler": "flowmatch",
            "sample_every": c["sample_every"],
            "width": c.get("sample_width") or c["sample_size"],
            "height": c.get("sample_height") or c["sample_size"],
            "prompts": prompts,
            "neg": "",
            "seed": seed,
            "walk_seed": False,  # same seed at every sampling step, so checkpoints are comparable
            "guidance_scale": 4,
            "sample_steps": 25,
        },
    }
    if trigger:
        process["trigger_word"] = trigger
    # per-step loss/lr go to <output>/<name>/loss_log.db (read by the app's loss curve)
    process["logging"] = {"log_every": 1, "use_ui_logger": True}

    return {
        "job": "extension",
        "config": {"name": c["name"], "process": [process]},
        "meta": {"name": "[name]", "version": "1.0"},
    }


def write_config(c: dict, path: str) -> str:
    cfg = build_config(c)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return path
