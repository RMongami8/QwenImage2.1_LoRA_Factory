"""Download missing Qwen-Image 2.1 model files into model\\ (ComfyUI folder layout)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "model")
HF_HOME = os.path.join(MODELS, "hf_cache")
os.environ["HF_HOME"] = HF_HOME
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402
from huggingface_hub.errors import EntryNotFoundError  # noqa: E402

REPO = "Comfy-Org/Qwen-Image-2.1"
BASE = "Qwen/Qwen-Image-2.1"
# (repo path, local subfolder name accepted anywhere under that folder, approx GB)
FILES = [
    ("diffusion_models/qwen_image_2.1_int8_convrot.safetensors", "qwen_image_2.1_int8_convrot.safetensors", 7.26),
    ("text_encoders/qwen3vl_8b_int8_convrot.safetensors", "qwen3vl_8b_int8_convrot.safetensors", 9.35),
    ("vae/qwen_image_2.1_vae_bf16.safetensors", "qwen_image_2.1_vae_bf16.safetensors", 0.68),
]


def exists_anywhere(sub, fname):
    for dp, _d, files in os.walk(os.path.join(MODELS, sub)):
        if fname in files:
            return True
    return False


def main():
    missing = [f for f in FILES if not exists_anywhere(f[0].split("/")[0], f[1])]
    total = sum(f[2] for f in missing)
    print(f"Models folder: {MODELS}")
    if missing:
        print("Files to download:")
        for repo_path, _n, gb in missing:
            print(f"  {REPO}/{repo_path}  (~{gb} GB)")
        print(f"Total ~{total:.1f} GB, plus small config files from {BASE}.")
        if "--yes" not in sys.argv and input("Proceed? [y/N] ").strip().lower() != "y":
            print("Cancelled.")
            return 1
        for repo_path, _n, _gb in missing:
            print("Downloading", repo_path, "...")
            hf_hub_download(REPO, repo_path, local_dir=MODELS)
    else:
        print("All weight files are already present.")

    print("Fetching config files ...")
    snapshot_download(BASE, allow_patterns=["*.json", "*.jinja", "*.txt", "processor/*"],
                      ignore_patterns=["*.safetensors.index.json", "assets/*"])
    # processor/config.json does not exist upstream; asking once records the miss in the
    # cache so later runs can resolve it without a network round trip.
    try:
        hf_hub_download(BASE, "processor/config.json")
    except EntryNotFoundError:
        pass
    except Exception as e:  # noqa: BLE001
        print("Note:", e)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
