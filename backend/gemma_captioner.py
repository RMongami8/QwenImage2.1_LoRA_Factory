"""Local Gemma captioner. Runs in its own venv (transformers>=5.5)."""
import argparse
import glob
import os
import sys

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

DEFAULT_PROMPT = (
    "Describe this image in one detailed paragraph for image-generation model training. "
    "Describe subject, pose, clothing, background, lighting, and art style in plain natural language. "
    "Do not mention that it is an image or add any meta commentary. Do not use markdown."
)


def collect_images(folder):
    seen, paths = set(), []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        for p in glob.glob(os.path.join(folder, ext)):
            key = os.path.normcase(os.path.normpath(p))
            if key not in seen:
                seen.add(key)
                paths.append(p)
    return sorted(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_data_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--trigger_word", default="")
    ap.add_argument("--overwrite", action="store_true",
                    help="Regenerate captions even if a .txt already exists")
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.model_path, "config.json")):
        print(f"Error: '{args.model_path}' is not a local model folder (config.json not found).")
        print('Download first: hf download google/gemma-4-E4B-it --local-dir "<folder>"')
        sys.exit(2)

    images = collect_images(args.train_data_dir)
    total = len(images)
    if total == 0:
        print("No images found in the dataset folder.")
        sys.exit(2)

    todo = [p for p in images
            if args.overwrite or not os.path.exists(os.path.splitext(p)[0] + ".txt")]
    skipped = total - len(todo)
    print(f"Found {total} images ({skipped} already captioned, {len(todo)} to process).")
    if not todo:
        print(f"[CAPTION_PROGRESS] {total}/{total}")
        print("[CAPTION_RESULT] ok=0 failed=0 skipped=%d" % skipped)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Loading Gemma from {args.model_path} on {device} ...")
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, device_map="auto" if device == "cuda" else None,
        dtype=dtype, local_files_only=True)
    if device == "cpu":
        model.to(device)
    model.eval()

    ok = failed = 0
    for i, path in enumerate(todo):
        try:
            image = Image.open(path).convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt}]}]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device, dtype=dtype if device == "cuda" else None)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=180, do_sample=False)
            caption = processor.decode(
                out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
            if not caption:
                raise RuntimeError("empty caption")
            if args.trigger_word:
                caption = f"{args.trigger_word}, {caption}"
            with open(os.path.splitext(path)[0] + ".txt", "w", encoding="utf-8") as f:
                f.write(caption)
            ok += 1
        except Exception as e:  # noqa: BLE001 - report per image and continue
            failed += 1
            print(f"Error processing {path}: {e}")
        print(f"[CAPTION_PROGRESS] {skipped + i + 1}/{total}")

    print(f"[CAPTION_RESULT] ok={ok} failed={failed} skipped={skipped}")
    if device == "cuda":
        del model
        torch.cuda.empty_cache()
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
