"""Center-crop every image in a folder to one fixed size (originals are untouched).

Usage: python crop_dataset.py <src> <dst> [width height]
Width/height should be multiples of 32 (Qwen-Image 2.1 bucket grid).
"""
import os
import shutil
import sys

from PIL import Image

EXT = {".png", ".jpg", ".jpeg", ".webp"}


def crop_to(im, w, h):
    sw, sh = im.size
    scale = max(w / sw, h / sh)  # cover the target, never leave padding
    nw, nh = max(w, round(sw * scale)), max(h, round(sh * scale))
    im = im.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - w) // 2, (nh - h) // 2
    return im.crop((left, top, left + w, top + h)), scale


def main():
    src, dst = sys.argv[1], sys.argv[2]
    w, h = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) >= 5 else (736, 1312)
    os.makedirs(dst, exist_ok=True)
    for f in sorted(os.listdir(src)):
        base, ext = os.path.splitext(f)
        if ext.lower() not in EXT:
            continue
        im = Image.open(os.path.join(src, f)).convert("RGB")
        out, scale = crop_to(im, w, h)
        out.save(os.path.join(dst, base + ".png"))
        txt = os.path.join(src, base + ".txt")
        if os.path.exists(txt):
            shutil.copy2(txt, os.path.join(dst, base + ".txt"))
        print(f"{f}: {im.size} -> {w}x{h}  scale x{scale:.2f}")


if __name__ == "__main__":
    main()
