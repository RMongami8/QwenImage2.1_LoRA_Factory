"""Parse ai-toolkit / tqdm output (carriage-return updates, ANSI codes)."""
import re

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
STEP_RE = re.compile(r"(\d+)/(\d+) \[")
LOSS_RE = re.compile(r"loss:\s*([0-9.eE+-]+)")
LR_RE = re.compile(r"lr:\s*([0-9.eE+-]+)")
SPEED_RE = re.compile(r"([0-9.]+)(it/s|s/it)")
TQDM_RE = re.compile(r"\d+%\|")


def clean(line: str) -> str:
    return ANSI_RE.sub("", line).strip()


def is_tqdm(line: str) -> bool:
    return bool(TQDM_RE.search(line)) and bool(STEP_RE.search(line))


def parse_tqdm(line: str, job_name: str):
    """Return dict(step, total, loss, lr, speed, is_training) or None."""
    m = STEP_RE.search(line)
    if not m:
        return None
    info = {
        "step": int(m.group(1)),
        "total": int(m.group(2)),
        "is_training": line.startswith(job_name + ":"),
        "desc": "" if re.match(r"\s*\d+%\|", line) else line.split(":", 1)[0][:40],
    }
    lm = LOSS_RE.search(line)
    if lm:
        try:
            info["loss"] = float(lm.group(1))
        except ValueError:
            pass
    rm = LR_RE.search(line)
    if rm:
        try:
            info["lr"] = float(rm.group(1))
        except ValueError:
            pass
    sm = SPEED_RE.search(line)
    if sm:
        info["speed"] = sm.group(1) + sm.group(2)
    return info


def iter_lines(read):
    """Yield text segments split on \\r and \\n from a binary stream reader."""
    buf = b""
    while True:
        chunk = read(4096)
        if not chunk:
            break
        buf += chunk
        while True:
            idx = min([i for i in (buf.find(b"\n"), buf.find(b"\r")) if i >= 0] or [-1])
            if idx < 0:
                break
            seg, buf = buf[:idx], buf[idx + 1:]
            if seg:
                yield seg.decode("utf-8", errors="replace")
    if buf:
        yield buf.decode("utf-8", errors="replace")
