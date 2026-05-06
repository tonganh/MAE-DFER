import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def checkpoint_file_looks_valid(path: Path) -> tuple[bool, str]:
    try:
        head = path.read_bytes()[:2048]
    except OSError as e:
        return False, str(e)
    if not head:
        return False, "empty file"
    lead = head.lstrip()
    if lead.startswith(b"<!") or lead[:5].lower() == b"<html":
        return (
            False,
            "checkpoint is HTML (Drive virus-scan page saved as file); "
            "run: python scripts/download_readme_checkpoints.py --force",
        )
    if lead.startswith(b"<"):
        return False, "file begins with markup, not a PyTorch checkpoint"
    if head.startswith(b"PK\x03\x04"):
        return True, ""
    if head[:1] == b"\x80":
        return True, ""
    return False, "unrecognized header (corrupt or not a .pth)"


_PROB_LINE = re.compile(r"^\s*(\d+)\s+(.+?):\s*([\d.]+(?:e[+-]?\d+)?)\s*$", re.I)


def parse_infer_stdout(stdout: str) -> dict | None:
    idx_m = re.search(r"^Predicted class index:\s*(\d+)\s*$", stdout, re.M)
    lab_m = re.search(r"^Predicted label:\s*(.+?)\s*$", stdout, re.M)
    if not idx_m or not lab_m:
        return None
    probs: dict[str, float] = {}
    in_block = False
    for line in stdout.splitlines():
        if line.strip() == "Class probabilities:":
            in_block = True
            continue
        if not in_block:
            continue
        m = _PROB_LINE.match(line)
        if not m:
            if line.strip() and not line.startswith(" "):
                break
            continue
        _, name, val = m.groups()
        probs[name.strip()] = float(val)
    return {
        "class_index": int(idx_m.group(1)),
        "label": lab_m.group(1).strip(),
        "probabilities": probs,
    }


INFER = REPO_ROOT / "infer_video.py"
ROOT_DEFAULT = REPO_ROOT / "saved" / "model" / "readme_checkpoints"

INFER_TARGETS = [
    {"rel": "finetuning/dfew/fold01/checkpoint.pth", "dataset": "dfew"},
    {"rel": "finetuning/dfew/fold02/checkpoint.pth", "dataset": "dfew"},
    {"rel": "finetuning/dfew/fold03/checkpoint.pth", "dataset": "dfew"},
    {"rel": "finetuning/dfew/fold04/checkpoint.pth", "dataset": "dfew"},
    {"rel": "finetuning/dfew/fold05/checkpoint.pth", "dataset": "dfew"},
    {"rel": "finetuning/ferv39k/checkpoint.pth", "dataset": "ferv39k"},
    {"rel": "finetuning/mafw/fold01/checkpoint.pth", "dataset": "mafw"},
    {"rel": "finetuning/mafw/fold02/checkpoint.pth", "dataset": "mafw"},
    {"rel": "finetuning/mafw/fold03/checkpoint.pth", "dataset": "mafw"},
    {"rel": "finetuning/mafw/fold04/checkpoint.pth", "dataset": "mafw"},
    {"rel": "finetuning/mafw/fold05/checkpoint.pth", "dataset": "mafw"},
]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run infer_video.py once per published fine-tuned weight (README tables)."
    )
    p.add_argument("--video", type=Path, required=True, help="Input video path.")
    p.add_argument("--root", type=Path, default=ROOT_DEFAULT, help="Checkpoint root (see download script).")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write JSON array: per-run prediction fields + optional raw I/O (--include-raw-io).",
    )
    p.add_argument(
        "--include-raw-io",
        action="store_true",
        help="With --json-out, also store stdout and stderr strings (noisy).",
    )
    args = p.parse_args()
    video = args.video.resolve()
    if not video.is_file():
        raise SystemExit(f"video not found: {video}")
    if not INFER.is_file():
        raise SystemExit(f"missing {INFER}")
    root: Path = args.root
    logs: list[dict[str, str]] = []
    for spec in INFER_TARGETS:
        ckpt = (root / spec["rel"]).resolve()
        if not ckpt.is_file():
            print(f"MISSING {ckpt} — run: python scripts/download_readme_checkpoints.py", flush=True)
            continue
        ok, reason = checkpoint_file_looks_valid(ckpt)
        if not ok:
            msg = f"INVALID CHECKPOINT {ckpt}: {reason}"
            print(msg, flush=True)
            row: dict = {
                "readme_weight": spec["rel"],
                "checkpoint": str(ckpt),
                "dataset": spec["dataset"],
                "skipped": True,
                "skip_reason": reason,
                "returncode": None,
                "prediction": None,
            }
            if args.include_raw_io:
                row["stdout"] = ""
                row["stderr"] = msg + "\n"
            logs.append(row)
            continue
        cmd = [
            sys.executable,
            str(INFER),
            "--video",
            str(video),
            "--checkpoint",
            str(ckpt),
            "--dataset",
            spec["dataset"],
            "--device",
            args.device,
        ]
        print("RUN", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
        out = r.stdout or ""
        pred = parse_infer_stdout(out) if r.returncode == 0 else None
        entry: dict = {
            "readme_weight": spec["rel"],
            "checkpoint": str(ckpt),
            "dataset": spec["dataset"],
            "skipped": False,
            "skip_reason": None,
            "returncode": r.returncode,
            "prediction": pred,
            "parse_ok": pred is not None,
        }
        if r.returncode != 0 or pred is None:
            err = (r.stderr or "").strip()
            entry["infer_error"] = err[:8000] if err else None
        if args.include_raw_io:
            entry["stdout"] = out
            entry["stderr"] = r.stderr or ""
        logs.append(entry)
        print(r.stdout, end="" if r.stdout.endswith("\n") else "\n", flush=True)
        if r.stderr:
            print(r.stderr, file=sys.stderr, flush=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(logs, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
