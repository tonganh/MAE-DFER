import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_parser():
    path = REPO_ROOT / "scripts" / "infer_all_readme_weights.py"
    spec = importlib.util.spec_from_file_location("_infer_batch", path)
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load infer_all_readme_weights.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.parse_infer_stdout


def main() -> None:
    p = argparse.ArgumentParser(description="Convert legacy all_weights_log.json to prediction-focused JSON.")
    p.add_argument("input_json", type=Path, help="Path to log JSON with stdout per entry.")
    p.add_argument("--out", type=Path, required=True, help="Output JSON path.")
    p.add_argument("--keep-raw", action="store_true", help="Include stdout/stderr in output entries.")
    args = p.parse_args()
    parse_infer_stdout = load_parser()
    data = json.loads(args.input_json.read_text(encoding="utf-8"))
    out_rows = []
    for e in data:
        stdout = e.get("stdout") or ""
        pred = parse_infer_stdout(stdout)
        row = {
            "readme_weight": e.get("readme_weight"),
            "checkpoint": e.get("checkpoint"),
            "dataset": e.get("dataset"),
            "skipped": e.get("skipped", False),
            "skip_reason": e.get("skip_reason"),
            "returncode": int(e["returncode"]) if e.get("returncode") not in (None, "") else None,
            "prediction": pred,
            "parse_ok": pred is not None and not e.get("skipped"),
        }
        if e.get("skipped"):
            row["parse_ok"] = False
        rc = row["returncode"]
        if rc not in (0, None) or (not e.get("skipped") and pred is None):
            err = (e.get("stderr") or "").strip()
            row["infer_error"] = err[:8000] if err else None
        elif not e.get("skipped") and pred is None:
            row["infer_error"] = "failed to parse stdout"
        if args.keep_raw:
            row["stdout"] = stdout
            row["stderr"] = e.get("stderr") or ""
        if row.get("readme_weight") is None and row.get("checkpoint"):
            ck = row["checkpoint"]
            marker = "/readme_checkpoints/"
            if marker in ck:
                row["readme_weight"] = ck.split(marker, 1)[1]
        out_rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_rows, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
