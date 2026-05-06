import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mae_dfer.api.services.checkpoint_service import download_google_drive_file

ROOT_DEFAULT = REPO_ROOT / "saved" / "model" / "readme_checkpoints"

ENTRIES = [
    {
        "id": "1nzvMITUHic9fKwjQ7XLcnaXYViWTawRv",
        "rel": "pretraining/voxceleb2/videomae_pretrain_voxceleb2.pth",
        "note": "SSL pretrain only; not for infer_video.py (use after fine-tuning).",
    },
    {"id": "1wRxwEZlrc3z3DqQ84xm_olmqRsj2obH3", "rel": "finetuning/dfew/fold01/checkpoint.pth"},
    {"id": "1lY4L2PMVWuF93K6_VqEl7QrPoSK0SvaQ", "rel": "finetuning/dfew/fold02/checkpoint.pth"},
    {"id": "1FPKxBoGO3VXvLhcHY8iOPb9lQPi0_C1z", "rel": "finetuning/dfew/fold03/checkpoint.pth"},
    {"id": "1yFDc1n8SaTEQWrVX8k65loQm45rfwQeO", "rel": "finetuning/dfew/fold04/checkpoint.pth"},
    {"id": "1wmXO4M2kjpAOnvof8CmpJE6wUrxMUOgw", "rel": "finetuning/dfew/fold05/checkpoint.pth"},
    {"id": "1vq9WxuV229spEX7JQCMTluLXvvefLRzq", "rel": "finetuning/ferv39k/checkpoint.pth"},
    {"id": "1BXTp-2mdy0fvrcjwFsZ4fPY53E4DotvR", "rel": "finetuning/mafw/fold01/checkpoint.pth"},
    {"id": "1Lzm50nPzZtTODfNYSkSLm09mhExHu2eL", "rel": "finetuning/mafw/fold02/checkpoint.pth"},
    {"id": "1WlCTw5OjV6SZ7L7rU5xzvbVD_GTcp8CC", "rel": "finetuning/mafw/fold03/checkpoint.pth"},
    {"id": "18fdMGu-GdYxUmCEG-33NR3hxvHFyrO-v", "rel": "finetuning/mafw/fold04/checkpoint.pth"},
    {"id": "1CO6OY-P6oM5LMikLTlB5iz_AXack2vfH", "rel": "finetuning/mafw/fold05/checkpoint.pth"},
]


def main() -> None:
    p = argparse.ArgumentParser(description="Download Google Drive checkpoints listed in README.")
    p.add_argument(
        "--root",
        type=Path,
        default=ROOT_DEFAULT,
        help="Directory under which relative paths are stored.",
    )
    p.add_argument(
        "--only-finetuned",
        action="store_true",
        help="Skip the VoxCeleb2 SSL pretrain file.",
    )
    p.add_argument("--force", action="store_true", help="Re-download even if file exists.")
    args = p.parse_args()
    root: Path = args.root
    root.mkdir(parents=True, exist_ok=True)
    for e in ENTRIES:
        if args.only_finetuned and "pretraining" in e["rel"]:
            continue
        dst = root / e["rel"]
        if dst.is_file() and not args.force:
            print(f"skip exists: {dst}")
            continue
        print(f"download -> {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        download_google_drive_file(e["id"], str(dst))
        note = e.get("note")
        if note:
            print(f"  ({note})")
    print("done")


if __name__ == "__main__":
    main()
