import argparse
import glob
import os
from collections import OrderedDict

import numpy as np
import torch
from scipy.special import softmax
from timm.models import create_model

import modeling_finetune
import video_transforms
import volume_transforms
from decord import VideoReader, cpu

DFEW_CLASSES = [
    "Happy",
    "Sad",
    "Neutral",
    "Angry",
    "Surprise",
    "Disgust",
    "Fear",
]


def load_checkpoint(model, path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    if not isinstance(state_dict, dict):
        raise ValueError("Unsupported checkpoint format")
    new_state = OrderedDict()
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[7:]
        if nk.startswith("backbone."):
            nk = nk[9:]
        elif nk.startswith("encoder."):
            nk = nk[8:]
        new_state[nk] = v
    msg = model.load_state_dict(new_state, strict=False)
    if msg.missing_keys:
        print("missing_keys:", msg.missing_keys[:20], "...")
    if msg.unexpected_keys:
        print("unexpected_keys:", msg.unexpected_keys[:20], "...")


def load_video_buffer(path, clip_len, frame_sample_rate):
    if os.path.isdir(path):
        frames = sorted(glob.glob(os.path.join(path, "*.jpg")))
        if not frames:
            frames = sorted(glob.glob(os.path.join(path, "*.png")))
        if not frames:
            raise FileNotFoundError(f"No .jpg/.png frames in {path}")
        n = len(frames)
        all_index = [x for x in range(0, n, frame_sample_rate)]
        while len(all_index) < clip_len:
            all_index.append(all_index[-1])
        from PIL import Image

        buffer = []
        for i in all_index:
            with open(frames[i], "rb") as f:
                img = Image.open(f)
                buffer.append(np.array(img.convert("RGB")))
        return np.stack(buffer, axis=0)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if os.path.getsize(path) < 1024:
        raise ValueError(f"File too small: {path}")
    vr = VideoReader(path, num_threads=1, ctx=cpu(0))
    all_index = [x for x in range(0, len(vr), frame_sample_rate)]
    while len(all_index) < clip_len:
        all_index.append(all_index[-1])
    vr.seek(0)
    return vr.get_batch(all_index).asnumpy()


def build_model(args):
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        all_frames=args.num_frames * args.num_segments,
        tubelet_size=args.tubelet_size,
        drop_rate=0.0,
        drop_path_rate=0.0,
        attn_drop_rate=0.0,
        drop_block_rate=None,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
        depth=args.depth,
        attn_type=args.attn_type,
        lg_region_size=args.lg_region_size,
        lg_first_attn_type=args.lg_first_attn_type,
        lg_third_attn_type=args.lg_third_attn_type,
        lg_attn_param_sharing_first_third=args.lg_attn_param_sharing_first_third,
        lg_attn_param_sharing_all=args.lg_attn_param_sharing_all,
        lg_classify_token_type=args.lg_classify_token_type,
        lg_no_second=args.lg_no_second,
        lg_no_third=args.lg_no_third,
    )
    return model


def preprocess_test_views(buffer, clip_len, short_side_size, test_num_segment, test_num_crop):
    data_resize = video_transforms.Compose(
        [
            video_transforms.Resize(
                size=(short_side_size, short_side_size), interpolation="bilinear"
            )
        ]
    )
    clip_list = [buffer[i] for i in range(buffer.shape[0])]
    resized = data_resize(clip_list)
    buffer = np.stack(resized, 0)
    data_transform = video_transforms.Compose(
        [
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    views = []
    for chunk_nb in range(test_num_segment):
        for split_nb in range(test_num_crop):
            spatial_step = (
                1.0
                * (max(buffer.shape[1], buffer.shape[2]) - short_side_size)
                / (test_num_crop - 1)
                if test_num_crop > 1
                else 0.0
            )
            temporal_step = max(
                1.0 * (buffer.shape[0] - clip_len) / (test_num_segment - 1), 0
            )
            temporal_start = int(chunk_nb * temporal_step)
            spatial_start = int(split_nb * spatial_step)
            if buffer.shape[1] >= buffer.shape[2]:
                crop = buffer[
                    temporal_start : temporal_start + clip_len,
                    spatial_start : spatial_start + short_side_size,
                    :,
                    :,
                ]
            else:
                crop = buffer[
                    temporal_start : temporal_start + clip_len,
                    :,
                    spatial_start : spatial_start + short_side_size,
                    :,
                ]
            t = data_transform(crop)
            views.append(t.unsqueeze(0))
    return torch.cat(views, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Path to .mp4/.avi or folder of frames")
    parser.add_argument("--checkpoint", type=str, required=True, help="DFEW fine-tuned .pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--nb_classes", type=int, default=7)
    parser.add_argument("--model", type=str, default="vit_base_dim512_no_depth_patch16_160")
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--num_segments", type=int, default=1)
    parser.add_argument("--sampling_rate", type=int, default=4)
    parser.add_argument("--input_size", type=int, default=160)
    parser.add_argument("--short_side_size", type=int, default=160)
    parser.add_argument("--tubelet_size", type=int, default=2)
    parser.add_argument("--test_num_segment", type=int, default=2)
    parser.add_argument("--test_num_crop", type=int, default=2)
    parser.add_argument("--attn_type", type=str, default="local_global")
    parser.add_argument("--lg_region_size", type=int, nargs=3, default=[2, 5, 10])
    parser.add_argument("--lg_first_attn_type", type=str, default="self")
    parser.add_argument("--lg_third_attn_type", type=str, default="cross")
    parser.add_argument(
        "--lg_classify_token_type", type=str, default="region", choices=["org", "region", "all"]
    )
    parser.add_argument("--lg_attn_param_sharing_first_third", action="store_true")
    parser.add_argument("--lg_attn_param_sharing_all", action="store_true")
    parser.add_argument("--lg_no_second", action="store_true")
    parser.add_argument("--lg_no_third", action="store_true")
    parser.add_argument("--use_mean_pooling", action="store_true", default=True)
    parser.add_argument("--use_cls", action="store_false", dest="use_mean_pooling")
    parser.add_argument("--init_scale", type=float, default=0.001)
    args = parser.parse_args()
    args.lg_region_size = tuple(args.lg_region_size)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    buffer = load_video_buffer(
        args.video, args.num_frames, args.sampling_rate
    )
    batch = preprocess_test_views(
        buffer,
        args.num_frames,
        args.short_side_size,
        args.test_num_segment,
        args.test_num_crop,
    )
    batch = batch.to(device)

    model = build_model(args)
    load_checkpoint(model, args.checkpoint)
    model.eval()
    model.to(device)

    probs_list = []
    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        with torch.no_grad():
            for i in range(batch.shape[0]):
                logits = model(batch[i : i + 1])
                probs_list.append(softmax(logits.float().cpu().numpy().ravel()))
    avg_prob = np.mean(probs_list, axis=0)
    pred_idx = int(np.argmax(avg_prob))
    name = DFEW_CLASSES[pred_idx]
    print("Predicted class index:", pred_idx)
    print("Predicted label (DFEW):", name)
    print("Class probabilities:")
    for i, c in enumerate(DFEW_CLASSES):
        print(f"  {i} {c}: {avg_prob[i]:.4f}")


if __name__ == "__main__":
    main()
