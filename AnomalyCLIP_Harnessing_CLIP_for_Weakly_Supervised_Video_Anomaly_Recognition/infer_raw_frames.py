#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import pandas as pd

from src.models.components.anomaly_clip import AnomalyCLIP


# -----------------------
# CLIP preprocess
# -----------------------
def build_clip_preprocess():
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(clip_mean, clip_std),
    ])


# -----------------------
# Video frame sampler
# -----------------------
def sample_video_frames(video_path, fps_target=4.0, max_frames=None, save_dir=None, video_name="video"):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src_fps / fps_target)))
    frames = []
    idx, saved_idx = 0, 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append(pil_img)

            # 🚩 Save frame to disk
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                frame_filename = f"{video_name}_frame{saved_idx:05d}.jpg"
                pil_img.save(save_dir / frame_filename)
            saved_idx += 1

            if max_frames is not None and saved_idx >= max_frames:
                break
        idx += 1
    cap.release()
    if len(frames) == 0:
        raise RuntimeError("No frames were sampled.")
    return frames


# -----------------------
# Build net args (UCF-Crime YAML)
# -----------------------
def build_net_args(labels_file, arch="ViT-B/16", normal_id=0):
    return dict(
        arch=arch,
        shared_context=False,
        ctx_init="",
        n_ctx=8,
        seg_length=16,
        num_segments=32,
        select_idx_dropout_topk=0.7,
        select_idx_dropout_bottomk=0.7,
        heads=8,
        dim_heads=None,
        concat_features=False,
        emb_size=256,
        depth=1,
        num_topk=3,
        num_bottomk=3,
        labels_file=labels_file,
        normal_id=normal_id,
        dropout_prob=0.0,
        temporal_module="axial",
        direction_module="learned_encoder_finetune",
        selector_module="directions",
        batch_norm=True,
        feature_size=512,
        use_similarity_as_features=False,
        ctx_dim=512,
        load_from_features=False,
        stride=1,
        ncrops=1,
    )


# -----------------------
# Load weights (net.*)
# -----------------------
def load_net_weights_from_ckpt(net, ckpt_path, device="cuda", strict=False):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    net_state = {k.replace("net.", "", 1): v for k, v in state.items() if k.startswith("net.")}
    missing, unexpected = net.load_state_dict(net_state, strict=strict)
    if missing:
        print(f"[warn] Missing keys: {missing[:5]}...")
    if unexpected:
        print(f"[warn] Unexpected keys: {unexpected[:5]}...")


# -----------------------
# ncentroid helper
# -----------------------
def compute_ncentroid_from_frames(tensor_frames, net, device, batch=64):
    feats = []
    with torch.no_grad():
        for i in range(0, tensor_frames.shape[0], batch):
            x = tensor_frames[i:i+batch].to(device)
            f = net.image_encoder(x.float())  # [B,512]
            f = F.normalize(f, dim=-1)
            feats.append(f)
    return torch.cat(feats, dim=0).mean(dim=0)


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser("AnomalyCLIP raw-video inference + frame saving")
    ap.add_argument("--video", required=True, help="Path to video")
    ap.add_argument("--ckpt", required=True, help="Checkpoint path")
    ap.add_argument("--labels_file", required=True, help="labels.csv")
    ap.add_argument("--out", default="outputs_single/video_scores.csv", help="Output CSV path")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--normal_id", type=int, default=0)
    ap.add_argument("--device", choices=["cuda", "cpu"], default=None)
    ap.add_argument("--out_frames_dir", default="outputs_frames", help="Directory to save sampled frames")  # 🚩
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = Path(args.out_frames_dir) / Path(args.video).stem  # 🚩 one subdir per video

    # 1) Sample + save frames
    preprocess = build_clip_preprocess()
    frames_pil = sample_video_frames(
        args.video, fps_target=args.fps, max_frames=args.max_frames,
        save_dir=frames_dir, video_name=Path(args.video).stem
    )
    frames_tensor = torch.stack([preprocess(img) for img in frames_pil])  # [T,3,224,224]

    # 2) Build net
    net_kwargs = build_net_args(args.labels_file, normal_id=args.normal_id)
    net = AnomalyCLIP(**net_kwargs).to(device).eval()
    load_net_weights_from_ckpt(net, args.ckpt, device=device, strict=False)

    # 3) ncentroid
    ncentroid = compute_ncentroid_from_frames(frames_tensor, net, device)

    # 4) Pad to multiple of BASE
    T = frames_tensor.shape[0]
    N, L = net.num_segments, net.seg_length
    BASE = N * L
    s = math.ceil(T / BASE)
    target_len = BASE * s
    if T < target_len:
        pad = frames_tensor[-1:].repeat(target_len - T, 1, 1, 1)
        frames_padded = torch.cat([frames_tensor, pad], dim=0)
    else:
        frames_padded = frames_tensor
    frames_padded = frames_padded.unsqueeze(0)  # [1,T_total,3,224,224]

    labels = torch.zeros(target_len, dtype=torch.long, device=device)

    # 5) Forward
    with torch.no_grad():
        similarity, scores = net(
            image_features=frames_padded.to(device),
            labels=labels,
            ncentroid=ncentroid,
            segment_size=s,
            test_mode=True,
        )
    # scores = scores[:T].detach().cpu().numpy()
    # similarity = similarity[:T]
    
    similarity = similarity[:T]        # [T, C-1]  (all classes except 'Normal')
    scores = scores[:T]                # [T]

    # 7) Build full per-class probabilities (insert 'Normal' = 1 - score at normal_id)
    #    Order follows the labels.csv order, like the repo’s eval step.
    labels_df = pd.read_csv(args.labels_file)
    class_names = labels_df["name"].tolist()
    normal_idx = args.normal_id

    softmax_sim = torch.softmax(similarity, dim=1)            # [T, C-1]
    class_probs_abn = softmax_sim * scores.unsqueeze(1)       # [T, C-1]
    normal_probs = (1.0 - scores).unsqueeze(1)                # [T, 1]

    # split & insert normal at normal_idx
    left = class_probs_abn[:, :normal_idx]
    right = class_probs_abn[:, normal_idx:]
    class_probs_full = torch.cat([left, normal_probs, right], dim=1)  # [T, C]

    # 8) Save CSV: idx, score, then per-class columns in class_names order
    header = ["idx", "score"] + class_names
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        sc = scores.detach().cpu().numpy()
        probs = class_probs_full.detach().cpu().numpy()
        for i in range(T):
            w.writerow([i, float(sc[i])] + [float(p) for p in probs[i].tolist()])
    print(f"[ok] Saved scores+class-probs → {out_path}")

    
    #     --- Decide overall anomaly class ---
    # Simple rule: average class probs over frames with score > 0.5
    # sc = scores.detach().cpu().numpy()
    # probs = class_probs_full.detach().cpu().numpy()

    # threshold = 0.2
    # abnormal_mask = sc > threshold
    # if abnormal_mask.any():
    #     mean_probs = probs[abnormal_mask].mean(axis=0)
    #     pred_class_idx = int(mean_probs.argmax())
    #     pred_class_name = class_names[pred_class_idx]
    #     print(f"[result] Detected anomaly class: {pred_class_name} (id={pred_class_idx})")
    # else:
    #     print("[result] Video detected as Normal (no frames above threshold).")
    
    #     # --- Decide overall anomaly class (video-level) ---
    # sc = scores.detach().cpu().numpy()
    # probs = class_probs_full.detach().cpu().numpy()

    # threshold = 0.1  # tune this if needed
    # abnormal_mask = sc > threshold

    # if abnormal_mask.any():
    #     # Use only abnormal frames
    #     mean_probs = probs[abnormal_mask].mean(axis=0)
    #     pred_class_idx = int(mean_probs.argmax())
    #     pred_class_name = class_names[pred_class_idx]
    #     print(f"[result] Video anomaly detected: {pred_class_name} (id={pred_class_idx})")
    # else:
    #     print("[result] Video detected as Normal (no abnormal frames above threshold).")

    # --- Majority-vote anomaly class over abnormal frames ---
    sc_np = scores.detach().cpu().numpy()
    probs_np = class_probs_full.detach().cpu().numpy()
    labels_df = pd.read_csv(args.labels_file)
    class_names = labels_df["name"].tolist()
    normal_idx = args.normal_id

    threshold = 0.16   # tune as needed
    abnormal_idx = np.where(sc_np > threshold)[0]

    if len(abnormal_idx) == 0:
        print(f"[result] Video detected as Normal (no frames above threshold {threshold}).")
    else:
        votes = []
        for i in abnormal_idx:
            # take per-class probs for this frame
            frame_probs = probs_np[i].copy()
            frame_probs[normal_idx] = -1.0   # exclude Normal from argmax
            pred_cls = int(frame_probs.argmax())
            votes.append(pred_cls)

        # majority vote
        values, counts = np.unique(votes, return_counts=True)
        majority_cls = values[counts.argmax()]
        majority_name = class_names[majority_cls]

        print(f"[result] Video anomaly detected: {majority_name} (majority vote over {len(abnormal_idx)} abnormal frames)")

    
    
    # 7) Plot
    try:
        import matplotlib.pyplot as plt
        xs = np.arange(len(scores))
        plt.figure(figsize=(10,3)); plt.plot(xs, scores)
        plt.xlabel("Frame idx"); plt.ylabel("Anomaly score")
        plt.title(Path(args.video).stem)
        png = out_path.with_suffix(".png")
        plt.tight_layout(); plt.savefig(png, dpi=150)
        print(f"[ok] Saved plot   → {png}")
    except Exception as e:
        print("[warn] Plot skipped:", e)


if __name__ == "__main__":
    main()

#     try:
#         import matplotlib.pyplot as plt
#         xs = np.arange(T)
#         plt.figure(figsize=(10, 3))
#         plt.plot(xs, sc)
#         plt.xlabel("Frame idx (sampled)"); plt.ylabel("Anomaly score")
#         plt.title(Path(args.video).name)
#         png = out_path.with_suffix(".png")
#         plt.tight_layout(); plt.savefig(png, dpi=150)
#         print(f"[ok] Saved plot → {png}")
#     except Exception as e:
#         print("[warn] Plot skipped:", e)

#     print(f"[ok] Saved sampled frames → {frames_dir}")


# if __name__ == "__main__":
#     main()