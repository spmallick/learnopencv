import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import math

# --- your net ---
from src.models.components.anomaly_clip import AnomalyCLIP

def load_feats(path, normalize=False, device="cpu"):
    x = np.load(path)  # [T, D]
    if x.ndim != 2:
        raise ValueError(f"{path} must be [T, D], got {x.shape}")
    t, d = x.shape
    x = torch.tensor(x, dtype=torch.float32, device=device)
    if normalize:
        x = F.normalize(x, dim=-1)
    # AnomalyCLIP (test_mode, load_from_features=True) expects [B, NC, T, D]
    x = x.unsqueeze(0).unsqueeze(0)  # [1,1,T,D]
    return x, t, d


def build_net_args(labels_file, arch="ViT-B/16"):
    return dict(
        # --- from anomaly_clip_ucfcrime.yaml ---
        arch=arch,
        shared_context=False,
        ctx_init="",
        n_ctx=8,
        seg_length=16,
        num_segments=32,
        select_idx_dropout_topk=0.7,
        select_idx_dropout_bottomk=0.7,
        heads=8,
        dim_heads=None,              # OK to leave None unless your code requires a number
        concat_features=False,
        emb_size=256,
        depth=1,
        num_topk=3,
        num_bottomk=3,
        labels_file=labels_file,
        normal_id=7,                 # <-- set to your actual "Normal" id if not 0
        dropout_prob=0.0,
        temporal_module="axial",
        direction_module="learned_encoder_finetune",
        selector_module="directions",
        batch_norm=True,
        feature_size=512,
        use_similarity_as_features=False,

        # --- crucial fix ---
        ctx_dim=512,                 # <— make this an INT so PromptLearner gets 512, not a DotMap

        # --- single‑video equivalents for ${data.*} ---
        load_from_features=True,
        stride=1,
        ncrops=1,
    )


def load_net_weights_from_ckpt(net, ckpt_path, strict=False, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    # keep only keys that belong to "net."
    net_state = {k.replace("net.", "", 1): v for k, v in state.items() if k.startswith("net.")}
    missing, unexpected = net.load_state_dict(net_state, strict=strict)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)

def main():
    ap = argparse.ArgumentParser("Pure AnomalyCLIP inference from .npy")
    ap.add_argument("--feats", required=True, help="Path to [T,D] .npy features")
    ap.add_argument("--ckpt", required=True, help="Lightning checkpoint with net.* weights")
    ap.add_argument("--labels_file", required=True, help="CSV with class id,name (same used in training)")
    ap.add_argument("--arch", default="ViT-B/16")
    ap.add_argument("--normalize", action="store_true", help="L2-normalize feature rows")
    ap.add_argument("--ncentroid", default=None, help="Optional ncentroid.pt from training")
    ap.add_argument("--out", default="outputs_single/scores.csv")
    ap.add_argument("--device", choices=["cuda","cpu"], default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # 1) Build net with the SAME args as training
    net_kwargs = build_net_args(labels_file=args.labels_file, arch=args.arch)
    net = AnomalyCLIP(**net_kwargs).to(device).eval()

    # 2) Load only net.* weights from the checkpoint
    load_net_weights_from_ckpt(net, args.ckpt, strict=False, device=device)

    # 3) Features + labels + ncentroid
    feats, T, D = load_feats(args.feats, normalize=args.normalize, device=device)  # [1,1,T,D]
    
    N = net.num_segments          # 32
    L = net.seg_length            # 16
    BASE = N * L                  # 512
    s = math.ceil(T / BASE)       # minimal segment_size so N*s*L >= T
    target_len = BASE * s
    if T < target_len:
        # repeat last feature row to pad
        pad_rows = feats[:, :, -1:, :].repeat(1, 1, target_len - T, 1)
        feats = torch.cat([feats, pad_rows], dim=2)  # [1,1,target_len,D]
    else:
        feats = feats
    
    labels = torch.zeros(T, dtype=torch.long, device=device)  # placeholder
    if args.ncentroid and Path(args.ncentroid).is_file():
        ncentroid = torch.load(args.ncentroid, map_location=device).view(-1)
    else:
        # quick fallback: mean of this sequence (works but not perfectly calibrated)
        # ncentroid = feats.view(T, D).mean(dim=0)
        ncentroid = feats.squeeze(0).squeeze(0).mean(dim=0)

    # 4) Forward (test_mode=True)
    with torch.no_grad():
        similarity, scores = net(
            image_features=feats,  # [B=1,NC=1,T,D]
            labels=labels,
            ncentroid=ncentroid,
            segment_size=1,
            test_mode=True,
        )  # similarity: [T, C-1], scores: [T]

    # 5) Save CSV
    import csv, numpy as np
    scores = scores[:T].detach().cpu().numpy()
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx","score"])
        for i, s in enumerate(scores):
            w.writerow([i, float(s)])
    print(f"Saved → {args.out}")

    # optional plot
    try:
        import matplotlib.pyplot as plt
        xs = np.arange(len(scores))
        plt.figure(figsize=(10,3)); plt.plot(xs, scores)
        plt.xlabel("Index (frame/segment)"); plt.ylabel("Anomaly score"); plt.tight_layout()
        png = Path(args.out).with_suffix(".png")
        plt.savefig(png, dpi=150); print(f"Saved → {png}")
    except Exception as e:
        print("Plot skipped:", e)

if __name__ == "__main__":
    main()
