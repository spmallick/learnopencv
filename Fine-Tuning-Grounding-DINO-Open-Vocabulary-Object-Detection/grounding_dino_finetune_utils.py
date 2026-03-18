from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import groundingdino
import torch
import torch.nn.functional as F
from torchvision.ops import box_convert, generalized_box_iou

from groundingdino.util.inference import load_image, load_model
from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span


def default_model_config_path() -> str:
    package_dir = Path(groundingdino.__file__).resolve().parent
    return str(package_dir / "config" / "GroundingDINO_SwinT_OGC.py")


def default_model_weights_path() -> str:
    package_dir = Path(groundingdino.__file__).resolve().parent
    repo_root = package_dir.parent
    candidates = [
        Path("weights") / "groundingdino_swint_ogc.pth",
        repo_root / "weights" / "groundingdino_swint_ogc.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[-1])


def load_finetuned_weights(model, checkpoint_path: str, device: str = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    return model


def _normalize_labels(labels: Iterable[str]) -> list[str]:
    normalized = []
    for label in labels:
        normalized_label = label.strip().lower()
        if normalized_label:
            normalized.append(normalized_label)
    return normalized


def build_training_caption(labels: Sequence[str]) -> tuple[str, list[list[list[int]]]]:
    normalized_labels = _normalize_labels(labels)
    unique_labels = list(dict.fromkeys(normalized_labels))
    caption, label_to_token_spans = build_captions_and_token_span(
        unique_labels, force_lowercase=True
    )
    per_box_token_spans = [label_to_token_spans[label] for label in normalized_labels]
    return caption, per_box_token_spans


def _prepare_targets(
    box_targets: Sequence[Sequence[float]], image_height: int, image_width: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    xyxy_boxes = torch.tensor(box_targets, dtype=torch.float32, device=device)
    scale = torch.tensor(
        [image_width, image_height, image_width, image_height], dtype=torch.float32, device=device
    )
    xyxy_boxes = xyxy_boxes / scale
    cxcywh_boxes = box_convert(xyxy_boxes, in_fmt="xyxy", out_fmt="cxcywh")
    return xyxy_boxes, cxcywh_boxes


def _greedy_match(cost_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if cost_matrix.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=cost_matrix.device)
        return empty, empty

    working_cost = cost_matrix.clone()
    num_predictions, num_targets = working_cost.shape
    matched_predictions = []
    matched_targets = []

    while len(matched_predictions) < min(num_predictions, num_targets):
        flat_index = working_cost.argmin()
        prediction_index = int(flat_index // num_targets)
        target_index = int(flat_index % num_targets)
        if not torch.isfinite(working_cost[prediction_index, target_index]):
            break

        matched_predictions.append(prediction_index)
        matched_targets.append(target_index)
        working_cost[prediction_index, :] = float("inf")
        working_cost[:, target_index] = float("inf")

    return (
        torch.tensor(matched_predictions, dtype=torch.long, device=cost_matrix.device),
        torch.tensor(matched_targets, dtype=torch.long, device=cost_matrix.device),
    )


def train_image(
    model,
    image_source,
    image: torch.Tensor,
    caption_objects: Sequence[str],
    box_target: Sequence[Sequence[float]],
    device: str | None = None,
) -> torch.Tensor:
    """Compute a lightweight fine-tuning loss with the public GroundingDINO API.

    GroundingDINO does not currently ship the private `groundingdino.util.train`
    helper that the original tutorial referenced. This helper recreates the
    minimum training path needed by the example by:
    1. building a GroundingDINO caption from the object labels,
    2. matching predictions to GT boxes with a simple greedy cost,
    3. optimizing token alignment plus box regression losses.
    """

    if device is None:
        device = next(model.parameters()).device

    model = model.to(device)
    image = image.to(device)

    caption, token_spans = build_training_caption(caption_objects)
    outputs = model(image[None], captions=[caption], unset_image_tensor=True)

    pred_logits = outputs["pred_logits"][0]
    pred_boxes = outputs["pred_boxes"][0]

    target_boxes_xyxy, target_boxes_cxcywh = _prepare_targets(
        box_targets=box_target,
        image_height=image_source.shape[0],
        image_width=image_source.shape[1],
        device=device,
    )

    tokenized = model.tokenizer(caption, return_tensors="pt")
    positive_map = create_positive_map_from_span(
        tokenized=tokenized,
        token_span=token_spans,
        max_text_len=pred_logits.shape[-1],
    ).to(device)

    pred_boxes_xyxy = box_convert(pred_boxes, in_fmt="cxcywh", out_fmt="xyxy")
    token_scores = pred_logits.sigmoid() @ positive_map.T
    matching_cost = (
        (1 - token_scores)
        + torch.cdist(pred_boxes, target_boxes_cxcywh, p=1)
        + (1 - generalized_box_iou(pred_boxes_xyxy, target_boxes_xyxy))
    )

    matched_predictions, matched_targets = _greedy_match(matching_cost.detach())
    if matched_predictions.numel() == 0:
        return pred_logits.sigmoid().mean() * 0

    matched_logits = pred_logits[matched_predictions]
    matched_text_targets = positive_map[matched_targets]
    finite_logit_mask = torch.isfinite(matched_logits)
    if finite_logit_mask.any():
        classification_loss = F.binary_cross_entropy_with_logits(
            matched_logits[finite_logit_mask],
            matched_text_targets[finite_logit_mask],
        )
    else:
        classification_loss = pred_boxes.sum() * 0
    box_loss = F.l1_loss(
        pred_boxes[matched_predictions],
        target_boxes_cxcywh[matched_targets],
        reduction="none",
    ).sum(dim=1).mean()
    giou_loss = (
        1
        - generalized_box_iou(
            pred_boxes_xyxy[matched_predictions], target_boxes_xyxy[matched_targets]
        ).diag()
    ).mean()

    unmatched_mask = torch.ones(pred_logits.shape[0], dtype=torch.bool, device=device)
    unmatched_mask[matched_predictions] = False
    if unmatched_mask.any():
        background_logits = pred_logits[unmatched_mask]
        background_logits = background_logits[torch.isfinite(background_logits)]
        background_loss = background_logits.sigmoid().mean() if background_logits.numel() else pred_boxes.sum() * 0
    else:
        background_loss = pred_boxes.sum() * 0

    return classification_loss + (5.0 * box_loss) + (2.0 * giou_loss) + (0.1 * background_loss)
