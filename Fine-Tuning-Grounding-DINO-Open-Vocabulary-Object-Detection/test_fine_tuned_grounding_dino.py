import argparse

from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import torch
import torchvision.ops as ops
from torchvision.ops import box_convert

from grounding_dino_finetune_utils import (
    default_model_config_path,
    default_model_weights_path,
    load_finetuned_weights,
)


def apply_nms_per_phrase(image_source, boxes, logits, phrases, threshold=0.3):
    h, w, _ = image_source.shape
    scaled_boxes = boxes * torch.Tensor([w, h, w, h])
    scaled_boxes = box_convert(boxes=scaled_boxes, in_fmt="cxcywh", out_fmt="xyxy")
    nms_boxes_list, nms_logits_list, nms_phrases_list = [], [], []

    print(f"The unique detected phrases are {set(phrases)}")

    for unique_phrase in set(phrases):
        indices = [i for i, phrase in enumerate(phrases) if phrase == unique_phrase]
        phrase_scaled_boxes = scaled_boxes[indices]
        phrase_boxes = boxes[indices]
        phrase_logits = logits[indices]

        keep_indices = ops.nms(phrase_scaled_boxes, phrase_logits, threshold)
        nms_boxes_list.extend(phrase_boxes[keep_indices])
        nms_logits_list.extend(phrase_logits[keep_indices])
        nms_phrases_list.extend([unique_phrase] * len(keep_indices))

    if not nms_boxes_list:
        empty_boxes = boxes.new_empty((0, boxes.shape[-1]))
        empty_logits = logits.new_empty((0,))
        return empty_boxes, empty_logits, []

    return torch.stack(nms_boxes_list), torch.stack(nms_logits_list), nms_phrases_list


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned GroundingDINO checkpoint.")
    parser.add_argument(
        "--config",
        default=default_model_config_path(),
        help="Path to the GroundingDINO model config.",
    )
    parser.add_argument(
        "--weights",
        default="weights_less/model_weights_epoch_50.pth",
        help="Path to the fine-tuned checkpoint.",
    )
    parser.add_argument(
        "--base-weights",
        default=default_model_weights_path(),
        help="Path to the pretrained GroundingDINO weights used to initialize the model.",
    )
    parser.add_argument(
        "--image",
        default="test/images/maksssksksss774.png",
        help="Path to the test image.",
    )
    parser.add_argument(
        "--text-prompt",
        default="face mask . no face mask . mask worn incorrectly .",
        help="Text prompt used during inference.",
    )
    parser.add_argument("--box-threshold", type=float, default=0.68)
    parser.add_argument("--text-threshold", type=float, default=0.4)
    parser.add_argument(
        "--output",
        default="result.jpg",
        help="Where to save the annotated result.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device for inference.",
    )
    return parser.parse_args()


def process_image(
    model_config="groundingdino/config/GroundingDINO_SwinT_OGC.py",
    base_model_weights="weights/groundingdino_swint_ogc.pth",
    model_weights="weights_less/model_weights_epoch_50.pth",
    image_path="test/images/maksssksksss774.png",
    text_prompt="face mask . no face mask . mask worn incorrectly .",
    box_threshold=0.68,
    text_threshold=0.4,
    output_path="result.jpg",
    device="cpu",
):
    model = load_model(model_config, base_model_weights, device=device)
    load_finetuned_weights(model, model_weights, device=device)
    image_source, image = load_image(image_path)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )

    print(f"Original boxes size {boxes.shape}")
    boxes, logits, phrases = apply_nms_per_phrase(image_source, boxes, logits, phrases)
    print(f"NMS boxes size {boxes.shape}")

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite(output_path, annotated_frame)


if __name__ == "__main__":
    args = parse_args()
    process_image(
        model_config=args.config,
        base_model_weights=args.base_weights,
        model_weights=args.weights,
        image_path=args.image,
        text_prompt=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        output_path=args.output,
        device=args.device,
    )
