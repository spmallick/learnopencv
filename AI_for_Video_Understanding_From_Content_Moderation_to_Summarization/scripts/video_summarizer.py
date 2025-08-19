#!/usr/bin/env python3
"""
Simple Video Summarizer using Qwen2.5-VL-3B-Instruct (single-model)

References:
- HF model card & quickstart examples for video usage:
  Qwen/Qwen2.5-VL-3B-Instruct
  (Requires recent transformers + qwen-vl-utils; see model card for details.)

Install (suggested):
  pip install -U "accelerate" "qwen-vl-utils[decord]==0.0.8"
  pip install -U "git+https://github.com/huggingface/transformers"

Usage:
  python qwen_video_summarizer.py --video path/to/video.mp4 --out summary.md
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


def summarize_video(
    video_path: str,
    max_new_tokens: int = 256,
) -> str:
    """
    Ask Qwen2.5-VL-3B-Instruct to summarize a local video.
    Returns raw generated text from the model.
    """

    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Load model & processor (per HF card)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # Build messages with a local video path and a prompt
    # Using "file://" absolute path so the processor/loader can find the file.
    vpath = Path(video_path).resolve()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{vpath}",
                },
                {
                    "type": "text",
                    "text": (
                        "Summarize this video. Return ONLY valid JSON with two keys:\n"
                        '  "bullets": a list of 5-7 concise bullet points (chronological),\n'
                        '  "paragraph": a short 120-word paragraph summary.\n'
                    ),
                },
            ],
        }
    ]

    # Prepare inputs (per HF card pattern)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    # Trim prompt tokens
    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output_ids)]
    out_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return out_text


def try_parse_json(text: str):
    """
    Try to parse strict JSON from the model output. If it fails,
    attempt to extract the JSON object from the text.
    """
    try:
        return json.loads(text)
    except Exception:
        # crude fallback: find the first {...} block
        import re
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def write_outputs(obj_or_text, out_path: Path):
    """
    Save Markdown + JSON. If obj_or_text is dict-like with bullets/paragraph,
    write pretty Markdown; otherwise, save raw text.
    """
    out_path = out_path.with_suffix(".md")
    json_path = out_path.with_suffix(".json")

    if isinstance(obj_or_text, dict) and "bullets" in obj_or_text and "paragraph" in obj_or_text:
        bullets = obj_or_text.get("bullets") or []
        paragraph = obj_or_text.get("paragraph") or ""

        md = ["# Video Summary", ""]
        if bullets:
            md.append("## Key Points")
            for b in bullets:
                md.append(f"- {b}")
            md.append("")
        if paragraph:
            md.append("## Short Summary")
            md.append(paragraph)
            md.append("")

        out_path.write_text("\n".join(md), encoding="utf-8")
        json_path.write_text(json.dumps(obj_or_text, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        # Raw text fallback
        out_path.write_text(str(obj_or_text), encoding="utf-8")

    return str(out_path), str(json_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to a local video file (e.g., .mp4)")
    ap.add_argument("--out", default="video_summary.md", help="Output Markdown path (JSON will be next to it)")
    ap.add_argument("--max_new_tokens", type=int, default=256, help="Max new tokens for generation")
    args = ap.parse_args()

    raw = summarize_video(args.video, max_new_tokens=args.max_new_tokens)
    maybe_json = try_parse_json(raw)

    out_md, out_json = write_outputs(maybe_json if maybe_json is not None else raw, Path(args.out))
    print(f"Saved:\n- {out_md}\n- {out_json if Path(out_json).exists() else '(no JSON parsed)'}")


if __name__ == "__main__":
    main()
