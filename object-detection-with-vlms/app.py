import os
import json
import time

import gradio as gr
import numpy as np
import torch
# from gradio.themes.Soft import Soft
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Gemma3ForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

from spaces import GPU
import supervision as sv

# --- Config ---
# IMPORTANT: Both models are gated. You must be logged in to your Hugging Face account
# and have been granted access to use them.
from huggingface_hub import login
hf_token = os.environ.get("HF_TOKEN")
login(token=hf_token)



model_qwen_id = "Qwen/Qwen2.5-VL-3B-Instruct"
model_gemma_id = "google/gemma-3-4b-it"

# Load Qwen Model
model_qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_qwen_id, torch_dtype="auto", device_map="auto"
)
min_pixels = 224 * 224
max_pixels = 1024 * 1024
processor_qwen = AutoProcessor.from_pretrained(
    model_qwen_id, min_pixels=min_pixels, max_pixels=max_pixels
)

# Load Gemma Model
model_gemma = Gemma3ForConditionalGeneration.from_pretrained(
    model_gemma_id,
    torch_dtype=torch.bfloat16,  # Recommended dtype for Gemma
    device_map="auto"
)
processor_gemma = AutoProcessor.from_pretrained(model_gemma_id)


def extract_model_short_name(model_id):
    return model_id.split("/")[-1].replace("-", " ").replace("_", " ")


model_qwen_name = extract_model_short_name(model_qwen_id)  # → "Qwen2.5 VL 3B Instruct"
model_gemma_name = extract_model_short_name(model_gemma_id) # → "gemma 3 4b it"


def create_annotated_image(image, json_data, height, width):
    try:
        # Standardize parsing for outputs wrapped in markdown
        if "```json" in json_data:
            parsed_json_data = json_data.split("```json")[1].split("```")[0]
        else:
            parsed_json_data = json_data
        bbox_data = json.loads(parsed_json_data)
    except Exception:
        # If parsing fails, return the original image
        return image

    # Ensure bbox_data is a list
    if not isinstance(bbox_data, list):
        bbox_data = [bbox_data]


    original_width, original_height = image.size
    x_scale = original_width / width
    y_scale = original_height / height

    points = []
    point_labels = []

    annotated_image = np.array(image.convert("RGB"))
    detections_exist = False

    # Check if there are bounding boxes in the data to create detections
    if any("box_2d" in item for item in bbox_data):
        detections_exist = True
        # Use Qwen parser as a generic VLM parser for bounding boxes
        detections = sv.Detections.from_vlm(vlm = sv.VLM.QWEN_2_5_VL,
                                            result=json_data,
                                            # resolution_wh is the size model "sees"
                                            resolution_wh=(width, height))
        bounding_box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
        label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

        annotated_image = bounding_box_annotator.annotate(
            scene=annotated_image, detections=detections
        )
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections
        )

    # Handle points separately
    for item in bbox_data:
        label = item.get("label", "")
        if "point_2d" in item:
            x, y = item["point_2d"]
            scaled_x = int(x * x_scale)
            scaled_y = int(y * y_scale)
            points.append([scaled_x, scaled_y])
            point_labels.append(label)

    if points:
        points_array = np.array(points).reshape(1, -1, 2)
        key_points = sv.KeyPoints(xy=points_array)
        vertex_annotator = sv.VertexAnnotator(radius=5, color=sv.Color.BLUE)
        annotated_image = vertex_annotator.annotate(
            scene=annotated_image, key_points=key_points
        )

    return Image.fromarray(annotated_image)


@GPU
def detect_qwen(image, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    t0 = time.perf_counter()
    text = processor_qwen.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor_qwen(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model_qwen.device)

    generated_ids = model_qwen.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor_qwen.batch_decode(
        generated_ids_trimmed,
        do_sample=True,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    elapsed_ms = (time.perf_counter() - t0) * 1_000

    # These dimensions are specific to how Qwen's processor handles images
    input_height = inputs["image_grid_thw"][0][1] * 14
    input_width = inputs["image_grid_thw"][0][2] * 14

    annotated_image = create_annotated_image(
        image, output_text, input_height, input_width
    )

    time_taken = f"**Inference time ({model_qwen_name}):** {elapsed_ms:.0f} ms"
    return annotated_image, output_text, time_taken


@GPU
def detect_gemma(image, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    t0 = time.perf_counter()
    inputs = processor_gemma.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model_gemma.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model_gemma.generate(**inputs, max_new_tokens=1024, do_sample=False)
    
    generation_trimmed = generation[0][input_len:]
    output_text = processor_gemma.decode(generation_trimmed, skip_special_tokens=True)
    elapsed_ms = (time.perf_counter() - t0) * 1_000

    # Gemma's vision encoder normalizes images to a fixed size (e.g., 896x896)
    input_height = 896
    input_width = 896

    annotated_image = create_annotated_image(
        image, output_text, input_height, input_width
    )

    time_taken = f"**Inference time ({model_gemma_name}):** {elapsed_ms:.0f} ms"
    return annotated_image, output_text, time_taken


def detect(image, prompt_model_1, prompt_model_2):
    STANDARD_SIZE = (1024, 1024)
    image.thumbnail(STANDARD_SIZE)

    annotated_image_model_1, output_text_model_1, timing_1 = detect_qwen(
        image, prompt_model_1
    )
    annotated_image_model_2, output_text_model_2, timing_2 = detect_gemma(
        image, prompt_model_2
    )

    return (
        annotated_image_model_1,
        output_text_model_1,
        timing_1,
        annotated_image_model_2,
        output_text_model_2,
        timing_2,
    )


css_hide_share = """
button#gradio-share-link-button-0 {
    display: none !important;
}
"""

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(), css=css_hide_share) as demo:
    gr.Markdown("# Object Detection & Understanding: Qwen vs. Gemma")
    gr.Markdown(
        "### Compare object detection, visual grounding, and keypoint detection using natural language prompts with two leading VLMs."
    )
    gr.Markdown("""
    *Powered by [Qwen2.5-VL 3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) and [Gemma 3 4B IT](https://huggingface.co/google/gemma-3-4b-it). For best results, ask the model to return a JSON list in a markdown block. Inspired by the [HF Team's space](https://huggingface.co/spaces/sergiopaniego/vlm_object_understanding), selecting `detect` for categories with "Object Detection" `point` for the ones with "Keypoint Detection", and reasoning-based querying for all others.*
    """)

    with gr.Row():
        with gr.Column(scale=2):
            image_input = gr.Image(label="Upload an image", type="pil", height=400)
            prompt_input_model_1 = gr.Textbox(
                label=f"Enter your prompt for {model_qwen_name}",
                placeholder="e.g., Detect all red cars. Return a JSON list with 'box_2d' and 'label'.",
            )
            prompt_input_model_2 = gr.Textbox(
                label=f"Enter your prompt for {model_gemma_name}",
                placeholder="e.g., Detect all red cars. Return a JSON list with 'box_2d' and 'label'.",
            )
            generate_btn = gr.Button(value="Generate")

        with gr.Column(scale=1):
            output_image_model_1 = gr.Image(
                type="pil", label=f"Annotated image from {model_qwen_name}", height=400
            )
            output_textbox_model_1 = gr.Textbox(
                label=f"Model response from {model_qwen_name}", lines=10
            )
            output_time_model_1 = gr.Markdown()

        with gr.Column(scale=1):
            output_image_model_2 = gr.Image(
                type="pil",
                label=f"Annotated image from {model_gemma_name}",
                height=400,
            )
            output_textbox_model_2 = gr.Textbox(
                label=f"Model response from {model_gemma_name}", lines=10
            )
            output_time_model_2 = gr.Markdown()

    gr.Markdown("### Examples")
    
    prompt_obj_detect = "Detect all objects in this image. For each object, provide a 'box_2d' and a 'label'. Return the output as a JSON list inside a markdown block."
    prompt_candy_detect = "Detect all individual candies in this image. For each, provide a 'box_2d' and a 'label'. Return the output as a JSON list inside a markdown block."
    prompt_car_count = "Count the number of red cars in the image."
    prompt_candy_count = "Count the number of blue candies in the image."
    prompt_car_keypoint = "Identify the red cars in this image. For each, detect its key points and return their positions as 'point_2d' in a JSON list inside a markdown block."
    prompt_candy_keypoint = "Identify the blue candies in this image. For each, detect its key points and return their positions as 'point_2d' in a JSON list inside a markdown block."
    prompt_car_ground = "Detect the red car that is leading in this image. Return its location with 'box_2d' and 'label' in a JSON list inside a markdown block."
    prompt_candy_ground = "Detect the blue candy at the top of the group. Return its location with 'box_2d' and 'label' in a JSON list inside a markdown block."


    example_prompts = [
        ["examples/example_1.jpg", prompt_obj_detect, prompt_obj_detect],
        ["examples/example_2.jpg", prompt_candy_detect, prompt_candy_detect],
        ["examples/example_1.jpg", prompt_car_count, prompt_car_count],
        ["examples/example_2.jpg", prompt_candy_count, prompt_candy_count],
        ["examples/example_1.jpg", prompt_car_keypoint, prompt_car_keypoint],
        ["examples/example_2.jpg", prompt_candy_keypoint, prompt_candy_keypoint],
        ["examples/example_1.jpg", prompt_car_ground, prompt_car_ground],
        ["examples/example_2.jpg", prompt_candy_ground, prompt_candy_ground],
    ]

    gr.Examples(
        examples=example_prompts,
        inputs=[
            image_input,
            prompt_input_model_1,
            prompt_input_model_2,
        ],
        label="Click an example to populate the input",
    )

    generate_btn.click(
        fn=detect,
        inputs=[
            image_input,
            prompt_input_model_1,
            prompt_input_model_2,
        ],
        outputs=[
            output_image_model_1,
            output_textbox_model_1,
            output_time_model_1,
            output_image_model_2,
            output_textbox_model_2,
            output_time_model_2,
        ],
    )

if __name__ == "__main__":
    demo.launch()