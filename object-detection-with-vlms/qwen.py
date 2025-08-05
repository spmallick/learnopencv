import json
import time

import gradio as gr
import numpy as np

# from gradio.themes.Soft import Soft
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from spaces import GPU
import supervision as sv

# --- Config ---
model_qwen_id = "Qwen/Qwen2.5-VL-3B-Instruct"

model_qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_qwen_id, torch_dtype="auto", device_map="auto")


def extract_model_short_name(model_id):
    return model_id.split("/")[-1].replace("-", " ").replace("_", " ")


model_qwen_name = extract_model_short_name(model_qwen_id)  # → "Qwen2.5 VL 3B Instruct"


min_pixels = 224 * 224
max_pixels = 1024 * 1024
processor_qwen = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels
)


def create_annotated_image(image, json_data, height, width):
    try:
        parsed_json_data = json_data.split("```json")[1].split("```")[0]
        bbox_data = json.loads(parsed_json_data)
    except Exception:
        return image

    original_width, original_height = image.size
    x_scale = original_width / width
    y_scale = original_height / height

    points = []
    point_labels = []

    for item in bbox_data:
        label = item.get("label", "")
        if "point_2d" in item:
            x, y = item["point_2d"]
            scaled_x = int(x * x_scale)
            scaled_y = int(y * y_scale)
            points.append([scaled_x, scaled_y])
            point_labels.append(label)

        annotated_image = np.array(image.convert("RGB"))

        detections = sv.Detections.from_vlm(
            vlm=sv.VLM.QWEN_2_5_VL,
            result=json_data,
            input_wh=(original_width, original_height),
            resolution_wh=(original_width, original_height),
        )
        bounding_box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
        label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

        annotated_image = bounding_box_annotator.annotate(scene=annotated_image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    if points:
        points_array = np.array(points).reshape(1, -1, 2)
        key_points = sv.KeyPoints(xy=points_array)
        vertex_annotator = sv.VertexAnnotator(radius=5, color=sv.Color.BLUE)
        # vertex_label_annotator = sv.VertexLabelAnnotator(text_scale=0.5, border_radius=2)

        annotated_image = vertex_annotator.annotate(scene=annotated_image, key_points=key_points)

        # annotated_image = vertex_label_annotator.annotate(
        #     scene=annotated_image,
        #     key_points=key_points,
        #     labels=point_labels
        # )

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
    text = processor_qwen.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor_qwen(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model_qwen.device)

    generated_ids = model_qwen.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor_qwen.batch_decode(
        generated_ids_trimmed,
        do_sample=True,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    elapsed_ms = (time.perf_counter() - t0) * 1_000

    input_height = inputs["image_grid_thw"][0][1] * 14
    input_width = inputs["image_grid_thw"][0][2] * 14

    annotated_image = create_annotated_image(image, output_text, input_height, input_width)

    time_taken = f"**Inference time ({model_qwen_name}):** {elapsed_ms:.0f} ms"
    return annotated_image, output_text, time_taken


def detect(image, prompt):
    STANDARD_SIZE = (1024, 1024)
    image.thumbnail(STANDARD_SIZE)

    annotated_image, output_text, time_taken = detect_qwen(image, prompt)

    return (
        annotated_image,
        output_text,
        time_taken,
    )


css_hide_share = """
button#gradio-share-link-button-0 {
    display: none !important;
}
"""

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft(), css=css_hide_share) as demo:
    gr.Markdown("# Object Detection & Understanding with Vision Language Models")
    gr.Markdown(
        "### Explore object detection, visual grounding, keypoint detection, and/or object counting through natural language prompts."
    )
    gr.Markdown(
        """
    *Powered by [Qwen2.5-VL 3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct). Inspired by the [HF Team's space](https://huggingface.co/spaces/sergiopaniego/vlm_object_understanding).*
    """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload an image", type="pil", height=400)
            prompt_input = gr.Textbox(
                label=f"Enter your prompt for {model_qwen_name}",
                placeholder="e.g., Detect all red cars in the image",
            )
            generate_btn = gr.Button(value="Generate")

        with gr.Column(scale=1):
            output_image = gr.Image(type="pil", label=f"Annotated image from {model_qwen_name}", height=400)
            output_textbox = gr.Textbox(label=f"Model response from {model_qwen_name}", lines=10)
            output_time = gr.Markdown()

    gr.Markdown("### Examples")
    example_prompts = [
        [
            "examples/example_1.jpg",
            "Detect all objects in the image and return their locations and labels.",
        ],
        [
            "examples/example_2.jpg",
            "Detect all the individual candies in the image and return their locations and labels.",
        ],
        [
            "examples/example_1.jpg",
            "Count the number of red cars in the image.",
        ],
        [
            "examples/example_2.jpg",
            "Count the number of blue candies in the image.",
        ],
        [
            "examples/example_1.jpg",
            "Identify the red cars in this image, detect their key points and return their positions in the form of points.",
        ],
        [
            "examples/example_2.jpg",
            "Identify the blue candies in this image, detect their key points and return their positions in the form of points.",
        ],
        [
            "examples/example_1.jpg",
            "Detect the red car that is leading in this image and return its location and label.",
        ],
        [
            "examples/example_2.jpg",
            "Detect the blue candy located at the top of the group in this image and return its location and label.",
        ],
    ]

    gr.Examples(
        examples=example_prompts,
        inputs=[
            image_input,
            prompt_input,
        ],
        label="Click an example to populate the input",
    )

    generate_btn.click(
        fn=detect,
        inputs=[
            image_input,
            prompt_input,
        ],
        outputs=[
            output_image,
            output_textbox,
            output_time,
        ],
    )

if __name__ == "__main__":
    demo.launch()
