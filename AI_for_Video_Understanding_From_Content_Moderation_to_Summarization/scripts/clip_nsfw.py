import torch
import gradio as gr
from PIL import Image
import cv2
import google.generativeai as genai
import numpy as np
import requests
# from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel
import io
import os
import json

# ------------------------ GEMINI SETUP ------------------------
genai.configure(api_key="")

# ------------------------ LOAD CLIP MODEL ------------------------
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
# ------------------------ EXTRACT KEY FRAMES FROM VIDEO ------------------------
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    # video_fps = cap.get(cv2.CAP_PROP_FPS)
    # print(f"Video FPS: {video_fps}")

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Capture keyframes at regular intervals (e.g., every 5 seconds)
        # if idx % 20 == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append((idx, Image.fromarray(rgb)))
        idx += 1
    cap.release()
    return frames

# ------------------------ CLASSIFICATION FUNCTION ------------------------
# def classify_frame_with_blip(frame: Image.Image):
#     # BLIP uses image captioning to determine the content of an image

#     labels = ["normal", "violence", "nudity", "accident"] 
#     inputs = processor(images=frame, text=labels, return_tensors="pt", padding=True)

#     with torch.no_grad():
#         # Generate the caption or classification (here we use captioning)
#         output = model.generate(**inputs)

#     # Decode the caption
#     caption = processor.decode(output[0], skip_special_tokens=True)
    
#     # Classify based on the content of the caption
#     # You may want to add custom labels
#     # Simplified classification based on caption
#     if "violence" in caption.lower():
#         label = "violence"
#     elif "nudity" in caption.lower():
#         label = "nudity"
#     elif "accident" in caption.lower():
#         label = "accident"
#     else:
#         label = "normal"
    
#     # Confidence is a bit trickier to determine, so we set it manually here for simplicity
#     confidence = 0.9  # Placeholder; you can use other methods to measure confidence

#     return label, confidence

def classify_frame_with_clip(frame: Image.Image):
    text_inputs = ["this image is normal", "this image contains nudity", "this image contains enticing or sensual content", "this image contains violence"]
    inputs = processor(text=text_inputs, images=frame, return_tensors="pt", padding=True)

    with torch.no_grad():
        # Get the full output from the CLIP model
        output = model(**inputs)

    # Access the logits from the output object
    logits_per_image = output.logits_per_image  # Logits for image classification

    # Convert logits to probabilities (softmax)
    probs = logits_per_image.softmax(dim=-1)  # Convert logits to probabilities
    confidence, predicted_class = probs.max(dim=-1)
    print("Predicted Class: ", predicted_class)

    labels = ["normal", "nudity", "enticing or sensual", "violent"]  # Example labels, modify as needed
    label = labels[predicted_class.item()]
    print("predicted label", label)
    return label, confidence.item()


# ------------------------ GEMINI EXPLANATION FUNCTION ------------------------

def get_gemini_explanation(images: list, flaggedd_frames_: list) -> str:

    try:
        import io

        llm = genai.GenerativeModel("gemini-2.5-flash")

        # Convert all images to byte format
        image_parts = []
        for img in images:
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            img_bytes = buffer.getvalue()
            image_parts.append({
                "mime_type": "image/jpeg",
                "data": img_bytes
            })

        # Prompt specifically for images and NSFW classification
        prompt = f"""
                    You are given a set of image frames extracted from a video, and a list of predictions for some of those frames indicating they are NSFW (Not Safe For Work). Each dictionary in the list contains:
                    - frame_id: the frame number,
                    - label: always "nsfw",
                    - confidence: the model's confidence score.

                    Here is the metadata:
                    {json.dumps(flaggedd_frames_, indent=2)}
                    **Do not include words like metadata or images in the explanation**
                   Analyze the flagged images and their metadata to explain clearly in **no more than 2 lines** why the video has been classified as NSFW.
                """

        # Combine prompt and all image parts
        response = llm.generate_content(
            [prompt] + image_parts
        )

        explanation = response.parts[0].text.strip()
        return explanation

    except Exception as e:
        return f"Error: {str(e)}"


# ------------------------ MAIN MODERATION FUNCTION ------------------------

def moderate_video(video_file):
    if video_file is None:
        return "No video uploaded", {}

    frames = extract_frames(video_file)
    if not frames:
        return "No frames extracted", {}

    flagged_frames = []
    img_list = []
    for idx, img in frames:
        # Classify the frame (normal or nsfw)
        label, confidence = classify_frame_with_clip(img)
        
        # If the frame is classified as "nsfw", get the reasoning from Gemini
        if label != "normal" and confidence > 0.5:
              # Get the explanation from Gemini
            flagged_frames.append({
                "frame_id": idx,
                "classification": label,
                "confidence": confidence,
                # "explanation": explanation
            })
            img_list.append(img)
            #remember to generate the whole video summary through the combined flagged frames throughout the video.

    if not flagged_frames:
        return "✅ All frames are appropriate.", "None", {}
    
    explanation = get_gemini_explanation(img_list, flagged_frames)

    final_perct = (len(img_list)/len(frames))*100

    return explanation, final_perct, flagged_frames

# ------------------------ GRADIO INTERFACE ------------------------

iface = gr.Interface(
    fn=moderate_video,
    inputs=gr.Video(label="Upload a Video"),
    outputs=[
        gr.Textbox(label="Why is it Flagged? 🤔"),
        gr.Textbox(label="NSFW Percentage throughout the video"),
        gr.JSON(label="Flagged Frames")
    ],
    title="Video Content Moderation with CLIP and Gemini 📼",
    description="Detects unsafe content in a video and provides detailed classification and reasoning using Google Gemini."
)

if __name__ == "__main__":
    iface.launch(share=True)
