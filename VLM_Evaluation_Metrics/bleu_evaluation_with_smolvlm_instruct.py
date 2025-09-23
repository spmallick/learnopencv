import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from PIL import Image
import numpy as np
from typing import List, Dict, Any

import warnings
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# Download required NLTK data
nltk.download('punkt_tab', quiet=True)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Create input messages
messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Can you describe the image?"}
                ]
            },
        ]

# Initialize smolVLM model and processor
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-Instruct",
    torch_dtype=torch.bfloat16,
    _attn_implementation=None if DEVICE == "cuda" else "eager",
).to(DEVICE)

#make predictions with smolVLM model
def model_pred(img):
    try:
        # Prepare inputs
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=img, return_tensors="pt")
        inputs = inputs.to(DEVICE)

        # Generate outputs
        generated_ids = model.generate(**inputs, max_new_tokens=500)
        generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        return generated_texts

    except Exception as e:
        print(f"Error loading smolVLM model: {e}")

#load image using Pillow library to perform captionin.
def load_image(image_path: str) -> Image.Image:
    """Load an image from the given path."""
    try:
        return Image.open(image_path)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

#Compute BLEU score for a single generated text against reference texts.
def compute_bleu_score(reference_texts: List[str], generated_text: str) -> float:
    reference_tokens = [nltk.word_tokenize(ref.lower()) for ref in reference_texts]
    generated_tokens = nltk.word_tokenize(generated_text.lower())
    
    smoothing = SmoothingFunction().method1
    bleu_score = sentence_bleu(reference_tokens, generated_tokens, 
                             weights=(0.25, 0.25, 0.25, 0.25), 
                             smoothing_function=smoothing)
    return bleu_score

#Evaluate VLM on a dataset using BLEU metric.
def evaluate_vlm(image_paths: List[str], reference_captions: List[List[str]]) -> Dict[str, Any]:
    bleu_scores = []
    
    for img_path, refs in zip(image_paths, reference_captions):
        image = load_image(img_path)
        if image is None:
            continue
            
        generated_caption = model_pred(image)
        bleu = compute_bleu_score(refs, generated_caption)
        bleu_scores.append(bleu)
        
        print(f"Image: {img_path}")
        print(f"Generated: {generated_caption}")
        print(f"References: {refs}")
        print(f"BLEU Score: {bleu:.4f}\n")
    
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0.0
    
    return {
        "individual_bleu_scores": bleu_scores,
        "average_bleu_score": avg_bleu
    }

def main():
    dataset = [
        {
            "image_path": "MEDIA/surf.jpg",
            "references": [
                "A close-up of a purple flower with a small insect resting on one of its petals.",
                "A delicate pinkish-purple bloom with a bright yellow center and a fly resting on the petal.",
                "A single daisy-like flower captured in detail, with soft petals and a visiting insect."
            ]
        },
        {
            "image_path": "MEDIA/sample_image_2.jpg",
            "references": [
                "A surfer in a red shirt riding a wave on a yellow and white surfboard.",
                "A man skillfully carving through the ocean surf as water splashes around him.",
                "A wave crashes while a surfer balances low on his board, maintaining speed and control."
            ]
        }
    ]
    
    image_paths = [item["image_path"] for item in dataset]
    reference_captions = [item["references"] for item in dataset]
    
    results = evaluate_vlm(image_paths, reference_captions)
    
    print("Evaluation Summary:")
    print(f"Average BLEU Score: {results['average_bleu_score']:.4f}")
    print(f"Individual BLEU Scores: {[f'{score:.4f}' for score in results['individual_bleu_scores']]}")

if __name__ == "__main__":
    main()