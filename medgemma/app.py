import gradio as gr
from transformers import pipeline
from PIL import Image
import torch
import os
import spaces
import time

# Initialize the model pipeline
print("Loading MedGemma model...")
pipe = pipeline(
    "image-text-to-text",
    model="unsloth/medgemma-4b-it-bnb-4bit",
    torch_dtype=torch.bfloat16,
    # device="cuda" if torch.cuda.is_available() else "cpu",
    device_map="auto",
)
print("Model loaded successfully!")


@spaces.GPU()
def analyze_img(image, custom_prompt=None):
    """
    Analyze image using MedGemma model
    """
    if image is None:
        return "Please upload an image first."

    try:
        # System prompt for the model
        system_prompt_text = """You are a expert medical AI assistant with years of experience in interpreting medical images. Your purpose is to assist qualified clinicians by providing an detailed analysis of the provided medical image."""
        # Use custom prompt if provided, otherwise use default
        if custom_prompt and custom_prompt.strip():
            prompt_text = custom_prompt.strip()
        else:
            prompt_text = "Describe this image in detail, including any abnormalities or notable findings."

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt_text,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image", "image": image},
                ],
            },
        ]

        # Generate analysis
        output = pipe(text=messages, max_new_tokens=1024)
        full_response = output[0]["generated_text"][-1]["content"]

        partial_message = ""
        for char in full_response:
            partial_message += char
            time.sleep(0.01)  # Add a small delay to make the typing visible
            yield partial_message

    except Exception as e:
        return f"Error analyzing image: {str(e)}"


def load_sample_image():
    """Load the sample image if it exists"""
    sample_path = "./images/Infection.jpg"
    if os.path.exists(sample_path):
        return Image.open(sample_path)
    return None


# Create Gradio interface
with gr.Blocks(
    theme=gr.themes.Citrus(),
    title="MedGemma",
    css="""
    .header {
        text-align: center;
        background: linear-gradient(135deg, #f5af19 0%, #f12711 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .warning {
        background-color: #fff0e6;
        border: 3px solid #ffab73;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #8c2b00;
    }
    .gradio-container {
        max-width: 1200px;
        margin: auto;
    }
    .warning strong{
        color: inherit;
    }
    """,
) as demo:

    # Header
    gr.HTML(
        """
        <div class="header">
            <h1> MedGemma Medical Image Analysis and QnA</h1>
            <p>Advanced medical image analysis powered by Google's MedGemma</p>
        </div>
    """
    )

    # Warning disclaimer
    gr.HTML(
        """
        <div class="warning">
            <strong> Medical Disclaimer:</strong> This model is for educational and research purposes only. 
            It should not be used as a substitute for professional medical diagnosis or treatment. 
            Always consult qualified healthcare professionals for medical advice.
        </div>
    """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Medical Image (Radiology, Pathology, Dermatology, CT, X-Ray)")

            # Image input
            image_input = gr.Image(label="Input Image", type="pil", height=400, sources=["upload", "clipboard"])

            # Sample image button
            sample_btn = gr.Button("📋 Load Sample Image", variant="secondary", size="sm")

            # Custom prompt input
            gr.Markdown("### 💬 Custom Analysis Prompt (Optional)")
            custom_prompt = gr.Textbox(
                label="Custom Prompt",
                placeholder="Enter specific questions about the Image (e.g., 'Focus on the heart area' or 'Look for signs of pneumonia')",
                value="Describe this Image and Generate a compact Clinical report",
                lines=3,
                max_lines=5,
            )

            # Analyze button
            analyze_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Analysis Report")

            # Output text
            output_text = gr.Textbox(
                label="Generated Report",
                lines=28,
                max_lines=1024,
                show_label=False,
                show_copy_button=False,
                placeholder="Upload an X-ray image or CT scan or any othe medical image and click 'Analyze Image' to see the AI analysis results here...",
            )

            # Quick action buttons
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="sm")
                copy_btn = gr.Button("📋 Copy Results", variant="secondary", size="sm")

    # Example prompts section
    gr.Markdown("### 💡 Example Prompts")
    with gr.Row():
        example_prompts = [
            "Describe this X-ray in detail, including any abnormalities or notable findings.",
            "Describe the morphology of this skin lesion, focusing on color, border, and texture.",
            "What are the key histological features visible in this tissue sample?",
            "Look for any signs of fractures or bone abnormalities.",
            "Analyze this fundus image and describe the condition of the optic disc and vasculature.",
        ]

        for i, prompt in enumerate(example_prompts):
            gr.Button(f"Example {i+1}", size="sm").click(lambda p=prompt: p, outputs=custom_prompt)

    # Event handlers
    def clear_all():
        return None, "", ""

    sample_btn.click(fn=load_sample_image, outputs=image_input)

    analyze_btn.click(fn=analyze_img, inputs=[image_input, custom_prompt], outputs=output_text)

    clear_btn.click(fn=clear_all, outputs=[image_input, custom_prompt, output_text])

    copy_btn.click(
        fn=None,  # No Python function needed for this client-side action
        inputs=[output_text],
        js="""
    (text_to_copy) => {
        if (text_to_copy) {
            navigator.clipboard.writeText(text_to_copy);
            alert("Results copied to clipboard!");
        } else {
            alert("Nothing to copy!");
        }
    }
    """,
    )

    # Auto-analyze when image is uploaded (optional)
    image_input.change(
        fn=lambda img: analyze_img(img) if img is not None else "", inputs=image_input, outputs=output_text
    )

# Launch the app
if __name__ == "__main__":
    print("Starting Gradio interface...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True if you want to create a public link
        show_error=True,
        favicon_path=None,
    )
