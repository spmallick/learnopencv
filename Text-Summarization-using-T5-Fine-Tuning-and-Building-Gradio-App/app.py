import gradio as gr

from transformers import T5ForConditionalGeneration, T5Tokenizer

def summarize_text(text):
    # Preprocess the text
    inputs = tokenizer.encode(
        "summarize: " + text,
        return_tensors='pt',
        max_length=512,
        truncation=True,
        padding='max_length'
    )

    # Generate the summary
    summary_ids = model.generate(
        inputs,
        max_length=50,
        num_beams=5,
        # early_stopping=True
    )

    # Decode and return the summary
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

model_path = 'results_t5base/checkpoint-4450'  # the path where you saved your model
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained('results_t5base')

interface = gr.Interface(
    fn=summarize_text, 
    inputs=gr.Textbox(lines=10, placeholder='Enter Text Here...', label='Input text'), 
    outputs=gr.Textbox(label='Summarized Text'),
    title='Text Summarizer using T5'
)
interface.launch()