from transformers import pipeline
import gradio as gr

pipe = pipeline(
    model='whisper_small_atco2/best_model',
    tokenizer='whisper_small_atco2/best_model',
    task='automatic-speech-recognition',
    device='cuda'
)  

def transcribe(audio):
    text = pipe(audio)['text']
    return text

iface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=['microphone', 'upload'], type='filepath'),
    outputs='text'
)

iface.launch(share=True)