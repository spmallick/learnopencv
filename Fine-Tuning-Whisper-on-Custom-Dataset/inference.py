import torch
import argparse

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    required=True
)
parser.add_argument(
    '--input',
    required=True
)
args = parser.parse_args()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = args.model

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, 
    low_cpu_mem_usage=True, 
    use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    'automatic-speech-recognition',
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device
)

result = pipe(args.input, generate_kwargs={'task': 'transcribe'})

print('\n', result)