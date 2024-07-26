"""
Script to compare time for fine-tuned Whisper models.
"""

import torch
import time
import os

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

model_dirs = [
    'whisper_tiny_atco2_v2/best_model',
    'whisper_base_atco2/best_model',
    'whisper_small_atco2/best_model'
]

input_dir = 'inference_data'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

for model_id in model_dirs:
    print(f"\nEvaluating model: {model_id}")

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

    total_time = 0
    num_runs = 0

    for _ in range(10):
        for filename in os.listdir(input_dir):
            if filename.endswith('.wav'):
                start_time = time.time()
                result = pipe(os.path.join(input_dir, filename))
                end_time = time.time()
                total_time += (end_time - start_time)
                num_runs += 1

    average_time = total_time / num_runs
    print(f"\nAverage time taken for {model_id}: {average_time} seconds")
