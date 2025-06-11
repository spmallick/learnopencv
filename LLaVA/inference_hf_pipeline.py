import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "llava-hf/llava-1.5-7b-hf" #define model id here, there are couple of other variants lile: 
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    load_in_4bit = True
).to(0)

processor = AutoProcessor.from_pretrained(model_id)

conversation = [
    {

      "role": "user",
      "content": [
          {"type": "text", "text": "input_prompt"},
          {"type": "image"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)


image = "input_image_path"
raw_image = Image.open(image)
inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))
