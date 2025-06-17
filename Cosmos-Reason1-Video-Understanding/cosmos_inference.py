from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import torch

# Model and quantization setup
MODEL_PATH = "nvidia/Cosmos-Reason1-7B"
# MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# Load model
llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Message structure
video_messages = [
    {"role": "system", "content": (
        "You are a helpful assistant. Answer the question in the following format:\n"
        "<think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>."
    )},
    {"role": "user", "content": [
        {"type": "text", "text": "Is it safe to turn right?"},
        {"type": "video", "video": "assets/av_example.mp4", "fps": 4},
    ]},
]

# Processor and inputs
processor = AutoProcessor.from_pretrained(MODEL_PATH)
prompt = processor.apply_chat_template(
    video_messages,
    tokenize=False,
    add_generation_prompt=True,
)

image_inputs, video_inputs, video_kwargs = process_vision_info(video_messages, return_video_kwargs=True)


inputs = processor(
    text=[prompt],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(llm.device)


# Generate
generated_ids = llm.generate(
    **inputs,
    temperature=0.6,
    top_p=0.95,
    repetition_penalty=1.05,
    max_new_tokens=4096,
)

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)