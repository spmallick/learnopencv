import torch
import time
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from transformers import AutoProcessor

# Load model (replace with your checkpoint if needed)
policy = SmolVLAPolicy.from_pretrained(
    "/home/opencvuniv/lerobot/outputs/train/2025-06-05/18-49-20_smolvla/checkpoints/last/pretrained_model"
).to("cuda")
policy.eval()

# Monkey-patch: The loaded policy is missing the language_tokenizer attribute.
policy.language_tokenizer = AutoProcessor.from_pretrained(policy.config.vlm_model_name).tokenizer

# Dummy batch config for a single observation
batch_size = 1
img_shape = (3, 512, 512)  # (C, H, W)
# Infer state_dim from the loaded normalization stats
state_dim = policy.normalize_inputs.buffer_observation_state.mean.shape[-1]

dummy_batch = {
    # a single image observation
    "observation.images.top": torch.rand(batch_size, *img_shape, device="cuda"),
    # a single state observation
    "observation.state": torch.rand(batch_size, state_dim, device="cuda"),
    "task": ["stack the blocks"] * batch_size,
}

# --- Prepare inputs for the model ---
# The policy expects normalized inputs and specific data preparation.
normalized_batch = policy.normalize_inputs(dummy_batch)
images, img_masks = policy.prepare_images(normalized_batch)
state = policy.prepare_state(normalized_batch)
lang_tokens, lang_masks = policy.prepare_language(normalized_batch)
# ---

# Warmup
for _ in range(3):
    with torch.no_grad():
        _ = policy.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)

# Benchmark
torch.cuda.reset_peak_memory_stats()
start = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = policy.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state)
end = time.time()

print(f"Avg inference time: {(end - start)/100:.6f} s")
print(f"Max GPU memory used: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
