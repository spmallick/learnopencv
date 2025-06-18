from transformers import AutoVideoProcessor, AutoModel

hf_repo = "facebook/vjepa2-vitl-fpc64-256"

model = AutoModel.from_pretrained(hf_repo)
processor = AutoVideoProcessor.from_pretrained(hf_repo)


import torch
from torchcodec.decoders import VideoDecoder
import numpy as np

video_url = "input_video_path"
vr = VideoDecoder(video_url)
frame_idx = np.arange(0, 64) # choosing some frames. here, you can define more complex sampling strategy
video = vr.get_frames_at(indices=frame_idx).data  # T x C x H x W
video = processor(video, return_tensors="pt").to(model.device)
with torch.no_grad():
    video_embeddings = model.get_vision_features(**video)

print(video_embeddings.shape)
# output shape for video sample(shown above) we used is:  torch.Size([1, 8192, 1024])

