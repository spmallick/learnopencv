# MiniCPM-o 4.5: Full-Duplex Real-Time Video Understanding

**This repository contains the notebook accompanying the LearnOpenCV blog post _MiniCPM-o 4.5: A 9B Model That Can See, Hear, and Speak at the Same Time_.**

MiniCPM-o 4.5 is a 9B omni-modal model capable of processing video and audio streams while generating speech in real time. This repository contains the notebook and supporting code used to evaluate its full-duplex streaming capabilities across a variety of real-world scenarios.

Rather than benchmarking the model, these experiments illustrate how it behaves under different prompting strategies and streaming inputs, complementing the discussion presented in the accompanying LearnOpenCV article.

---

# Experiments

The notebook includes the following streaming experiments:

### 1. Live Sports Commentary

The model continuously watches a football penalty clip and generates real-time commentary, speaking only when notable events occur.

### 2. Behavioral Narration

The model observes a retail surveillance video and describes the ongoing actions. The experiment evaluates whether prolonged behavioural patterns can be identified from visual observations alone.

### 3. Cooking Narration

The model follows a cooking video using both the visual stream and the accompanying narration, generating synchronized spoken descriptions of the preparation process.

### 4. Real-Time Question Answering

Pre-recorded questions are injected into the audio stream at specific timestamps. The model detects these questions during streaming and answers them using the visual and audio context available up to that point.

### 5. Voice Cloning

The model performs zero-shot voice cloning by conditioning speech generation on different reference voice clips.

---

# Streaming Pipeline

Across all experiments, the notebook follows the same streaming pipeline.

| Step | Function | Description |
|------|----------|-------------|
| 1 | `strip_video_audio` | Creates a silent copy of the source video that will later be combined with the model's generated speech. |
| 2 | `get_video_frame_audio_segments` | Splits the source video into one-second video frames and audio segments. |
| 3 | `build_audio_chunks_qa` | (Q&A only) Injects pre-recorded question audio into the streaming audio at specified timestamps. |
| 4 | `run_mode` | Streams video and audio chunks through MiniCPM-o 4.5 using its full-duplex interface. |
| 5 | `generate_duplex_video` | Combines the generated speech with the silent video to produce the final output. |
| 6 | `overlay_user_questions` | (Optional) Overlays injected question text onto the output video for visualization. |

The original video audio is used only as input to the model. The final rendered video contains only the model's generated speech, making it easier to evaluate the model's responses independently of the original narration.

---

# Model

| Component | Model |
|-----------|-------|
| Full-Duplex Vision, Audio and Speech | [`openbmb/MiniCPM-o-4_5`](https://huggingface.co/openbmb/MiniCPM-o-4_5) |

The notebook loads the model entirely from a local Hugging Face snapshot using offline mode.

---

# Repository Contents

```text
minicpm_o_full_duplex_video_qa.ipynb     # Main notebook containing all experiments
requirements.txt                         # Python dependencies
```

---

# Expected Assets

Media files are not included in the repository.

Place your own assets inside an `assets/` directory.

```text
assets/
├── football.mp4          # Sports commentary video
├── theft.mp4             # Retail 
├── cooking.mp4           # Cooking video
├── vid.mp4               # Voice cloning video
├── audio1.wav            # Voice cloning reference 1
├── audio2.wav            # Voice cloning reference 2
├── q1.m4a                # Question 1
└── q2.m4a                # Question 2
```

Generated videos are written to:

```text
new_outputs/
```

---

# Requirements

- NVIDIA GPU with CUDA support 
- Python 3.10
- ffmpeg
- Conda (or virtual environment)
- Jupyter Notebook / JupyterLab

The experiments presented in the accompanying article were run in **bfloat16** on GPU.

---

# System Specs

The experiments in this repository were run on the following machine:

| Component | Spec |
|-----------|------|
| GPU | NVIDIA GeForce RTX 5090 (32 GB VRAM) |
| GPU Driver | 595.71.05 |
| CPU Cores | 24 |
| RAM | 125 GB |
| OS | Ubuntu 24.04.4 LTS |
| Kernel | 7.0.0-28-generic |
| Python | 3.10 (conda environment) |

---

# Setup

## 1. Create the environment

```bash
conda create -n minicpmo python=3.10 -y
conda run -n minicpmo pip install -r requirements.txt
conda install -n minicpmo -c conda-forge ffmpeg -y
conda run -n minicpmo python -m ipykernel install --user --name minicpmo --display-name "minicpmo"
```

---

## 2. Download the model

```bash
huggingface-cli download openbmb/MiniCPM-o-4_5
```

Update `LOCAL_PATH` in the notebook (or `qa.py`) to point to the downloaded snapshot.

The code runs entirely in offline mode using:

```python
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

---

## 3. Run the notebook

```bash
jupyter lab minicpm_o_full_duplex_video_qa.ipynb
```

or execute the script directly

```bash
conda run -n minicpmo python qa.py
```

---

# References

- [MiniCPM-o 4.5 Technical Report](https://arxiv.org/abs/2604.27393)
- [MiniCPM-o 4.5 Hugging Face](https://huggingface.co/openbmb/MiniCPM-o-4_5)
- [MiniCPM-o GitHub](https://github.com/OpenBMB/MiniCPM-o)
- [OpenCV](https://opencv.org/)

---

# AI Courses by OpenCV

Want to become an expert in AI?

Explore **AI Courses by OpenCV**:

https://opencv.org/courses/