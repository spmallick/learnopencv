# Fine-Tuning Grounding DINO: Open-Vocabulary Object Detection

This repository contains the scripts used for fine-tuning Grounding DINO on the Face Mask Detection Dataset. `fine_tune_grounding_dino.py` runs training, `test_fine_tuned_grounding_dino.py` runs inference with a fine-tuned checkpoint, `create_annotations_final.py` converts the dataset annotations into the CSV format expected by the trainer, and `grounding_dino_finetune_utils.py` provides the local training helper that works with the current public GroundingDINO API.


It is part of the LearnOpenCV blog post - [Fine-Tuning-Grounding-DINO-Open-Vocabulary-Object-Detection](https://learnopencv.com/fine-tuning-grounding-dino/).

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="Download Code" width="200">](https://www.dropbox.com/scl/fo/umthzd0t9gq33cc3tfmza/APprHLYeh2AqPMYl-70TsfI?rlkey=42xfz52ejgrgk40jiohmtwoh4&st=wuwkwfyr&dl=1)

![](https://learnopencv.com/wp-content/uploads/2025/06/Fine-Tuning-Grounding-DINO-Open-Vocabulary-Object-Detection.png)

## Setup

The training helper in this folder works with the public GroundingDINO repository, but the setup is sensitive to dependency drift. The validated path is Python 3.12 with a pinned 4.x `transformers` release.

Install the public GroundingDINO package first:

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -r requirements.txt
pip install "transformers<4.40"
pip install --no-build-isolation -e .
mkdir -p weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O weights/groundingdino_swint_ogc.pth
```

Then return to this folder, prepare the dataset split, and train with relative paths instead of machine-specific paths:

```bash
python create_annotations_final.py

python fine_tune_grounding_dino.py \
  --image-dir train/images \
  --annotations train/annotations_final.csv \
  --save-dir weights_less
```

To test a fine-tuned checkpoint:

```bash
python test_fine_tuned_grounding_dino.py \
  --weights weights_less/model_weights_epoch_50.pth \
  --image test/images/maksssksksss774.png \
  --output result.jpg
```

Note: the original tutorial referenced `groundingdino.util.train`, but that helper is not part of the public GroundingDINO package. This repository now ships a local training utility so the example remains runnable with the current upstream API. The above setup was smoke-tested against the public GroundingDINO repo on Python 3.12 with GPU training and inference.


## AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start.

[![img](https://learnopencv.com/wp-content/uploads/2023/01/AI-Courses-By-OpenCV-Github.png)](https://opencv.org/courses/)
