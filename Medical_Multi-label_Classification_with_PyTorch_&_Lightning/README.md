# Medical Multi-label Classification with PyTorch & Lightning

<img src="https://learnopencv.com/wp-content/uploads/2023/07/medical_multi-label_feature.gif">

This repository contains the files related to the LearnOpenCV blog post: [Medical Multi-label Classification with PyTorch & Lightning](https://learnopencv.com/medical-multi-label/)

<img src="https://learnopencv.com/wp-content/uploads/2023/07/medical_multi-label_per_class_image.png">

This project was created using the subset of <a href="https://www.kaggle.com/competitions/human-protein-atlas-image-classification/overview" target="_blank">Human Protein Atlas Image Classification</a> challenge dataset.

The subset dataset includes the top 10 classes of the above dataset and was used as an in-class competition here:  <a href="https://www.kaggle.com/competitions/jovian-pytorch-z2g/overview" target="_blank">Zero to GANs - Human Protein Classification</a>.

As the competition is ended, the subset dataset can be downloaded from over here:

1. <a href="https://www.kaggle.com/datasets/aakashns/jovian-pytorch-z2g" target="_blank">Subset - 512x512 size</a>
2. <a href="https://www.kaggle.com/datasets/learnopencvblog/human-protein-atlas-384x384" target="_blank">Subset - 384x384 size (used for the project)</a>

We used EfficientNetV2-Small model from Torchvision for training and deployed the app using Gradio onto HuggingFace spaces.

You can use the demo app here --> <a href="https://huggingface.co/spaces/veb-101/Medical_MultiLabel_Image_Classification" target="_blank">Medical Multi-Label Image Classification Gradio App</a>

You can also check/access and download the following files:

1. `app.py` python file.
2. Trained lightning checkpoint file.
3. `requirements.txt` for the gradio app deployment on HuggingFace

From here --> <a href="https://huggingface.co/spaces/veb-101/Medical_MultiLabel_Image_Classification/tree/main" target="_blank">Gradio App Files</a>

**You can run it on your local system or on Colab.** [![Click to open notebook on Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/spmallick/learnopencv/blob/master/Medical_Multi-label_Classification_with_PyTorch_&_Lightning/HPA_PT_lightning_Colab.ipynb)

---

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="download" width="200">](https://www.dropbox.com/sh/4qa0vsoazxjbw2z/AABeaYqdMlWCgWPYIvsYg1QEa?dl=1)

---

## AI Courses by OpenCV

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/university/) is a great place to start.

[![img](https://learnopencv.com/wp-content/uploads/2023/01/AI-Courses-By-OpenCV-Github.png)](https://opencv.org/university/)
