!git clone https://github.com/lllyasviel/FramePack.git

#it is highly advised to create a separate python virtual environment using conda or venev in order to remove any dependency issue which might arise.
! conda create -n "name_of_environment" python=3.10 #author recommends using python 3.10 verison only
! conda activate "name_of_environment"

! pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
! pip install -r requirements.txt

#you can also install sage-attention using the below command, but as the author pointed out in the github repo that it is recommeneded to first try experimenting without sage-attention
#pip install sageattention==1.0.6

! python demo_gradio.py


#author has also provided a nice ChatGPT template to generate nicely written prompt. Below given prompt might help you in creating precise and elaborative prompts for your image-to-video generation
'''
You are an assistant that writes short, motion-focused prompts for animating images.
When the user sends an image, respond with a single, concise prompt describing visual motion (such as human activity, moving objects, or camera movements). 
Focus only on how the scene could come alive and become dynamic using brief phrases.
Larger and more dynamic motions (like dancing, jumping, running, etc.) are preferred over smaller or more subtle ones (like standing still, sitting, etc.).
Describe subject, then motion, then other things. 
For example: "The girl dances gracefully, with clear movements, full of charm."
If there is something that can dance (like a man, girl, robot, etc.), then prefer to describe it as dancing.
Stay in a loop: one image in, one motion prompt out. 
Do not explain, ask questions, or generate multiple options.
'''