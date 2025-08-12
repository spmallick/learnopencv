from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration, WhisperProcessor, CLIPProcessor, CLIPModel
import copy
from decord import VideoReader, cpu
import numpy as np
import json
from tqdm import tqdm
import os
import easyocr
from tools.rag_retriever_dynamic import retrieve_documents_with_dynamic
import re
import ast
import socket
import pickle
from tools.filter_keywords import filter_keywords
from tools.scene_graph import generate_scene_graph_description
import ffmpeg, torchaudio

max_frames_num = 32
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336", torch_dtype=torch.float16, device_map="auto")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
# whisper_model = WhisperForConditionalGeneration.from_pretrained(
#     "openai/whisper-large",
#     torch_dtype=torch.float16,
#     device_map="auto"
# )
# whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large")

def process_video(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()

    return spare_frames, frame_time, video_time

def extract_audio(video_path, audio_path):
    if not os.path.exists(audio_path):
        ffmpeg.input(video_path).output(audio_path, acodec='pcm_s16le', ac=1, ar='16k').run()

def chunk_audio(audio_path, chunk_length_s=30):
    speech, sr = torchaudio.load(audio_path)
    speech = speech.mean(dim=0)  
    speech = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(speech)  
    num_samples_per_chunk = chunk_length_s * 16000 
    chunks = []
    for i in range(0, len(speech), num_samples_per_chunk):
        chunks.append(speech[i:i + num_samples_per_chunk])
    return chunks

# def transcribe_chunk(chunk):

#     inputs = whisper_processor(chunk, return_tensors="pt")
#     inputs["input_features"] = inputs["input_features"].to(whisper_model.device, torch.float16)
#     with torch.no_grad():
#         predicted_ids = whisper_model.generate(
#             inputs["input_features"],
#             no_repeat_ngram_size=2,
#             early_stopping=True
#         )
#     transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
#     return transcription

# def get_asr_docs(video_path, audio_path):

#     full_transcription = []

#     try:
#         extract_audio(video_path, audio_path)
#     except:
#         return full_transcription
#     audio_chunks = chunk_audio(audio_path, chunk_length_s=30)
    
#     for chunk in audio_chunks:
#         transcription = transcribe_chunk(chunk)
#         full_transcription.append(transcription)

#     return full_transcription

def get_ocr_docs(frames):
    reader = easyocr.Reader(['en']) 
    text_set = []
    ocr_docs = []
    for img in frames:
        ocr_results = reader.readtext(img)
        det_info = ""
        for result in ocr_results:
            text = result[1]
            confidence = result[2]
            if confidence > 0.5 and text not in text_set:
                det_info += f"{text}; "
                text_set.append(text)
        if len(det_info) > 0:
            ocr_docs.append(det_info)

    return ocr_docs

    
def save_frames(frames):
    file_paths = []
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        file_path = f'restore/frame_{i}.png'
        img.save(file_path)
        file_paths.append(file_path)
    return file_paths
    
def get_det_docs(frames, prompt):
    prompt = ",".join(prompt)
    frames_path = save_frames(frames)
    res = []
    if len(frames) > 0:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('0.0.0.0', 9999))
        data = (frames_path, prompt)
        client_socket.send(pickle.dumps(data))
        result_data = client_socket.recv(4096)
        try:
            res = pickle.loads(result_data)
        except:
            res = []
    return res

def det_preprocess(det_docs, location, relation, number):

    scene_descriptions = []

    for det_doc_per_frame in det_docs:
        objects = []
        scene_description = ""
        if len(det_doc_per_frame) > 0:
            for obj_id, objs in enumerate(det_doc_per_frame.split(";")):
                obj_name = objs.split(":")[0].strip()
                obj_bbox = objs.split(":")[1].strip()
                obj_bbox = ast.literal_eval(obj_bbox)
                objects.append({"id": obj_id, "label": obj_name, "bbox": obj_bbox})

            scene_description = generate_scene_graph_description(objects, location, relation, number)
        scene_descriptions.append(scene_description)
    
    return scene_descriptions


# load your VLM
device = "cuda"
overwrite_config = {}
tokenizer, model, image_processor, max_length = load_pretrained_model(
    "lmms-lab/LLaVA-Video-7B-Qwen2", 
    None, 
    "llava_qwen", 
    torch_dtype="bfloat16", 
    device_map="auto", 
    offload_buffers=True,
    overwrite_config=overwrite_config)  # Add any other thing you want to pass in llava_model_args


# 2) Check vocab sizes and fix BEFORE dispatching
vsz_model = model.get_input_embeddings().weight.shape[0]
vsz_tok   = len(tokenizer)
if vsz_tok != vsz_model:
    print(f"[fix] resizing embeddings: model={vsz_model} -> tokenizer={vsz_tok}")
    model.resize_token_embeddings(vsz_tok)
    # optional: init new rows
    with torch.no_grad():
        added = vsz_tok - vsz_model
        if added > 0:
            emb = model.get_input_embeddings().weight
            emb[-added:].normal_(mean=0.0, std=0.02)



# from accelerate import init_empty_weights
# model = model.to("cuda") 
# from accelerate import infer_auto_device_map, dispatch_model
# device_map = "auto"  # or infer_auto_device_map(model, ...)
# model = dispatch_model(model, device_map=device_map, offload_buffers=True)   

model.eval()
conv_template = "qwen_2"  # Make sure you use correct chat template for different models
            

# The inference function of your VLM
def llava_inference(qs, video):
    if video is not None:
        question = DEFAULT_IMAGE_TOKEN + qs
    else:
        question = qs
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    # input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    # cont = model.generate(
    #     input_ids,
    #     video=video,
    #     modalities= ["video"],
    #     do_sample=True,
    #     temperature=0.7,
    #     max_new_tokens=4096,
    # )

    if video is not None:
        cont = model.generate(
            input_ids,
            images=video,
            modalities=["video"],
            do_sample=True,
            temperature=0.7,
            max_new_tokens=512
        )
    else:
        cont = model.generate(
            input_ids,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=512
        )
        
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    return text_outputs


# super-parameters setting
rag_threshold = 0.3
clip_threshold = 0.3
beta = 3.0 

# Choose the auxiliary texts you want
USE_OCR = True
USE_ASR = False 
USE_DET = True
print(f"---------------OCR{rag_threshold}: {USE_OCR}-----------------")
print(f"---------------ASR{rag_threshold}: {USE_ASR}-----------------")
print(f"---------------DET{beta}-{clip_threshold}: {USE_DET}-----------------")
print(f"---------------Frames: {max_frames_num}-----------------")


video_path = "/workspace/LLaVA-NeXT/Chest X-Ray Report Generation and VQA.mp4"  # your video path
question = "Can you summarize what is happening in the video?"  # your question


frames, frame_time, video_time = process_video(video_path, max_frames_num, 1, force_sample=True)
raw_video = [f for f in frames]

video = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].cuda().bfloat16()
video = [video]

if USE_DET:
    video_tensor = []
    for frame in raw_video:
        processed = clip_processor(images=frame, return_tensors="pt")["pixel_values"].to(clip_model.device, dtype=torch.float16)
        video_tensor.append(processed.squeeze(0))
    video_tensor = torch.stack(video_tensor, dim=0)

if USE_OCR:
    ocr_docs_total = get_ocr_docs(frames)

if USE_ASR:
    if os.path.exists(os.path.join("restore/audio", os.path.basename(video_path).split(".")[0] + ".txt")):
        with open(os.path.join("restore/audio", os.path.basename(video_path).split(".")[0] + ".txt"), 'r', encoding='utf-8') as f:
            asr_docs_total = f.readlines()
    # else:
    #     audio_path = os.path.join("restore/audio", os.path.basename(video_path).split(".")[0] + ".wav")
    #     # asr_docs_total = get_asr_docs(video_path, audio_path)
    #     with open(os.path.join("restore/audio", os.path.basename(video_path).split(".")[0] + ".txt"), 'w', encoding='utf-8') as f:
    #         for doc in asr_docs_total:
    #             f.write(doc + '\n')

# step 0: get cot information
retrieve_pmt_0 = "Question: " + question
# you can change this decouple prompt to fit your requirements
retrieve_pmt_0 += "\nTo answer the question step by step, you can provide your retrieve request to assist you by the following json format:"
retrieve_pmt_0 += '''{
    "ASR": Optional[str]. The subtitles of the video that may relavent to the question you want to retrieve, in two sentences. If you no need for this information, please return null.
    "DET": Optional[list]. (The output must include only physical entities, not abstract concepts, less than five entities) All the physical entities and their location related to the question you want to retrieve, not abstract concepts. If you no need for this information, please return null.
    "TYPE": Optional[list]. (The output must be specified as null or a list containing only one or more of the following strings: 'location', 'number', 'relation'. No other values are valid for this field) The information you want to obtain about the detected objects. If you need the object location in the video frame, output "location"; if you need the number of specific object, output "number"; if you need the positional relationship between objects, output "relation". 
}
## Example 1: 
Question: How many blue balloons are over the long table in the middle of the room at the end of this video? A. 1. B. 2. C. 3. D. 4.
Your retrieve can be:
{
    "ASR": "The location and the color of balloons, the number of the blue balloons.",
    "DET": ["blue ballons", "long table"],
    "TYPE": ["relation", "number"]
}
## Example 2: 
Question: In the lower left corner of the video, what color is the woman wearing on the right side of the man in black clothes? A. Blue. B. White. C. Red. D. Yellow.
Your retrieve can be:
{
    "ASR": null,
    "DET": ["the man in black", "woman"],
    "TYPE": ["location", "relation"]
}
## Example 3: 
Question: In which country is the comedy featured in the video recognized worldwide? A. China. B. UK. C. Germany. D. United States.
Your retrieve can be:
{
    "ASR": "The country recognized worldwide for its comedy.",
    "DET": null,
    "TYPE": null
}
Note that you don't need to answer the question in this step, so you don't need any infomation about the video of image. You only need to provide your retrieve request (it's optional), and I will help you retrieve the infomation you want. Please provide the json format.'''

json_request = llava_inference(retrieve_pmt_0, None)

# step 1: get docs information
query = [question]

# APE fetch
if USE_DET:
    det_docs = []
    try:
        request_det = json.loads(json_request)["DET"]
        request_det = filter_keywords(request_det)
        clip_text = ["A picture of " + txt for txt in request_det]
        if len(clip_text) == 0:
            clip_text = ["A picture of object"]
    except:
        request_det = None
        clip_text = ["A picture of object"]

    clip_inputs = clip_processor(text=clip_text, return_tensors="pt", padding=True, truncation=True).to(clip_model.device)
    clip_img_feats = clip_model.get_image_features(video_tensor)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**clip_inputs)
        similarities = (clip_img_feats @ text_features.T).squeeze(0).mean(1).cpu()
        similarities = np.array(similarities, dtype=np.float64)
        alpha = beta * (len(similarities) / 16)
        similarities = similarities * alpha / np.sum(similarities)

    del clip_inputs, clip_img_feats, text_features
    torch.cuda.empty_cache()

    det_top_idx = [idx for idx in range(max_frames_num) if similarities[idx] > clip_threshold]
        
    if request_det is not None and len(request_det) > 0:
        det_docs = get_det_docs(frames[det_top_idx], request_det)  

        L, R, N = False, False, False
        try:
            det_retrieve_info = json.loads(json_request)["TYPE"]
        except:
            det_retrieve_info = None
        if det_retrieve_info is not None:
            if "location" in det_retrieve_info:
                L = True
            if "relation" in det_retrieve_info:
                R = True
            if "number" in det_retrieve_info:
                N = True
        det_docs = det_preprocess(det_docs, location=L, relation=R, number=N)  # pre-process of APE information


# OCR fetch
if USE_OCR:
    try:
        request_det = json.loads(json_request)["DET"]
        request_det = filter_keywords(request_det)
    except:
        request_det = None
    ocr_docs = []
    if len(ocr_docs_total) > 0:
        ocr_query = query.copy()
        if request_det is not None and len(request_det) > 0:
            ocr_query.extend(request_det)
        ocr_docs, _ = retrieve_documents_with_dynamic(ocr_docs_total, ocr_query, threshold=rag_threshold)

# ASR fetch
if USE_ASR:
    asr_docs = []
    try:
        request_asr = json.loads(json_request)["ASR"]
    except:
        request_asr = None
    if len(asr_docs_total) > 0:
        asr_query = query.copy()
        if request_asr is not None:
            asr_query.append(request_asr)
        asr_docs, _ = retrieve_documents_with_dynamic(asr_docs_total, asr_query, threshold=rag_threshold)

qs = ""
if USE_DET and len(det_docs) > 0:
    for i, info in enumerate(det_docs):
        if len(info) > 0:
            qs += f"Frame {str(det_top_idx[i]+1)}: " + info + "\n"
    if len(qs) > 0:
        qs = f"\nVideo have {str(max_frames_num)} frames in total, the detected objects' information in specific frames: " + qs
if USE_ASR and len(asr_docs) > 0:
    qs += "\nVideo Automatic Speech Recognition information (given in chronological order of the video): " + " ".join(asr_docs)
if USE_OCR and len(ocr_docs) > 0:
    qs += "\nVideo OCR information (given in chronological order of the video): " + "; ".join(ocr_docs)
qs += "Select the best answer to the following multiple-choice question based on the video and the information (if given). Respond with only the letter (A, B, C, or D) of the correct option. Question: " + question  # you can change this prompt

res = llava_inference(qs, video)
print(res)
