# Learn LLM Serving
In this directory, we will build applications (clients) to access endpoints being served by engines like vLLM, TensorRT-LLM, TGI, MLX, Llama.cpp and more. 

![LLM Serving Engines](https://learnopencv.com/wp-content/uploads/2025/12/LLM-serving-engines-1.png)

Type of GPU Cores:
 - Turing (fp16, int8)
 - Ampere (fp16, bf16, int8)
 - Hooper (fp16, bf16, int8, fp8)
 - Blackwell (fp16, bf16, int8, fp8+)

Type of Kernels:
 - Merlin
 - Flash Attention
 - CUTLASS
 - TensorRT
 - Flashinfer
  
## Installation and Setup 

Install vllm, fastapi, uvicorn, and ngrok

```bash
pip install vllm fastapi uvicorn
```

```bash
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
  | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
  && echo "deb https://ngrok-agent.s3.amazonaws.com bookworm main" \
  | sudo tee /etc/apt/sources.list.d/ngrok.list \
  && sudo apt update \
  && sudo apt install ngrok
```

Add authentication.

```bash
ngrok config add-authtoken <token>
```

Terminal 1
```bash
uvicorn app.main:app --host 0.0.0.0 --port 3000
```

Terminal 2
```bash
vllm serve tiiuae/Falcon3-7B-Instruct-GPTQ-Int4 \
--port 8000 --max-model-len 4096 --gpu-memory-utilization 0.85
```

Terminal 3
Forward the FastAPI endpoint

```bash
ngrok http 3000
```
## Functional Models with Commands
The following are the list of models we have successfully tried so far on `vllm==0.12.x` versions. The errors we faced and fixes are also logged within. 

**1. Falcon3-7B-Instruct-GPTQ-Int4**
```bash
vllm serve tiiuae/Falcon3-7B-Instruct-GPTQ-Int4 --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.85
```
**2. Ministral-3-8B-Instruct-2512-AWQ-4bit**
```bash
vllm serve cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-4bit --port 8000 --gpu-memory-utilization 0.85 --max-model-len 6144 --max-num-batched-tokens 1024
```
**3. OpenGVLab/InternVL3-8B-AWQ**

Note that this AWQ quantized model won't work unless `--quantization awq` flag is explicitly set.

```bash
vllm serve OpenGVLab/InternVL3-8B-AWQ --max-model-len 4096 --gpu-memory-utilization 0.75 --max-num-batched-tokens 1024 --trust-remote-code --quantization awq
```

**4. OpenGVLab/InternVL3-2B**

```bash
vllm serve OpenGVLab/InternVL3-2B --max-model-len 4096 --gpu-memory-utilization 0.75 --max-num-batched-tokens 1024 --trust-remote-code
```

**5. Nemotron Cascade 8B**

```bash
vllm serve cyankiwi/Nemotron-Cascade-8B-AWQ-4bit --max-model-len 4096 --port 8000 --gpu-memory-utilization 0.85 --max-num-batched-tokens 1024 --trust-remote-code
```

**6. Nemotron Orchestrator 8B**

```bash
 vllm serve cyankiwi/Nemotron-Orchestrator-8B-AWQ-4bit --served-model-name Nemotron-orchestrator  --max-model-len 4096  --gpu-memory-utilization 0.85   --max-num-batched-tokens 1024 --trust-remote-code  --port 8000
```

**7. Qwen VL 2B Instruct**
```bash
vllm serve Qwen/Qwen2-VL-2B-Instruct --max-model-len 4096 --gpu-memory-utilization 0.75 --max-num-batched-tokens 1024 --port 8000
```

**8. Nvidia Cosmos Reason2 2B**
```bash
vllm serve nvidia/Cosmos-Reason2-2B --max-model-len 8192 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.8 --port 8000
```

**9. H20VL Mississipi 2B**
Does not support system prompt, hence need to take care of this. (Not yet fixed)

```bash
vllm serve h2oai/h2ovl-mississippi-2b   --max-model-len 4096  --max-num-batched-tokens 2048 --gpu-memory-utilization 0.75
```

**10. Gemma 3 4B Instriuct**
The original model is not loading due to OOM issue. Lowering model length to 1024 does not make sense for a Vision model. Hence using GPTQ quantized model from community. 

Max concurrency observed 9 Nos.
```bash
vllm serve ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g --max-model-len 4096 --max-num-batched-tokens 102
4 --gpu-memory-utilization 0.8
```