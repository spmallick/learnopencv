# [OpenCV 5.1 DNN on the GPU: When Does CUDA Actually Help?](https://learnopencv.com/opencv-5-1-dnn-cuda/)

[![OpenCV 5.1 DNN benchmark on the GTX 1080 Ti and NVIDIA L4](figures/opencv-5-1-dnn-cuda-featured.png)](https://learnopencv.com/opencv-5-1-dnn-cuda/)

This companion runs the same two ImageNet classifiers through OpenCV 5.1.0 on CPU and CUDA and through ONNX Runtime's CUDA execution provider. The [offline article](article.md) explains the GTX 1080 Ti and Colab L4 results.

[<img src="https://learnopencv.com/wp-content/uploads/2022/07/download-button-e1657285155454.png" alt="Download Code" width="200">](https://github.com/spmallick/learnopencv/releases/download/opencv-5-1-dnn-cuda-2026.11.23/OpenCV-5-1-DNN-CUDA.zip)

Download the [SHA-256 checksum](https://github.com/spmallick/learnopencv/releases/download/opencv-5-1-dnn-cuda-2026.11.23/OpenCV-5-1-DNN-CUDA.zip.sha256) beside the ZIP. Verify and extract it from the directory containing both files.

```bash
shasum -a 256 -c OpenCV-5-1-DNN-CUDA.zip.sha256
unzip OpenCV-5-1-DNN-CUDA.zip
cd OpenCV-5-1-DNN-CUDA
```

## Run the TF32 experiment in Colab

Open [the L4 notebook](tf32_colab.ipynb) in Google Colab, select **Runtime → Change runtime type → L4 GPU**, and choose **Run all**. It contains the companion code, builds OpenCV, exports the two fixed models, and measures three fresh pairs of TF32-off/on sessions per GPU backend. It saves a ZIP with timings, placement evidence and logits. Disconnect and delete the runtime after downloading the ZIP. L4 availability and compute-unit usage depend on your Colab account.

To repeat the L4 comparison with the exact ResNet-18 file from the GTX 1080 Ti benchmark, use its ONNX export and export report. In the directory containing the reference files, run `zip -0 reference-assets.zip resnet18-fixed.onnx resnet18-export.json` and upload the ZIP to `/content/reference-assets.zip` in Colab's Files pane before the export cell runs. The notebook verifies the model, its export report, the locally exported ViT and both photos before selecting the reference files. The resulting archive records both the local and selected ResNet hashes. Without an uploaded ZIP, the notebook uses its local exports and reports whether their hashes match the reference files.

## Prepare the two classifiers

Run these commands from this folder. Use a separate export environment so the model conversion tools cannot replace the inference runtime. The tested exporter uses Python 3.14 on macOS. On Linux, Python 3.11 or 3.12 with the CPU PyTorch wheels is a lighter install.

```bash
python3 -m venv .venv-export
source .venv-export/bin/activate
# Linux x86-64 only: install CPU PyTorch wheels before the remaining dependencies.
# python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 torchvision==0.25.0
python -m pip install -r requirements-export.txt
python prepare_assets.py --output-dir assets
deactivate
```

The script downloads the original torchvision ViT-B/16 and ResNet-18 IMAGENET1K_V1 checkpoints, verifies their SHA-256 hashes, and writes fixed-batch and symbolic-batch ONNX files, 1,000 labels, two photographs and an export report per model. It checks logits against eager PyTorch before accepting each export. Allow several GB for the dependencies, models, temporary exports and weight cache. Copy the entire `assets/` directory to the benchmark machine if you exported elsewhere. [ViT-B/16 weights](https://docs.pytorch.org/vision/0.25/models/generated/torchvision.models.vit_b_16.html) · [ResNet-18 weights](https://docs.pytorch.org/vision/0.25/models/generated/torchvision.models.resnet18.html)

## Build a CUDA-enabled Python runtime

The tested setup is x86-64 Ubuntu 24.04, Python 3.12, a CUDA 12.9 development toolkit with `nvcc`, GCC 13 and an NVIDIA driver. The official OpenCV Python wheels are CPU-only. For the GTX 1080 Ti, we compiled native code for compute capability 6.1 and used the same OpenCV build for the CPU baseline. The script pins the OpenCV and opencv_contrib sources, cuDNN 9.10.2.21 and ONNX Runtime GPU 1.22.0. It installs into a project-local directory and generates an activation script for the library search path. [OpenCV wheel documentation](https://pypi.org/project/opencv-python/5.1.0.95/) · [cuDNN 9.10.2 support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.10.2/reference/support-matrix.html)

```bash
# Ubuntu build dependencies, in addition to the driver and CUDA 12.x toolkit
sudo apt-get install build-essential cmake ninja-build git python3-dev python3-venv \
    libjpeg-dev libpng-dev liblapacke-dev liblapack-dev
nvidia-smi --query-gpu=index,name,compute_cap,uuid --format=csv
nvcc --version
CUDA_ARCH=61 ./build_cuda.sh
source .opencv51-cuda/activate.sh
```

Choose the architecture of your card, e.g. `CUDA_ARCH=89` for an L4. The architecture and driver must work with the installed CUDA toolkit. `CUDA_ROOT`, `BUILD_ROOT` and `BUILD_JOBS` override the toolkit directory, build directory and parallel build jobs. Source `activate.sh` from the chosen build root in each new shell. On a multi-GPU host, export `CUDA_VISIBLE_DEVICES` with the desired GPU's UUID before starting the benchmark.

## Classify and benchmark

For a CPU-only first run, install `opencv-python==5.1.0.95` in a fresh Python environment and use `--backend opencv-cpu` below. We tested this route with both photographs. To reproduce the CPU-to-CUDA comparison, use the same CUDA-enabled OpenCV build for both targets as described above.

```bash
python benchmark.py --model assets/vit_b_16-fixed.onnx \
    --image assets/chelsea.png --image assets/coffee.png --labels assets/labels.txt \
    --backend opencv-cuda --output runs/vit-cuda
```

Change `--backend` to `opencv-cpu` or `ort-cuda` and give each run a fresh output directory. `--warmup 10 --runs 50 --threads 6 --tf32 off` are the defaults. Each GPU run requires positive placement evidence before it produces `result.json`. The result includes p50, p95, every sample, the model hash, the predictions, peak process RSS and retained worker GPU memory where visible. `logits.npz` contains all scores. GPU proof logs and the ORT profiling JSON stay beside the result.

For a symbolic batch, run `--model assets/vit_b_16-dynamic.onnx --batches 1,2`. Use `--batches 1` and `--batches 2` for held-size controls on the same file. Use `--tf32 on` in a fresh CUDA session for the precision comparison. For your own model, supply its file, photos and labels. The example expects one NCHW FP32 input, ImageNet normalization, a 224 × 224 center crop and one output shaped `[batch, classes]`. Adapt `classifier.py` when the training recipe or output contract differs.

```bash
python -m unittest discover -s . -p test_benchmark.py -v
python run_matrix.py --assets assets --output runs/matrix
python summarize.py runs/matrix --output runs/tables
```

The matrix runs one process at a time and rotates backend order across three fixed-batch sessions. It then runs the shape and precision controls. For a shorter functional check, use `--repeats 1 --warmup 2 --runs 4`. Each run needs an empty output directory. The summary checks every saved logit vector against the same photograph in the fixed CPU reference and reports alternating batches separately. From another working directory, use absolute script, input and output paths.

## Check operator placement and graph replay

```bash
OPENCV_LOG_LEVEL=INFO python inspect_cuda.py --fallback-demo
OPENCV_LOG_LEVEL=INFO python inspect_cuda.py --model assets/vit_b_16-fixed.onnx
cuda_root=/usr/local/cuda
cc -std=c11 -O2 -fPIC -shared -I"$cuda_root/include" cuda_api_trace.c \
    -L"$cuda_root/lib64" -Wl,-rpath,"$cuda_root/lib64" -lcupti -o /tmp/libcuda_api_trace.so
OPENCV_LOG_LEVEL=INFO python inspect_cuda.py --trace-api /tmp/libcuda_api_trace.so \
    --model assets/vit_b_16-fixed.onnx
```

The first command creates a tiny network containing TopK and checks its output numerically. The model probe runs five forwards on changing pixels. The optional probe counts graph captures, launches, device allocations and frees. It also counts the bytes that actually crossed from host to GPU, including copies whose direction CUDA inferred. Compile it with the CUDA 12.9 development toolkit and its CUPTI headers and library. To probe changing shapes, substitute the dynamic model and add `--batches 1 2`. Run the latency benchmark in a separate process without the callback library. [NVIDIA CUPTI callback API](https://docs.nvidia.com/cupti/12.9.1/main/main.html#cupti-callback-api)

The measured samples, session summaries and five-forward launch counts are in [measurements](measurements/). To redraw the batch-one chart, run `python figures/plot_timings.py` in an environment with Matplotlib (tested with 3.10.9). Pass `--sessions runs/tables/sessions.csv --hardware "Your CPU and GPU" --output runs/tables/fixed-batch-latency.png` to chart your own data. Run `python figures/plot_previous_batch.py` to reproduce the transfer and allocation figure from the timing and CUPTI measurements.

The scikit-image photographs are CC0. Stefan van der Walt took the photograph of Chelsea. Rachel Michetti took the coffee photograph. [Image credits](https://scikit-image.org/docs/0.25.x/api/skimage.data.html)

## Project files

The standalone ZIP contains these 33 files. Running the exporter creates the ONNX models and sample inputs in a separate `assets/` folder.

```text
OpenCV-5-1-DNN-CUDA/
├── figures/
│   ├── fixed-batch-latency.png
│   ├── opencv-5-1-dnn-cuda-featured.png
│   ├── plot_previous_batch.py
│   ├── plot_timings.py
│   └── previous-batch-cost.png
├── measurements/
│   ├── README.md
│   ├── cuda-api-calls.csv
│   ├── exports.json
│   ├── host-activity.csv
│   ├── l4-environment.json
│   ├── l4-paired.csv
│   ├── l4-samples.csv
│   ├── l4-sessions.csv
│   ├── l4-summary.csv
│   ├── samples.csv
│   ├── sensitivity-host-activity.csv
│   ├── sensitivity-session-load.csv
│   ├── sensitivity-sessions.csv
│   └── sessions.csv
├── .gitignore
├── README.md
├── article.md
├── benchmark.py
├── build_cuda.sh
├── classifier.py
├── cuda_api_trace.c
├── inspect_cuda.py
├── prepare_assets.py
├── requirements-export.txt
├── run_matrix.py
├── summarize.py
├── test_benchmark.py
└── tf32_colab.ipynb
```
