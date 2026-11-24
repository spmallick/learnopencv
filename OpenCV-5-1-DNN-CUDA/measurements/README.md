# Benchmark evidence

## GeForce GTX 1080 Ti and Intel Core i7-6850K

The PNG-to-label measurements start at file decoding and end after class ranking and label lookup. The input files are reread from a warm OS cache. Forward-only samples start with an existing FP32 tensor in host memory and end when logits are available in host memory. A separate timer in a fresh process covers model or session construction and the first host-visible result. Python imports, label loading and preparing the first tensor take place before this cold-session timer.

- `sessions.csv` contains one row per session and observed batch size. The cold columns describe the first request only. Peak RSS covers the host process. GPU MiB is the worker's retained device memory queried with `nvidia-smi` after the run. Empty GPU-memory fields indicate CPU-only or unavailable readings.
- `samples.csv` preserves each observed latency in order within its phase. At alternating batch sizes, each of the two sizes contributes 25 samples per phase.
- `cuda-api-calls.csv` records five sequential requests in each of seven fresh probe processes, including the first forward. Each row spans `setInput()` and the host-visible return from `forward()`. The probe varied the first input channel on each request and recorded the first output score.
- `host-activity.csv` contains 60 samples at roughly five-second intervals during the main timing matrix. CPU usage is in logical-core equivalents, summed over the host and separately over Docker containers. The separate known CPU job contributes a lower bound from processes observed in consecutive samples. GPU utilization, watts, clocks, temperature and active compute-process counts are point samples on the timing card. No other compute process appeared in these snapshots. The API probes used a second GTX 1080 Ti.
- `exports.json` records the weight and graph hashes and largest discrepancy in the independent CPU export check.

### Repeating the matrix under different CPU load

We repeated all 40 sessions with the same model files, settings, six CPU threads and GPU. A CI build began during the repeat. All 98 saved rows still agreed with their CPU reference on the winning class and the five highest-ranked classes. The [repeat session summaries](sensitivity-sessions.csv) use the same columns as the main matrix. The repeat's 65 [host and GPU samples](sensitivity-host-activity.csv) show no competing compute PID on the measured GPU. Linux cgroup counters let us subtract the benchmark's CPU time from total host activity. The [per-session load estimates](sensitivity-session-load.csv) use the sampling intervals that overlap the timing worker. In the three fixed ViT CPU sessions, other processes used about 1.03, 1.17 and 7.43 logical cores on average.

## Reading the CUDA probe

| CSV fields | What the probe measures |
| --- | --- |
| `captures`, `graph_launches` | Successful CUDA runtime calls to begin stream capture or launch a graph. The capture forward also launches the new graph. |
| `runtime_kernel_launches` | Successful host submissions through runtime calls beginning with `cudaLaunchKernel`. A graph can execute many device kernels through one graph launch. This counter omits lower-level driver launches. |
| `h2d_copies`, `h2d_bytes` | Host-to-device transfers reported by CUPTI memcpy activity records, including transfers issued with `cudaMemcpyDefault`. The probe flushes completed records at each forward boundary and rejects dropped records. |
| `malloc_calls`, `malloc_bytes`, `free_calls` | Successful `cudaMalloc` requests, their summed requested bytes, and successful `cudaFree` calls during the request. A byte allocated, freed and allocated again contributes twice to the sum. Retained GPU memory is measured separately in `sessions.csv`. |

For symbolic ViT-B/16 at batch one, repeating the shape transferred 602,520 bytes in 14 copies. Returning to batch one after batch two transferred 346,264,888 bytes in 164 copies and issued 150 allocations totaling 345,662,368 bytes. For ResNet-18, the corresponding transfer totals were 602,112 and 47,340,960 bytes. The extra 46,738,848 bytes and 42 copies equal the aggregate byte size and count of its ONNX initializers. The probe records sizes and directions. Identifying individual tensors would need additional instrumentation.

The timing workers ran separately from this probe. Nsight Systems 2025.1.3 with CUDA graph-level tracing on this driver failed during the second fixed ViT forward. Both the plain application and this CUPTI probe completed the sequence. [NVIDIA CUPTI callback and activity APIs](https://docs.nvidia.com/cupti/12.9.1/main/main.html)

## Colab L4

The published L4 run used an NVIDIA L4 with compute capability 8.9, driver 580.82.07, CUDA 12.9, cuDNN 9.10.2, OpenCV 5.1.0 and ONNX Runtime GPU 1.22.0. The machine exposed 12 logical CPUs. Both engines used six CPU threads. The two fixed model files, two PNGs and labels matched the inputs from the GTX 1080 Ti benchmark byte for byte. The environment manifest records the original Colab ResNet export hash, the selected input hashes and the final ZIP checksum.

- [Individual samples](l4-samples.csv) contains all 2,600 timings from 26 sessions, including two CPU references.
- [Per-session results](l4-sessions.csv) includes p50 and p95, cold start, memory, GPU placement and score agreement against the CPU. A blank retained GPU-memory field means the runtime did not expose a matching process-memory reading.
- [Paired TF32 results](l4-paired.csv) gives the 12 off/on comparisons and agreement of all logits and the top-five sets on the two photographs. [Session-median ranges](l4-summary.csv) summarizes the three pairs per model and CUDA backend, including the observed off/on ratios.
- [Environment and hashes](l4-environment.json) identifies the hardware, software, models, photographs and benchmark source. The [notebook](../tf32_colab.ipynb) reproduces the procedure and downloads the raw scores, timings and per-model placement logs.
