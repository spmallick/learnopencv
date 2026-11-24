"""Inspect CUDA placement and optionally count launches, allocations, and transfers per forward."""

import argparse
import ctypes
import tempfile
from pathlib import Path

import cv2
import numpy as np


def write_control(path):
    """Add an unsupported TopK to a small, otherwise CUDA-capable network."""
    import onnx
    from onnx import TensorProto, helper

    # These dimensions match the classifier inputs so the same probe works on both.
    images = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 224, 224])
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 1])
    k = helper.make_tensor("k", TensorProto.INT64, [1], [1])

    # Pool each channel, flatten to [1, 3], then select the largest channel mean.
    nodes = [
        helper.make_node("GlobalAveragePool", ["images"], ["means"], name="pool"),
        helper.make_node("Flatten", ["means"], ["flat"], name="flatten"),
        helper.make_node("TopK", ["flat", "k"], ["scores", "indices"], name="top_channel"),
    ]
    graph = helper.make_graph(nodes, "fallback_control", [images], [scores], [k])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save_model(model, path)


def run(args, path):
    tracer = None
    if args.trace_api:
        tracer = ctypes.CDLL(str(args.trace_api.resolve()))
        tracer.trace_read.argtypes = [ctypes.c_uint]
        tracer.trace_read.restype = ctypes.c_uint64
        if tracer.trace_start() != 0:
            raise RuntimeError("CUPTI callback subscription failed")

    # Use the native engine, including when OpenCV was compiled with ORT support.
    net = cv2.dnn.readNetFromONNX(str(path), cv2.dnn.ENGINE_OPENCV)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    net.enableTF32(False)

    if tracer and tracer.trace_flush():
        raise RuntimeError("CUPTI could not flush its initialization records")

    # Reuse each host tensor so the trace highlights shape changes and network work.
    tensors = {b: np.ones((b, 3, 224, 224), np.float32) for b in args.batches}
    for index in range(args.forwards):
        batch = args.batches[index % len(args.batches)]
        blob = tensors[batch]
        blob[:, 0] = index + 2  # Changing values also checks replay reads fresh input.
        if tracer:
            before = [tracer.trace_read(i) for i in range(8)]
        net.setInput(blob, args.input_name)
        output = net.forward()  # Returning host scores waits for the requested output.
        if tracer:
            if tracer.trace_flush():
                raise RuntimeError("CUPTI dropped records or could not flush this forward")
            names = ("captures", "graph_launches", "runtime_kernel_launches", "h2d_copies",
                     "h2d_bytes", "malloc_calls", "malloc_bytes", "free_calls")
            counts = [tracer.trace_read(i) - n for i, n in enumerate(before)]
            metrics = " ".join(f"{name}={n}" for name, n in zip(names, counts))
            print(f"API forward={index + 1} batch={batch} {metrics}", flush=True)
        if not np.isfinite(output).all():
            raise RuntimeError("The network returned a non-finite value")
        if args.fallback_demo:
            np.testing.assert_allclose(output.reshape(-1), [index + 2], rtol=1e-5)
        print(f"forward {index + 1}  batch {batch}  first score {output.flat[0]:.6g}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path)
    source.add_argument("--fallback-demo", action="store_true", help="generate a small TopK control")
    parser.add_argument("--input-name", default="images")
    parser.add_argument("--batches", type=int, nargs="+", default=[1], help="cycle through these batch sizes")
    parser.add_argument("--forwards", type=int, default=5)
    parser.add_argument("--trace-api", type=Path, help="compiled CUPTI API-counter library")
    args = parser.parse_args()
    if min(args.batches) <= 0 or args.forwards <= 0:
        parser.error("batches and forwards must be positive")
    if args.fallback_demo and (args.batches != [1] or args.input_name != "images"):
        parser.error("the control has a fixed batch of one and an input named images")

    with tempfile.TemporaryDirectory() as temp:
        path = args.model
        if args.fallback_demo:
            path = Path(temp) / "topk-control.onnx"
            write_control(path)
        run(args, path)


if __name__ == "__main__":
    main()
