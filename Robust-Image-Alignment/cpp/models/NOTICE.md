# Native ALIKED and LightGlue models

These eleven ONNX graphs are a portable export of the **same** ALIKED-n16 and
ALIKED-LightGlue checkpoints used by the Python companion. They are included
so configuring, compiling, and running C++ needs no Python installation.
`manifest.json` records each graph's SHA-256 and byte count. The combined
ONNX payload is 50,779,236 bytes. No fixed 640-pixel square input, n16rot
checkpoint, or reduced feature budget is substituted.

ALIKED has two graphs: the dynamic dense encoder and sparse descriptor head.
Native C++ implements the original DKD detector between them. The dense graph
uses standard GridSample operations to express deformable convolution rather
than a custom TorchVision operator. Images remain RGB float values, with
Kornia-compatible Gaussian antialiasing and bilinear resize, symmetric
replicate padding to a multiple of 32, and unpadding before detection. The
shorter resize dimension is truncated, as in Kornia. Returned coordinates use
the original pixel-center scale conversion.

LightGlue has one graph per transformer layer. Native C++ retains the original
CPU early stopping and point pruning rather than always running all nine
layers. Defaults are depth confidence 0.95, width confidence 0.99, and match
threshold 0.1; point pruning is active at every count as in upstream CPU
inference. GPU-specific pruning thresholds and FlashAttention are not used by
this CPU companion. Floating-point differences can change a decision near a
threshold, so feature and match comparisons use explicit numerical tolerances.

## Sources and original checkpoints

- [cvg/LightGlue](https://github.com/cvg/LightGlue/tree/5a650c444718caffc2669dc0094cde908f42dd73),
  commit `5a650c444718caffc2669dc0094cde908f42dd73`; Apache-2.0.
- [ALIKED](https://github.com/Shiaoming/ALIKED/tree/683d7c65197395c0b3f01ebe76e1084a27e73a65),
  commit `683d7c65197395c0b3f01ebe76e1084a27e73a65`; BSD-3-Clause,
  copyright (c) 2022 Zhao Xiaoming.
- Original ALIKED checkpoint: `aliked-n16.pth`, 2,738,091 bytes,
  SHA-256 `5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2`,
  [source](https://github.com/Shiaoming/ALIKED/blob/683d7c65197395c0b3f01ebe76e1084a27e73a65/models/aliked-n16.pth).
- Original LightGlue checkpoint: `aliked_lightglue_v0-1_arxiv.pth`,
  47,632,827 bytes, SHA-256
  `d975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503`,
  [source](https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/aliked_lightglue.pth).
- Portable deformable-convolution and descriptor indexing were also informed
  by [LightGlue-ONNX's ALIKED adaptation](https://github.com/fabio-sim/LightGlue-ONNX/blob/d12b4ba1632f558234e3f084e1f3d8bdf9147890/lightglue_dynamo/models/aliked.py),
  commit `d12b4ba1632f558234e3f084e1f3d8bdf9147890`. Its repository is Apache-2.0;
  the ALIKED descriptor adaptation retains the original BSD-3-Clause notice.

Copies of all three source licenses accompany these graphs. The export and
native adaptations are modifications for this tutorial, not upstream model
releases. No model retraining is performed.

## Reproducing the export

The optional authoring tool is `../../scripts/export_cpp_models.py`. In the
project's pinned Python development environment, install `onnx==1.20.1` and
`onnxruntime==1.24.3`, then run:

```bash
python scripts/export_cpp_models.py --output-dir cpp/models --validate
```

The checked-in `export_validation.json` records comparisons with the original
PyTorch implementation at two dense input sizes and two sparse point counts,
including every output of all nine LightGlue layers. Its 132 comparisons pass
the declared maximum absolute error of 0.0002. This graph check supplements
the full-image feature/match and workflow tests described in the project README.

ONNX Runtime itself is downloaded from Microsoft's official NuGet package,
version 1.24.3, by `../onnxruntime.cmake`. That package includes its own license
and third-party notices; the archive is verified with the pinned SHA-256 before
the matching headers and platform's CPU library are extracted.
