# Colorization Asset Attribution

This demo uses the ECCV 2016 colorization model by Richard Zhang, Phillip
Isola, and Alexei A. Efros:

- Upstream project: <https://github.com/richzhang/colorization>
- Project page: <https://richzhang.github.io/colorization/>
- Paper: *Colorful Image Colorization*, ECCV 2016

## ONNX Release Asset

`models/colorization_eccv16.onnx` is an ONNX export of the official PyTorch
`eccv16` checkpoint. The export keeps the original 256×256 lightness-channel
input and two-channel Lab chroma output. It was checked numerically against the
upstream PyTorch model before publication.

- Original checkpoint:
  `https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth`
- ONNX SHA-256:
  `a1680679b609ca4d107edb83b8ac89c283cc474ce0a81edd6f01db85910e8201`

The ONNX file is hosted as a versioned LearnOpenCV GitHub Release asset so the
repository checkout stays small. `getModels.sh` verifies the checksum before
installing it. `export_onnx.py` reproduces the asset from an upstream checkout;
its extra packages are listed in `requirements-export.txt`.

## Legacy Caffe Support Files

`pts_in_hull.npy` and `models/colorization_deploy_v2.prototxt` are retained for
readers comparing the previous Caffe implementation. The OpenCV 5 code does not
load them.

Both files came from the `caffe` branch of the upstream project:

- `https://raw.githubusercontent.com/richzhang/colorization/caffe/resources/pts_in_hull.npy`
- `https://raw.githubusercontent.com/richzhang/colorization/caffe/models/colorization_deploy_v2.prototxt`

## License

The upstream project is copyright © 2016 Richard Zhang, Phillip Isola, and
Alexei A. Efros. Its BSD-style license permits redistribution and use in source
and binary forms when the copyright notice, conditions, and disclaimer are
retained.
