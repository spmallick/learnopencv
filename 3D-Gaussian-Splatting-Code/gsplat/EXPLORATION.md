# Exploration For Better 3D Gaussian Splatting

- AbsGrad: Uses absolute gradients in the image plane as the criterion for pruning. See [this paper](https://arxiv.org/pdf/2404.10484) for more details.
- Antialiasing: Applies a low pass filter on the projected covariance and scale the opacity accordingly. See [this paper](https://niujinshuchong.github.io/mip-splatting/) for more details. It might slightly hurt the metrics on in-distribution views but seem to improve the visual quality on view out of training distribution.

| Garden at 7k steps (TITAN RTX)       | T(train) | T(render) | Memory  | SSIM   | PSNR  | LPIPS | #GS.  |
| ------------------------------------ | -------- | --------- | ------- | ------ | ----- | ----- | ----- |
| default args                         | 7m07s    | 0.021s/im | 7.54 GB | 0.8332 | 26.29 | 0.123 | 4.46M |
| `--absgrad --grow_grad2d 8e-4`       | 5m50s    | 0.012s/im | 3.80 GB | 0.8365 | 26.44 | 0.121 | 2.17M |
| `--absgrad --grow_grad2d 8e-4` (30k) | --       | 0.013s/im | 4.04 GB | 0.8639 | 27.33 | 0.079 | 2.35M |
| `--antialiased`                      | 6m43s    | 0.020s/im | 6.74 GB | 0.8265 | 26.13 | 0.137 | 3.99M |

| U1 at 7k steps (RTX 2080 Ti)         | T(train) | T(render) | Memory  | SSIM   | PSNR  | LPIPS | #GS.  |
| ------------------------------------ | -------- | --------- | ------- | ------ | ----- | ----- | ----- |
| default args                         | 7m39s    | 0.013s/im | 4.94 GB | 0.6102 | 20.69 | 0.615 | 2.47M |
| default args (30k)                   | --       | 0.019s/im | --      | 0.7518 | 24.67 | 0.385 | 4.18M |
| `--absgrad --grow_grad2d 8e-4`       | 7m16s    | 0.011s/im | 3.41 GB | 0.6055 | 20.29 | 0.636 | 1.72M |
| `--absgrad --grow_grad2d 8e-4` (30k) | --       | 0.014s/im | 4.15 GB | 0.7494 | 24.65 | 0.390 | 2.37M |
| `--absgrad --grow_grad2d 6e-4`       | 8m58s    | 0.011s/im | 4.42 GB | 0.5966 | 19.58 | 0.654 | 2.21M |
| `--absgrad --grow_grad2d 6e-4` (30k) | --       | 0.016s/im | 5.09 GB | 0.7439 | 24.28 | 0.400 | 2.92M |

| U4 at 7k steps (RTX 2080 Ti)                 | T(train) | T(render) | Memory  | SSIM   | PSNR  | LPIPS | #GS.  |
| -------------------------------------------- | -------- | --------- | ------- | ------ | ----- | ----- | ----- |
| `--grow_grad2d 5e-5`                         | 7m30s    | 0.014s/im | 1.68 GB | 0.6271 | 20.86 | 0.583 | 0.61M |
| `--grow_grad2d 5e-5` (30k)                   | --       | 0.026s/im | 4.21 GB | 0.7402 | 24.05 | 0.299 | 2.44M |
| `--absgrad --grow_grad2d 2e-4`               | 8m30s    | 0.018s/im | 2.21 GB | 0.6251 | 20.68 | 0.587 | 0.89M |
| `--absgrad --grow_grad2d 2e-4` (30k)         | --       | 0.030s/im | 5.25 GB | 0.7442 | 24.12 | 0.291 | 2.62M |

Note: default args means running `CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --data_dir <DATA_DIR>` with:

- Garden ([Source](https://jonbarron.info/mipnerf360/)): `--result_dir results/garden`
- U1 (a.k.a University 1 from [Source](https://localrf.github.io/)): `--result_dir results/u1 --data_factor 1 --grow_scale3d 0.001`
- U4 (a.k.a University 4 from [Source](https://localrf.github.io/)): `--result_dir results/u4 --data_factor 1 --grow_scale3d 0.01 --refine_every 500`
