#!/usr/bin/env python3
"""Sample script to run DepthPro.

Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""


import argparse
import logging
from pathlib import Path
import cv2
import numpy as np
import PIL.Image
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from depth_pro import create_model_and_transforms, load_rgb

LOGGER = logging.getLogger(__name__)


def get_torch_device() -> torch.device:
    """Get the Torch device."""
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device


def run(args):
    """Run Depth Pro on a sample image."""
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Load model.
    model, transform = create_model_and_transforms(
        device=get_torch_device(),
        precision=torch.half,
    )
    model.eval()

    image_paths = [args.image_path]
    if args.image_path.is_dir():
        image_paths = args.image_path.glob("**/*")
        relative_path = args.image_path
    else:
        relative_path = args.image_path.parent

    if not args.skip_display:
        plt.ion()
        fig = plt.figure()
        ax_rgb = fig.add_subplot(121)
        ax_disp = fig.add_subplot(122)

    for image_path in tqdm(image_paths):
        # Load image and focal length from exif info (if found.).
        try:
            LOGGER.info(f"Loading image {image_path} ...")
            image, _, f_px = load_rgb(image_path)
        except Exception as e:
            LOGGER.error(str(e))
            continue
        # Run prediction. If `f_px` is provided, it is used to estimate the final metric depth,
        # otherwise the model estimates `f_px` to compute the depth metricness.
        prediction = model.infer(transform(image), f_px=f_px)

        # Extract the depth and focal length.
        depth = prediction["depth"].detach().cpu().numpy().squeeze()
        if f_px is not None:
            LOGGER.debug(f"Focal length (from exif): {f_px:0.2f}")
        elif prediction["focallength_px"] is not None:
            focallength_px = prediction["focallength_px"].detach().cpu().item()
            print(f"Estimated focal length: {focallength_px}")
            LOGGER.info(f"Estimated focal length: {focallength_px}")

        inverse_depth = 1 / depth
        # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
        max_invdepth_vizu = inverse_depth.max()
        min_invdepth_vizu = inverse_depth.min()
        
        inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
            max_invdepth_vizu - min_invdepth_vizu
        )
        inverse_depth_grayscale = (inverse_depth_normalized * 255).astype(np.uint8)
        
        max_depth_vizu = min(depth.max(), 1 / 0.1) 
        min_depth_vizu = max(depth.min(), 1 / 250)
        # min_depth_vizu, max_depth_vizu = np.min(depth), np.max(depth)
        
        # depth_normalized = (depth - min_depth_vizu) / (max_depth_vizu - min_depth_vizu)
        
        depth_clipped = np.clip(depth, min_depth_vizu, max_depth_vizu)
        
        depth_normalized = (depth_clipped - min_depth_vizu) / (max_depth_vizu - min_depth_vizu)
        
        grayscale_depth = (depth_normalized * 255).astype(np.uint8)
        
        print(f"Original depth range: min={depth.min()}, max={depth.max()}")
        print(f"Original Inverse Depth range: min={inverse_depth.min()}, max={inverse_depth.max()}")
        print(f"Clipped depth range: min={min_depth_vizu}, max={max_depth_vizu}")
        print(f"Inverse depth range: min={min_invdepth_vizu}, max={max_invdepth_vizu}")

        #***************** SURFACE NORMAL ***********************
        kernel_size = 7
        grad_x = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 1, 0, ksize = kernel_size)
        grad_y = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 0, 1, ksize = kernel_size)
        z = np.full(grad_x.shape, 1)
        normals = np.dstack((-grad_x, -grad_y, z))
        
        normals_mag = np.linalg.norm(normals, axis= 2, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            normals_normalized   = normals / (normals_mag + 1e-5)
        
        normals_normalized = np.nan_to_num(normals_normalized, nan = -1, posinf=-1, neginf=-1)
        normal_from_depth = ((normals_normalized + 1) / 2 * 255).astype(np.uint8)
        # normal_from_depth = normal_from_depth[:, :, ::-1] #RGB to BGR


        #***************** SURFACE NORMAL ***********************

        # Save Depth as npz file.
        if args.output_path is not None:
            output_file = (
                args.output_path
                / image_path.relative_to(relative_path).parent
                / image_path.stem
            )
            LOGGER.info(f"Saving depth map to: {str(output_file)}")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_file, depth=depth)

           # Save as color-mapped "turbo" jpg image.
            cmap = plt.get_cmap("inferno")
            color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(
                np.uint8
            )
            
            inverse_cmap_output_file = str(output_file) + "_inverse_depth_cmap.jpg"
            LOGGER.info(f"Saving inverse-mapped colored depth to: : {inverse_cmap_output_file}")
            PIL.Image.fromarray(color_depth).save(
                inverse_cmap_output_file, format="JPEG", quality=90
            )
            
            
            
            inverse_depth_map_output_file = str(output_file) + "_inverse_depth_bnw.jpg"
            LOGGER.info(f"Saving inverse-mapped depth to: : {inverse_depth_map_output_file}")
            # PIL.Image.fromarray(inverse_depth_grayscale).save(
            #     inverse_depth_map_output_file, format="JPEG", quality=90
            # )
            
            raw_depth_output_file = str(output_file) + "_depth_grayscale.jpg"
            LOGGER.info(f"Saving normal depth grayscale map to: {raw_depth_output_file}")
            
            # PIL.Image.fromarray(grayscale_depth).save(
            #     raw_depth_output_file, format="JPEG", quality = 90
            # )
            
            surface_normal_fname = str(output_file) + "_surface_normal.jpg"
            LOGGER.info(f"Saving normal depth grayscale map to: {surface_normal_fname}")
            
            # PIL.Image.fromarray(normal_from_depth).save(
            #     surface_normal_fname, format="JPEG", quality = 90
            # )
            

        # Display the image and estimated depth map.
        if not args.skip_display:
            ax_rgb.imshow(image)
            ax_disp.imshow(depth_normalized, cmap="inferno")
            fig.canvas.draw()
            fig.canvas.flush_events()

    LOGGER.info("Done predicting depth!")
    if not args.skip_display:
        plt.show(block=True)


def main():
    """Run DepthPro inference example."""
    parser = argparse.ArgumentParser(
        description="Inference scripts of DepthPro with PyTorch models."
    )
    parser.add_argument(
        "-i", 
        "--image-path", 
        type=Path, 
        default="./data/example.jpg",
        help="Path to input image.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        help="Path to store output files.",
    )
    parser.add_argument(
        "--skip-display",
        action="store_true",
        help="Skip matplotlib display.",
    )
    parser.add_argument(
        "-v", 
        "--verbose", 
        action="store_true", 
        help="Show verbose output."
    )
    
    run(parser.parse_args())


if __name__ == "__main__":
    main()









































# #!/usr/bin/env python3
# """Sample script to run DepthPro.

# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# """


# import argparse
# import logging
# from pathlib import Path

# import numpy as np
# import PIL.Image
# import torch
# from matplotlib import pyplot as plt
# from tqdm import tqdm

# from depth_pro import create_model_and_transforms, load_rgb

# LOGGER = logging.getLogger(__name__)


# def get_torch_device() -> torch.device:
#     """Get the Torch device."""
#     device = torch.device("cpu")
#     if torch.cuda.is_available():
#         device = torch.device("cuda:0")
#     elif torch.backends.mps.is_available():
#         device = torch.device("mps")
#     return device


# def run(args):
#     """Run Depth Pro on a sample image."""
#     if args.verbose:
#         logging.basicConfig(level=logging.INFO)

#     # Load model.
#     model, transform = create_model_and_transforms(
#         device=get_torch_device(),
#         precision=torch.half,
#     )
#     model.eval()

#     image_paths = [args.image_path]
#     if args.image_path.is_dir():
#         image_paths = args.image_path.glob("**/*")
#         relative_path = args.image_path
#     else:
#         relative_path = args.image_path.parent

#     if not args.skip_display:
#         plt.ion()
#         fig = plt.figure()
#         ax_rgb = fig.add_subplot(121)
#         ax_disp = fig.add_subplot(122)

#     for image_path in tqdm(image_paths):
#         # Load image and focal length from exif info (if found.).
#         try:
#             LOGGER.info(f"Loading image {image_path} ...")
#             image, _, f_px = load_rgb(image_path)
#         except Exception as e:
#             LOGGER.error(str(e))
#             continue
#         # Run prediction. If `f_px` is provided, it is used to estimate the final metric depth,
#         # otherwise the model estimates `f_px` to compute the depth metricness.
#         prediction = model.infer(transform(image), f_px=f_px)

#         # Extract the depth and focal length.
#         depth = prediction["depth"].detach().cpu().numpy().squeeze()
#         if f_px is not None:
#             LOGGER.debug(f"Focal length (from exif): {f_px:0.2f}")
#         elif prediction["focallength_px"] is not None:
#             focallength_px = prediction["focallength_px"].detach().cpu().item()
#             LOGGER.info(f"Estimated focal length: {focallength_px}")

#         inverse_depth = 1 / depth
#         # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
#         # max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
#         # min_invdepth_vizu = max(1 / 250, inverse_depth.min())
#         max_invdepth_vizu = inverse_depth.max()
#         min_invdepth_vizu = inverse_depth.min()
        
#         inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
#             max_invdepth_vizu - min_invdepth_vizu
#         )
#         inverse_depth_grayscale = (inverse_depth_normalized * 255).astype(np.uint8)
#         # Save Depth as npz file.
#         if args.output_path is not None:
#             output_file = (
#                 args.output_path
#                 / image_path.relative_to(relative_path).parent
#                 / image_path.stem
#             )
#             LOGGER.info(f"Saving depth map to: {str(output_file)}")
#             output_file.parent.mkdir(parents=True, exist_ok=True)
#             np.savez_compressed(output_file, depth=depth)

#             # Save as color-mapped "turbo" jpg image.
#             cmap = plt.get_cmap("magma")
#             color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(
#                 np.uint8
#             )
#             color_map_output_file = str(output_file) + "color.jpg"
#             LOGGER.info(f"Saving color-mapped depth to: : {color_map_output_file}")
#             PIL.Image.fromarray(color_depth).save(
#                 color_map_output_file, format="JPEG", quality=90
#             )
            
            
#             gray_map_output_file = str(output_file) + "gray_inverse_depth.jpg"
#             LOGGER.info(f"Saving color-mapped depth to: : {gray_map_output_file}")
#             PIL.Image.fromarray(inverse_depth_grayscale).save(
#                 gray_map_output_file, format="JPEG", quality=90
#             )
            
#         # Display the image and estimated depth map.
#         if not args.skip_display:
#             ax_rgb.imshow(image)
#             ax_disp.imshow(inverse_depth_normalized, cmap="magma")
#             fig.canvas.draw()
#             fig.canvas.flush_events()

#     LOGGER.info("Done predicting depth!")
#     if not args.skip_display:
#         plt.show(block=True)


# def main():
#     """Run DepthPro inference example."""
#     parser = argparse.ArgumentParser(
#         description="Inference scripts of DepthPro with PyTorch models."
#     )
#     parser.add_argument(
#         "-i", 
#         "--image-path", 
#         type=Path, 
#         default="./data/example.jpg",
#         help="Path to input image.",
#     )
#     parser.add_argument(
#         "-o",
#         "--output-path",
#         type=Path,
#         help="Path to store output files.",
#     )
#     parser.add_argument(
#         "--skip-display",
#         action="store_true",
#         help="Skip matplotlib display.",
#     )
#     parser.add_argument(
#         "-v", 
#         "--verbose", 
#         action="store_true", 
#         help="Show verbose output."
#     )
    
#     run(parser.parse_args())


# if __name__ == "__main__":
#     main()