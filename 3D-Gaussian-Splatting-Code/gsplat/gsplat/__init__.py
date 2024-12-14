import warnings

from .compression import PngCompression
from .cuda._torch_impl import accumulate
from .cuda._torch_impl_2dgs import accumulate_2dgs
from .cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    proj,
    quat_scale_to_covar_preci,
    rasterize_to_indices_in_range,
    rasterize_to_pixels,
    spherical_harmonics,
    world_to_cam,
    fully_fused_projection_2dgs,
    rasterize_to_pixels_2dgs,
    rasterize_to_indices_in_range_2dgs,
)
from .rendering import (
    rasterization,
    rasterization_2dgs,
    rasterization_inria_wrapper,
    rasterization_2dgs_inria_wrapper,
)
from .strategy import DefaultStrategy, MCMCStrategy, Strategy
from .version import __version__

all = [
    "PngCompression",
    "DefaultStrategy",
    "MCMCStrategy",
    "Strategy",
    "rasterization",
    "rasterization_2dgs",
    "rasterization_inria_wrapper",
    "spherical_harmonics",
    "isect_offset_encode",
    "isect_tiles",
    "proj",
    "fully_fused_projection",
    "quat_scale_to_covar_preci",
    "rasterize_to_pixels",
    "world_to_cam",
    "accumulate",
    "rasterize_to_indices_in_range",
    "full_fused_projection_2dgs",
    "rasterize_to_pixels_2dgs",
    "rasterize_to_indices_in_range_2dgs",
    "accumulate_2dgs",
    "rasterization_2dgs_inria_wrapper",
    "__version__",
]
