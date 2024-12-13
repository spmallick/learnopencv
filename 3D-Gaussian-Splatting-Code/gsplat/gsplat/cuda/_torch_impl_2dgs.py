from typing import Optional, Tuple

import torch
from torch import Tensor

from gsplat.cuda._torch_impl import _quat_scale_to_matrix


def _fully_fused_projection_2dgs(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """PyTorch implementation of `gsplat.cuda._wrapper.fully_fused_projection_2dgs()`

    .. note::

        This is a minimal implementation of fully fused version, which has more
        arguments. Not all arguments are supported.
    """
    R_cw = viewmats[:, :3, :3]  # [C, 3, 3]
    t_cw = viewmats[:, :3, 3]  # [C, 3]
    means_c = torch.einsum("cij,nj->cni", R_cw, means) + t_cw[:, None, :]  # (C, N, 3)
    RS_wl = _quat_scale_to_matrix(quats, scales)
    RS_cl = torch.einsum("cij,njk->cnik", R_cw, RS_wl)  # [C, N, 3, 3]

    # compute normals
    normals = RS_cl[..., 2]  # [C, N, 3]
    C, N, _ = normals.shape
    cos = -normals.reshape((C * N, 1, 3)) @ means_c.reshape((C * N, 3, 1))
    cos = cos.reshape((C, N, 1))
    multiplier = torch.where(cos > 0, torch.tensor(1.0), torch.tensor(-1.0))
    normals *= multiplier

    # ray transform matrix, omitting the z rotation
    T_cl = torch.cat([RS_cl[..., :2], means_c[..., None]], dim=-1)  # [C, N, 3, 3]
    T_sl = torch.einsum("cij,cnjk->cnik", Ks[:, :3, :3], T_cl)  # [C, N, 3, 3]
    # in paper notation M = (WH)^T
    # later h_u = M @ h_x, h_v = M @ h_y
    M = torch.transpose(T_sl, -1, -2)  # [C, N, 3, 3]

    # compute the AABB of gaussian
    test = torch.tensor([1.0, 1.0, -1.0], device=means.device).reshape(1, 1, 3)
    d = (M[..., 2] * M[..., 2] * test).sum(dim=-1, keepdim=True)  # [C, N, 1]
    valid = torch.abs(d) > eps
    f = torch.where(valid, test / d, torch.zeros_like(test)).unsqueeze(
        -1
    )  # (C, N, 3, 1)
    means2d = (M[..., :2] * M[..., 2:3] * f).sum(dim=-2)  # [C, N, 2]
    extents = torch.sqrt(
        means2d**2 - (M[..., :2] * M[..., :2] * f).sum(dim=-2)
    )  # [C, N, 2]

    depths = means_c[..., 2]  # [C, N]
    radius = torch.ceil(3.0 * torch.max(extents, dim=-1).values)  # (C, N)

    valid = valid.squeeze(-1) & (depths > near_plane) & (depths < far_plane)
    radius[~valid] = 0.0

    inside = (
        (means2d[..., 0] + radius > 0)
        & (means2d[..., 0] - radius < width)
        & (means2d[..., 1] + radius > 0)
        & (means2d[..., 1] - radius < height)
    )
    radius[~inside] = 0.0
    radii = radius.int()
    return radii, means2d, depths, M, normals


def accumulate_2dgs(
    means2d: Tensor,  # [C, N, 2]
    ray_transforms: Tensor,  # [C, N, 3, 3]
    opacities: Tensor,  # [C, N]
    colors: Tensor,  # [C, N, channels]
    normals: Tensor,  # [C, N, 3]
    gaussian_ids: Tensor,  # [M]
    pixel_ids: Tensor,  # [M]
    camera_ids: Tensor,  # [M]
    image_width: int,
    image_height: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Alpha compositing for 2DGS.

    .. warning::
        This function requires the nerfacc package to be installed. Please install it using the following command pip install nerfacc.

    Args:
        means2d: Gaussian means in 2D. [C, N, 2]
        ray_transforms: transformation matrices that transform rays in pixel space into splat's local frame. [C, N, 3, 3]
        opacities: Per-view Gaussian opacities (for example, when antialiasing is enabled, Gaussian in
            each view would efficiently have different opacity). [C, N]
        colors: Per-view Gaussian colors. Supports N-D features. [C, N, channels]
        normals: Per-view Gaussian normals. [C, N, 3]
        gaussian_ids: Collection of Gaussian indices to be rasterized. A flattened list of shape [M].
        pixel_ids: Collection of pixel indices (row-major) to be rasterized. A flattened list of shape [M].
        camera_ids: Collection of camera indices to be rasterized. A flattened list of shape [M].
        image_width: Image width.
        image_height: Image height.

    Returns:
        A tuple:

        **renders**: Accumulated colors. [C, image_height, image_width, channels]

        **alphas**: Accumulated opacities. [C, image_height, image_width, 1]

        **normals**: Accumulated opacities. [C, image_height, image_width, 3]
    """

    try:
        from nerfacc import accumulate_along_rays, render_weight_from_alpha
    except ImportError:
        raise ImportError("Please install nerfacc package: pip install nerfacc")

    C, N = means2d.shape[:2]
    channels = colors.shape[-1]

    pixel_ids_x = pixel_ids % image_width + 0.5
    pixel_ids_y = pixel_ids // image_width + 0.5
    pixel_coords = torch.stack([pixel_ids_x, pixel_ids_y], dim=-1)  # [M, 2]
    deltas = pixel_coords - means2d[camera_ids, gaussian_ids]  # [M, 2]

    M = ray_transforms[camera_ids, gaussian_ids]  # [M, 3, 3]

    h_u = -M[..., 0, :3] + M[..., 2, :3] * pixel_ids_x[..., None]  # [M, 3]
    h_v = -M[..., 1, :3] + M[..., 2, :3] * pixel_ids_y[..., None]  # [M, 3]
    tmp = torch.cross(h_u, h_v, dim=-1)
    us = tmp[..., 0] / tmp[..., 2]
    vs = tmp[..., 1] / tmp[..., 2]
    sigmas_3d = us**2 + vs**2  # [M]
    sigmas_2d = 2 * (deltas[..., 0] ** 2 + deltas[..., 1] ** 2)
    sigmas = 0.5 * torch.minimum(sigmas_3d, sigmas_2d)  # [M]

    alphas = torch.clamp_max(
        opacities[camera_ids, gaussian_ids] * torch.exp(-sigmas), 0.999
    )

    indices = camera_ids * image_height * image_width + pixel_ids
    total_pixels = C * image_height * image_width

    weights, trans = render_weight_from_alpha(
        alphas, ray_indices=indices, n_rays=total_pixels
    )
    renders = accumulate_along_rays(
        weights,
        colors[camera_ids, gaussian_ids],
        ray_indices=indices,
        n_rays=total_pixels,
    ).reshape(C, image_height, image_width, channels)
    alphas = accumulate_along_rays(
        weights, None, ray_indices=indices, n_rays=total_pixels
    ).reshape(C, image_height, image_width, 1)
    renders_normal = accumulate_along_rays(
        weights,
        normals[camera_ids, gaussian_ids],
        ray_indices=indices,
        n_rays=total_pixels,
    ).reshape(C, image_height, image_width, 3)

    return renders, alphas, renders_normal


def _rasterize_to_pixels_2dgs(
    means2d: Tensor,  # [C, N, 2]
    ray_transforms: Tensor,  # [C, N, 3, 3]
    colors: Tensor,  # [C, N, channels]
    normals: Tensor,  # [C, N, 3]
    opacities: Tensor,  # [C, N]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
    backgrounds: Optional[Tensor] = None,  # [C, channels]
    batch_per_iter: int = 100,
):
    """Pytorch implementation of `gsplat.cuda._wrapper.rasterize_to_pixels_2dgs()`.

    This function rasterizes 2D Gaussians to pixels in a Pytorch-friendly way. It
    iteratively accumulates the renderings within each batch of Gaussians. The
    interations are controlled by `batch_per_iter`.

    .. note::
        This is a minimal implementation of the fully fused version, which has more
        arguments. Not all arguments are supported.

    .. note::

        This function relies on Pytorch's autograd for the backpropagation. It is much slower
        than our fully fused rasterization implementation and comsumes much more GPU memory.
        But it could serve as a playground for new ideas or debugging, as no backward
        implementation is needed.

    .. warning::

        This function requires the `nerfacc` package to be installed. Please install it
        using the following command `pip install nerfacc`.
    """
    from ._wrapper import rasterize_to_indices_in_range_2dgs

    C, N = means2d.shape[:2]
    n_isects = len(flatten_ids)
    device = means2d.device

    render_colors = torch.zeros(
        (C, image_height, image_width, colors.shape[-1]), device=device
    )
    render_alphas = torch.zeros((C, image_height, image_width, 1), device=device)
    render_normals = torch.zeros((C, image_height, image_width, 3), device=device)

    # Split Gaussians into batches and iteratively accumulate the renderings
    block_size = tile_size * tile_size
    isect_offsets_fl = torch.cat(
        [isect_offsets.flatten(), torch.tensor([n_isects], device=device)]
    )
    max_range = (isect_offsets_fl[1:] - isect_offsets_fl[:-1]).max().item()
    num_batches = (max_range + block_size - 1) // block_size
    for step in range(0, num_batches, batch_per_iter):
        transmittances = 1.0 - render_alphas[..., 0]

        # Find the M intersections between pixels and gaussians.
        # Each intersection corresponds to a tuple (gs_id, pixel_id, camera_id)
        gs_ids, pixel_ids, camera_ids = rasterize_to_indices_in_range_2dgs(
            step,
            step + batch_per_iter,
            transmittances,
            means2d,
            ray_transforms,
            opacities,
            image_width,
            image_height,
            tile_size,
            isect_offsets,
            flatten_ids,
        )  # [M], [M]
        if len(gs_ids) == 0:
            break

        # Accumulate the renderings within this batch of Gaussians.
        renders_step, accs_step, renders_normal_step = accumulate_2dgs(
            means2d,
            ray_transforms,
            opacities,
            colors,
            normals,
            gs_ids,
            pixel_ids,
            camera_ids,
            image_width,
            image_height,
        )
        render_colors = render_colors + renders_step * transmittances[..., None]
        render_alphas = render_alphas + accs_step * transmittances[..., None]
        render_normals = (
            render_normals + renders_normal_step * transmittances[..., None]
        )

    render_alphas = render_alphas
    if backgrounds is not None:
        render_colors = render_colors + backgrounds[:, None, None, :] * (
            1.0 - render_alphas
        )

    return render_colors, render_alphas, render_normals
