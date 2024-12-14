import math
from typing import Dict, Optional, Tuple

import torch
import torch.distributed
import torch.nn.functional as F
from torch import Tensor
from typing_extensions import Literal

from .cuda._wrapper import (
    fully_fused_projection,
    fully_fused_projection_2dgs,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
    rasterize_to_pixels_2dgs,
    spherical_harmonics,
)
from .distributed import (
    all_gather_int32,
    all_gather_tensor_list,
    all_to_all_int32,
    all_to_all_tensor_list,
)
from .utils import depth_to_normal, get_projection_matrix


def rasterization(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    colors: Tensor,  # [(C,) N, D] or [(C,) N, K, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    packed: bool = True,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    channel_chunk: int = 32,
    distributed: bool = False,
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
    covars: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Dict]:
    """Rasterize a set of 3D Gaussians (N) to a batch of image planes (C).

    This function provides a handful features for 3D Gaussian rasterization, which
    we detail in the following notes. A complete profiling of the these features
    can be found in the :ref:`profiling` page.

    .. note::
        **Multi-GPU Distributed Rasterization**: This function can be used in a multi-GPU
        distributed scenario by setting `distributed` to True. When `distributed` is True,
        a subset of total Gaussians could be passed into this function in each rank, and
        the function will collaboratively render a set of images using Gaussians from all ranks. Note
        to achieve balanced computation, it is recommended (not enforced) to have similar number of
        Gaussians in each rank. But we do enforce that the number of cameras to be rendered
        in each rank is the same. The function will return the rendered images
        corresponds to the input cameras in each rank, and allows for gradients to flow back to the
        Gaussians living in other ranks. For the details, please refer to the paper
        `On Scaling Up 3D Gaussian Splatting Training <https://arxiv.org/abs/2406.18533>`_.

    .. note::
        **Batch Rasterization**: This function allows for rasterizing a set of 3D Gaussians
        to a batch of images in one go, by simplly providing the batched `viewmats` and `Ks`.

    .. note::
        **Support N-D Features**: If `sh_degree` is None,
        the `colors` is expected to be with shape [N, D] or [C, N, D], in which D is the channel of
        the features to be rendered. The computation is slow when D > 32 at the moment.
        If `sh_degree` is set, the `colors` is expected to be the SH coefficients with
        shape [N, K, 3] or [C, N, K, 3], where K is the number of SH bases. In this case, it is expected
        that :math:`(\\textit{sh_degree} + 1) ^ 2 \\leq K`, where `sh_degree` controls the
        activated bases in the SH coefficients.

    .. note::
        **Depth Rendering**: This function supports colors or/and depths via `render_mode`.
        The supported modes are "RGB", "D", "ED", "RGB+D", and "RGB+ED". "RGB" renders the
        colored image that respects the `colors` argument. "D" renders the accumulated z-depth
        :math:`\\sum_i w_i z_i`. "ED" renders the expected z-depth
        :math:`\\frac{\\sum_i w_i z_i}{\\sum_i w_i}`. "RGB+D" and "RGB+ED" render both
        the colored image and the depth, in which the depth is the last channel of the output.

    .. note::
        **Memory-Speed Trade-off**: The `packed` argument provides a trade-off between
        memory footprint and runtime. If `packed` is True, the intermediate results are
        packed into sparse tensors, which is more memory efficient but might be slightly
        slower. This is especially helpful when the scene is large and each camera sees only
        a small portion of the scene. If `packed` is False, the intermediate results are
        with shape [C, N, ...], which is faster but might consume more memory.

    .. note::
        **Sparse Gradients**: If `sparse_grad` is True, the gradients for {means, quats, scales}
        will be stored in a `COO sparse layout <https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html>`_.
        This can be helpful for saving memory
        for training when the scene is large and each iteration only activates a small portion
        of the Gaussians. Usually a sparse optimizer is required to work with sparse gradients,
        such as `torch.optim.SparseAdam <https://pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html#sparseadam>`_.
        This argument is only effective when `packed` is True.

    .. note::
        **Speed-up for Large Scenes**: The `radius_clip` argument is extremely helpful for
        speeding up large scale scenes or scenes with large depth of fields. Gaussians with
        2D radius smaller or equal than this value (in pixel unit) will be skipped during rasterization.
        This will skip all the far-away Gaussians that are too small to be seen in the image.
        But be warned that if there are close-up Gaussians that are also below this threshold, they will
        also get skipped (which is rarely happened in practice). This is by default disabled by setting
        `radius_clip` to 0.0.

    .. note::
        **Antialiased Rendering**: If `rasterize_mode` is "antialiased", the function will
        apply a view-dependent compensation factor
        :math:`\\rho=\\sqrt{\\frac{Det(\\Sigma)}{Det(\\Sigma+ \\epsilon I)}}` to Gaussian
        opacities, where :math:`\\Sigma` is the projected 2D covariance matrix and :math:`\\epsilon`
        is the `eps2d`. This will make the rendered image more antialiased, as proposed in
        the paper `Mip-Splatting: Alias-free 3D Gaussian Splatting <https://arxiv.org/pdf/2311.16493>`_.

    .. note::
        **AbsGrad**: If `absgrad` is True, the absolute gradients of the projected
        2D means will be computed during the backward pass, which could be accessed by
        `meta["means2d"].absgrad`. This is an implementation of the paper
        `AbsGS: Recovering Fine Details for 3D Gaussian Splatting <https://arxiv.org/abs/2404.10484>`_,
        which is shown to be more effective for splitting Gaussians during training.

    .. warning::
        This function is currently not differentiable w.r.t. the camera intrinsics `Ks`.

    Args:
        means: The 3D centers of the Gaussians. [N, 3]
        quats: The quaternions of the Gaussians (wxyz convension). It's not required to be normalized. [N, 4]
        scales: The scales of the Gaussians. [N, 3]
        opacities: The opacities of the Gaussians. [N]
        colors: The colors of the Gaussians. [(C,) N, D] or [(C,) N, K, 3] for SH coefficients.
        viewmats: The world-to-cam transformation of the cameras. [C, 4, 4]
        Ks: The camera intrinsics. [C, 3, 3]
        width: The width of the image.
        height: The height of the image.
        near_plane: The near plane for clipping. Default is 0.01.
        far_plane: The far plane for clipping. Default is 1e10.
        radius_clip: Gaussians with 2D radius smaller or equal than this value will be
            skipped. This is extremely helpful for speeding up large scale scenes.
            Default is 0.0.
        eps2d: An epsilon added to the egienvalues of projected 2D covariance matrices.
            This will prevents the projected GS to be too small. For example eps2d=0.3
            leads to minimal 3 pixel unit. Default is 0.3.
        sh_degree: The SH degree to use, which can be smaller than the total
            number of bands. If set, the `colors` should be [(C,) N, K, 3] SH coefficients,
            else the `colors` should [(C,) N, D] post-activation color values. Default is None.
        packed: Whether to use packed mode which is more memory efficient but might or
            might not be as fast. Default is True.
        tile_size: The size of the tiles for rasterization. Default is 16.
            (Note: other values are not tested)
        backgrounds: The background colors. [C, D]. Default is None.
        render_mode: The rendering mode. Supported modes are "RGB", "D", "ED", "RGB+D",
            and "RGB+ED". "RGB" renders the colored image, "D" renders the accumulated depth, and
            "ED" renders the expected depth. Default is "RGB".
        sparse_grad: If true, the gradients for {means, quats, scales} will be stored in
            a COO sparse layout. This can be helpful for saving memory. Default is False.
        absgrad: If true, the absolute gradients of the projected 2D means
            will be computed during the backward pass, which could be accessed by
            `meta["means2d"].absgrad`. Default is False.
        rasterize_mode: The rasterization mode. Supported modes are "classic" and
            "antialiased". Default is "classic".
        channel_chunk: The number of channels to render in one go. Default is 32.
            If the required rendering channels are larger than this value, the rendering
            will be done looply in chunks.
        distributed: Whether to use distributed rendering. Default is False. If True,
            The input Gaussians are expected to be a subset of scene in each rank, and
            the function will collaboratively render the images for all ranks.
        camera_model: The camera model to use. Supported models are "pinhole", "ortho",
            and "fisheye". Default is "pinhole".
        covars: Optional covariance matrices of the Gaussians. If provided, the `quats` and
            `scales` will be ignored. [N, 3, 3], Default is None.

    Returns:
        A tuple:

        **render_colors**: The rendered colors. [C, height, width, X].
        X depends on the `render_mode` and input `colors`. If `render_mode` is "RGB",
        X is D; if `render_mode` is "D" or "ED", X is 1; if `render_mode` is "RGB+D" or
        "RGB+ED", X is D+1.

        **render_alphas**: The rendered alphas. [C, height, width, 1].

        **meta**: A dictionary of intermediate results of the rasterization.

    Examples:

    .. code-block:: python

        >>> # define Gaussians
        >>> means = torch.randn((100, 3), device=device)
        >>> quats = torch.randn((100, 4), device=device)
        >>> scales = torch.rand((100, 3), device=device) * 0.1
        >>> colors = torch.rand((100, 3), device=device)
        >>> opacities = torch.rand((100,), device=device)
        >>> # define cameras
        >>> viewmats = torch.eye(4, device=device)[None, :, :]
        >>> Ks = torch.tensor([
        >>>    [300., 0., 150.], [0., 300., 100.], [0., 0., 1.]], device=device)[None, :, :]
        >>> width, height = 300, 200
        >>> # render
        >>> colors, alphas, meta = rasterization(
        >>>    means, quats, scales, opacities, colors, viewmats, Ks, width, height
        >>> )
        >>> print (colors.shape, alphas.shape)
        torch.Size([1, 200, 300, 3]) torch.Size([1, 200, 300, 1])
        >>> print (meta.keys())
        dict_keys(['camera_ids', 'gaussian_ids', 'radii', 'means2d', 'depths', 'conics',
        'opacities', 'tile_width', 'tile_height', 'tiles_per_gauss', 'isect_ids',
        'flatten_ids', 'isect_offsets', 'width', 'height', 'tile_size'])

    """
    meta = {}

    N = means.shape[0]
    C = viewmats.shape[0]
    device = means.device
    assert means.shape == (N, 3), means.shape
    if covars is None:
        assert quats.shape == (N, 4), quats.shape
        assert scales.shape == (N, 3), scales.shape
    else:
        assert covars.shape == (N, 3, 3), covars.shape
        quats, scales = None, None
        # convert covars from 3x3 matrix to upper-triangular 6D vector
        tri_indices = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
        covars = covars[..., tri_indices[0], tri_indices[1]]
    assert opacities.shape == (N,), opacities.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape
    assert render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED"], render_mode

    def reshape_view(C: int, world_view: torch.Tensor, N_world: list) -> torch.Tensor:
        view_list = list(
            map(
                lambda x: x.split(int(x.shape[0] / C), dim=0),
                world_view.split([C * N_i for N_i in N_world], dim=0),
            )
        )
        return torch.stack([torch.cat(l, dim=0) for l in zip(*view_list)], dim=0)

    if sh_degree is None:
        # treat colors as post-activation values, should be in shape [N, D] or [C, N, D]
        assert (colors.dim() == 2 and colors.shape[0] == N) or (
            colors.dim() == 3 and colors.shape[:2] == (C, N)
        ), colors.shape
        if distributed:
            assert (
                colors.dim() == 2
            ), "Distributed mode only supports per-Gaussian colors."
    else:
        # treat colors as SH coefficients, should be in shape [N, K, 3] or [C, N, K, 3]
        # Allowing for activating partial SH bands
        assert (
            colors.dim() == 3 and colors.shape[0] == N and colors.shape[2] == 3
        ) or (
            colors.dim() == 4 and colors.shape[:2] == (C, N) and colors.shape[3] == 3
        ), colors.shape
        assert (sh_degree + 1) ** 2 <= colors.shape[-2], colors.shape
        if distributed:
            assert (
                colors.dim() == 3
            ), "Distributed mode only supports per-Gaussian colors."

    if absgrad:
        assert not distributed, "AbsGrad is not supported in distributed mode."

    # If in distributed mode, we distribute the projection computation over Gaussians
    # and the rasterize computation over cameras. So first we gather the cameras
    # from all ranks for projection.
    if distributed:
        world_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Gather the number of Gaussians in each rank.
        N_world = all_gather_int32(world_size, N, device=device)

        # Enforce that the number of cameras is the same across all ranks.
        C_world = [C] * world_size
        viewmats, Ks = all_gather_tensor_list(world_size, [viewmats, Ks])

        # Silently change C from local #Cameras to global #Cameras.
        C = len(viewmats)

    # Project Gaussians to 2D. Directly pass in {quats, scales} is faster than precomputing covars.
    proj_results = fully_fused_projection(
        means,
        covars,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d=eps2d,
        packed=packed,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        sparse_grad=sparse_grad,
        calc_compensations=(rasterize_mode == "antialiased"),
        camera_model=camera_model,
    )

    if packed:
        # The results are packed into shape [nnz, ...]. All elements are valid.
        (
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            conics,
            compensations,
        ) = proj_results
        opacities = opacities[gaussian_ids]  # [nnz]
    else:
        # The results are with shape [C, N, ...]. Only the elements with radii > 0 are valid.
        radii, means2d, depths, conics, compensations = proj_results
        opacities = opacities.repeat(C, 1)  # [C, N]
        camera_ids, gaussian_ids = None, None

    if compensations is not None:
        opacities = opacities * compensations

    meta.update(
        {
            # global camera_ids
            "camera_ids": camera_ids,
            # local gaussian_ids
            "gaussian_ids": gaussian_ids,
            "radii": radii,
            "means2d": means2d,
            "depths": depths,
            "conics": conics,
            "opacities": opacities,
        }
    )

    # Turn colors into [C, N, D] or [nnz, D] to pass into rasterize_to_pixels()
    if sh_degree is None:
        # Colors are post-activation values, with shape [N, D] or [C, N, D]
        if packed:
            if colors.dim() == 2:
                # Turn [N, D] into [nnz, D]
                colors = colors[gaussian_ids]
            else:
                # Turn [C, N, D] into [nnz, D]
                colors = colors[camera_ids, gaussian_ids]
        else:
            if colors.dim() == 2:
                # Turn [N, D] into [C, N, D]
                colors = colors.expand(C, -1, -1)
            else:
                # colors is already [C, N, D]
                pass
    else:
        # Colors are SH coefficients, with shape [N, K, 3] or [C, N, K, 3]
        camtoworlds = torch.inverse(viewmats)  # [C, 4, 4]
        if packed:
            dirs = means[gaussian_ids, :] - camtoworlds[camera_ids, :3, 3]  # [nnz, 3]
            masks = radii > 0  # [nnz]
            if colors.dim() == 3:
                # Turn [N, K, 3] into [nnz, 3]
                shs = colors[gaussian_ids, :, :]  # [nnz, K, 3]
            else:
                # Turn [C, N, K, 3] into [nnz, 3]
                shs = colors[camera_ids, gaussian_ids, :, :]  # [nnz, K, 3]
            colors = spherical_harmonics(sh_degree, dirs, shs, masks=masks)  # [nnz, 3]
        else:
            dirs = means[None, :, :] - camtoworlds[:, None, :3, 3]  # [C, N, 3]
            masks = radii > 0  # [C, N]
            if colors.dim() == 3:
                # Turn [N, K, 3] into [C, N, K, 3]
                shs = colors.expand(C, -1, -1, -1)  # [C, N, K, 3]
            else:
                # colors is already [C, N, K, 3]
                shs = colors
            colors = spherical_harmonics(sh_degree, dirs, shs, masks=masks)  # [C, N, 3]
        # make it apple-to-apple with Inria's CUDA Backend.
        colors = torch.clamp_min(colors + 0.5, 0.0)

    # If in distributed mode, we need to scatter the GSs to the destination ranks, based
    # on which cameras they are visible to, which we already figured out in the projection
    # stage.
    if distributed:
        if packed:
            # count how many elements need to be sent to each rank
            cnts = torch.bincount(camera_ids, minlength=C)  # all cameras
            cnts = cnts.split(C_world, dim=0)
            cnts = [cuts.sum() for cuts in cnts]

            # all to all communication across all ranks. After this step, each rank
            # would have all the necessary GSs to render its own images.
            collected_splits = all_to_all_int32(world_size, cnts, device=device)
            (radii,) = all_to_all_tensor_list(
                world_size, [radii], cnts, output_splits=collected_splits
            )
            (means2d, depths, conics, opacities, colors) = all_to_all_tensor_list(
                world_size,
                [means2d, depths, conics, opacities, colors],
                cnts,
                output_splits=collected_splits,
            )

            # before sending the data, we should turn the camera_ids from global to local.
            # i.e. the camera_ids produced by the projection stage are over all cameras world-wide,
            # so we need to turn them into camera_ids that are local to each rank.
            offsets = torch.tensor(
                [0] + C_world[:-1], device=camera_ids.device, dtype=camera_ids.dtype
            )
            offsets = torch.cumsum(offsets, dim=0)
            offsets = offsets.repeat_interleave(torch.stack(cnts))
            camera_ids = camera_ids - offsets

            # and turn gaussian ids from local to global.
            offsets = torch.tensor(
                [0] + N_world[:-1],
                device=gaussian_ids.device,
                dtype=gaussian_ids.dtype,
            )
            offsets = torch.cumsum(offsets, dim=0)
            offsets = offsets.repeat_interleave(torch.stack(cnts))
            gaussian_ids = gaussian_ids + offsets

            # all to all communication across all ranks.
            (camera_ids, gaussian_ids) = all_to_all_tensor_list(
                world_size,
                [camera_ids, gaussian_ids],
                cnts,
                output_splits=collected_splits,
            )

            # Silently change C from global #Cameras to local #Cameras.
            C = C_world[world_rank]

        else:
            # Silently change C from global #Cameras to local #Cameras.
            C = C_world[world_rank]

            # all to all communication across all ranks. After this step, each rank
            # would have all the necessary GSs to render its own images.
            (radii,) = all_to_all_tensor_list(
                world_size,
                [radii.flatten(0, 1)],
                splits=[C_i * N for C_i in C_world],
                output_splits=[C * N_i for N_i in N_world],
            )
            radii = reshape_view(C, radii, N_world)

            (means2d, depths, conics, opacities, colors) = all_to_all_tensor_list(
                world_size,
                [
                    means2d.flatten(0, 1),
                    depths.flatten(0, 1),
                    conics.flatten(0, 1),
                    opacities.flatten(0, 1),
                    colors.flatten(0, 1),
                ],
                splits=[C_i * N for C_i in C_world],
                output_splits=[C * N_i for N_i in N_world],
            )
            means2d = reshape_view(C, means2d, N_world)
            depths = reshape_view(C, depths, N_world)
            conics = reshape_view(C, conics, N_world)
            opacities = reshape_view(C, opacities, N_world)
            colors = reshape_view(C, colors, N_world)

    # Rasterize to pixels
    if render_mode in ["RGB+D", "RGB+ED"]:
        colors = torch.cat((colors, depths[..., None]), dim=-1)
        if backgrounds is not None:
            backgrounds = torch.cat(
                [backgrounds, torch.zeros(C, 1, device=backgrounds.device)], dim=-1
            )
    elif render_mode in ["D", "ED"]:
        colors = depths[..., None]
        if backgrounds is not None:
            backgrounds = torch.zeros(C, 1, device=backgrounds.device)
    else:  # RGB
        pass

    # Identify intersecting tiles
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=packed,
        n_cameras=C,
        camera_ids=camera_ids,
        gaussian_ids=gaussian_ids,
    )
    # print("rank", world_rank, "Before isect_offset_encode")
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    meta.update(
        {
            "tile_width": tile_width,
            "tile_height": tile_height,
            "tiles_per_gauss": tiles_per_gauss,
            "isect_ids": isect_ids,
            "flatten_ids": flatten_ids,
            "isect_offsets": isect_offsets,
            "width": width,
            "height": height,
            "tile_size": tile_size,
            "n_cameras": C,
        }
    )

    # print("rank", world_rank, "Before rasterize_to_pixels")
    if colors.shape[-1] > channel_chunk:
        # slice into chunks
        n_chunks = (colors.shape[-1] + channel_chunk - 1) // channel_chunk
        render_colors, render_alphas = [], []
        for i in range(n_chunks):
            colors_chunk = colors[..., i * channel_chunk : (i + 1) * channel_chunk]
            backgrounds_chunk = (
                backgrounds[..., i * channel_chunk : (i + 1) * channel_chunk]
                if backgrounds is not None
                else None
            )
            render_colors_, render_alphas_ = rasterize_to_pixels(
                means2d,
                conics,
                colors_chunk,
                opacities,
                width,
                height,
                tile_size,
                isect_offsets,
                flatten_ids,
                backgrounds=backgrounds_chunk,
                packed=packed,
                absgrad=absgrad,
            )
            render_colors.append(render_colors_)
            render_alphas.append(render_alphas_)
        render_colors = torch.cat(render_colors, dim=-1)
        render_alphas = render_alphas[0]  # discard the rest
    else:
        render_colors, render_alphas = rasterize_to_pixels(
            means2d,
            conics,
            colors,
            opacities,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=backgrounds,
            packed=packed,
            absgrad=absgrad,
        )
    if render_mode in ["ED", "RGB+ED"]:
        # normalize the accumulated depth to get the expected depth
        render_colors = torch.cat(
            [
                render_colors[..., :-1],
                render_colors[..., -1:] / render_alphas.clamp(min=1e-10),
            ],
            dim=-1,
        )

    return render_colors, render_alphas, meta


def _rasterization(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    colors: Tensor,  # [(C,) N, D] or [(C,) N, K, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    channel_chunk: int = 32,
    batch_per_iter: int = 100,
) -> Tuple[Tensor, Tensor, Dict]:
    """A version of rasterization() that utilies on PyTorch's autograd.

    .. note::
        This function still relies on gsplat's CUDA backend for some computation, but the
        entire differentiable graph is on of PyTorch (and nerfacc) so could use Pytorch's
        autograd for backpropagation.

    .. note::
        This function relies on installing latest nerfacc, via:
        pip install git+https://github.com/nerfstudio-project/nerfacc

    .. note::
        Compared to rasterization(), this function does not support some arguments such as
        `packed`, `sparse_grad` and `absgrad`.
    """
    from gsplat.cuda._torch_impl import (
        _fully_fused_projection,
        _quat_scale_to_covar_preci,
        _rasterize_to_pixels,
    )

    N = means.shape[0]
    C = viewmats.shape[0]
    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert opacities.shape == (N,), opacities.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape
    assert render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED"], render_mode

    if sh_degree is None:
        # treat colors as post-activation values, should be in shape [N, D] or [C, N, D]
        assert (colors.dim() == 2 and colors.shape[0] == N) or (
            colors.dim() == 3 and colors.shape[:2] == (C, N)
        ), colors.shape
    else:
        # treat colors as SH coefficients, should be in shape [N, K, 3] or [C, N, K, 3]
        # Allowing for activating partial SH bands
        assert (
            colors.dim() == 3 and colors.shape[0] == N and colors.shape[2] == 3
        ) or (
            colors.dim() == 4 and colors.shape[:2] == (C, N) and colors.shape[3] == 3
        ), colors.shape
        assert (sh_degree + 1) ** 2 <= colors.shape[-2], colors.shape

    # Project Gaussians to 2D.
    # The results are with shape [C, N, ...]. Only the elements with radii > 0 are valid.
    covars, _ = _quat_scale_to_covar_preci(quats, scales, True, False, triu=False)
    radii, means2d, depths, conics, compensations = _fully_fused_projection(
        means,
        covars,
        viewmats,
        Ks,
        width,
        height,
        eps2d=eps2d,
        near_plane=near_plane,
        far_plane=far_plane,
        calc_compensations=(rasterize_mode == "antialiased"),
    )
    opacities = opacities.repeat(C, 1)  # [C, N]
    camera_ids, gaussian_ids = None, None

    if compensations is not None:
        opacities = opacities * compensations

    # Identify intersecting tiles
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=False,
        n_cameras=C,
        camera_ids=camera_ids,
        gaussian_ids=gaussian_ids,
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    # Turn colors into [C, N, D] or [nnz, D] to pass into rasterize_to_pixels()
    if sh_degree is None:
        # Colors are post-activation values, with shape [N, D] or [C, N, D]
        if colors.dim() == 2:
            # Turn [N, D] into [C, N, D]
            colors = colors.expand(C, -1, -1)
        else:
            # colors is already [C, N, D]
            pass
    else:
        # Colors are SH coefficients, with shape [N, K, 3] or [C, N, K, 3]
        camtoworlds = torch.inverse(viewmats)  # [C, 4, 4]
        dirs = means[None, :, :] - camtoworlds[:, None, :3, 3]  # [C, N, 3]
        masks = radii > 0  # [C, N]
        if colors.dim() == 3:
            # Turn [N, K, 3] into [C, N, 3]
            shs = colors.expand(C, -1, -1, -1)  # [C, N, K, 3]
        else:
            # colors is already [C, N, K, 3]
            shs = colors
        colors = spherical_harmonics(sh_degree, dirs, shs, masks=masks)  # [C, N, 3]
        # make it apple-to-apple with Inria's CUDA Backend.
        colors = torch.clamp_min(colors + 0.5, 0.0)

    # Rasterize to pixels
    if render_mode in ["RGB+D", "RGB+ED"]:
        colors = torch.cat((colors, depths[..., None]), dim=-1)
        if backgrounds is not None:
            backgrounds = torch.cat(
                [backgrounds, torch.zeros(C, 1, device=backgrounds.device)], dim=-1
            )
    elif render_mode in ["D", "ED"]:
        colors = depths[..., None]
        if backgrounds is not None:
            backgrounds = torch.zeros(C, 1, device=backgrounds.device)
    else:  # RGB
        pass
    if colors.shape[-1] > channel_chunk:
        # slice into chunks
        n_chunks = (colors.shape[-1] + channel_chunk - 1) // channel_chunk
        render_colors, render_alphas = [], []
        for i in range(n_chunks):
            colors_chunk = colors[..., i * channel_chunk : (i + 1) * channel_chunk]
            backgrounds_chunk = (
                backgrounds[..., i * channel_chunk : (i + 1) * channel_chunk]
                if backgrounds is not None
                else None
            )
            render_colors_, render_alphas_ = _rasterize_to_pixels(
                means2d,
                conics,
                colors_chunk,
                opacities,
                width,
                height,
                tile_size,
                isect_offsets,
                flatten_ids,
                backgrounds=backgrounds_chunk,
                batch_per_iter=batch_per_iter,
            )
            render_colors.append(render_colors_)
            render_alphas.append(render_alphas_)
        render_colors = torch.cat(render_colors, dim=-1)
        render_alphas = render_alphas[0]  # discard the rest
    else:
        render_colors, render_alphas = _rasterize_to_pixels(
            means2d,
            conics,
            colors,
            opacities,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=backgrounds,
            batch_per_iter=batch_per_iter,
        )
    if render_mode in ["ED", "RGB+ED"]:
        # normalize the accumulated depth to get the expected depth
        render_colors = torch.cat(
            [
                render_colors[..., :-1],
                render_colors[..., -1:] / render_alphas.clamp(min=1e-10),
            ],
            dim=-1,
        )

    meta = {
        "camera_ids": camera_ids,
        "gaussian_ids": gaussian_ids,
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "conics": conics,
        "opacities": opacities,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "tiles_per_gauss": tiles_per_gauss,
        "isect_ids": isect_ids,
        "flatten_ids": flatten_ids,
        "isect_offsets": isect_offsets,
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "n_cameras": C,
    }
    return render_colors, render_alphas, meta


# def rasterization_legacy_wrapper(
#     means: Tensor,  # [N, 3]
#     quats: Tensor,  # [N, 4]
#     scales: Tensor,  # [N, 3]
#     opacities: Tensor,  # [N]
#     colors: Tensor,  # [N, D] or [N, K, 3]
#     viewmats: Tensor,  # [C, 4, 4]
#     Ks: Tensor,  # [C, 3, 3]
#     width: int,
#     height: int,
#     near_plane: float = 0.01,
#     eps2d: float = 0.3,
#     sh_degree: Optional[int] = None,
#     tile_size: int = 16,
#     backgrounds: Optional[Tensor] = None,
#     **kwargs,
# ) -> Tuple[Tensor, Tensor, Dict]:
#     """Wrapper for old version gsplat.

#     .. warning::
#         This function exists for comparison purpose only. So we skip collecting
#         the intermidiate variables, and only return an empty dict.

#     """
#     from gsplat.cuda_legacy._wrapper import (
#         project_gaussians,
#         rasterize_gaussians,
#         spherical_harmonics,
#     )

#     assert eps2d == 0.3, "This is hard-coded in CUDA to be 0.3"
#     C = len(viewmats)

#     render_colors, render_alphas = [], []
#     for cid in range(C):
#         fx, fy = Ks[cid, 0, 0], Ks[cid, 1, 1]
#         cx, cy = Ks[cid, 0, 2], Ks[cid, 1, 2]
#         viewmat = viewmats[cid]

#         means2d, depths, radii, conics, _, num_tiles_hit, _ = project_gaussians(
#             means3d=means,
#             scales=scales,
#             glob_scale=1.0,
#             quats=quats,
#             viewmat=viewmat,
#             fx=fx,
#             fy=fy,
#             cx=cx,
#             cy=cy,
#             img_height=height,
#             img_width=width,
#             block_width=tile_size,
#             clip_thresh=near_plane,
#         )

#         if colors.dim() == 3:
#             c2w = viewmat.inverse()
#             viewdirs = means - c2w[:3, 3]
#             # viewdirs = F.normalize(viewdirs, dim=-1).detach()
#             if sh_degree is None:
#                 sh_degree = int(math.sqrt(colors.shape[1]) - 1)
#             colors = spherical_harmonics(sh_degree, viewdirs, colors)  # [N, 3]

#         background = (
#             backgrounds[cid]
#             if backgrounds is not None
#             else torch.zeros(colors.shape[-1], device=means.device)
#         )

#         render_colors_, render_alphas_ = rasterize_gaussians(
#             xys=means2d,
#             depths=depths,
#             radii=radii,
#             conics=conics,
#             num_tiles_hit=num_tiles_hit,
#             colors=colors,
#             opacity=opacities[..., None],
#             img_height=height,
#             img_width=width,
#             block_width=tile_size,
#             background=background,
#             return_alpha=True,
#         )
#         render_colors.append(render_colors_)
#         render_alphas.append(render_alphas_[..., None])
#     render_colors = torch.stack(render_colors, dim=0)
#     render_alphas = torch.stack(render_alphas, dim=0)
#     return render_colors, render_alphas, {}


def rasterization_inria_wrapper(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    colors: Tensor,  # [N, D] or [N, K, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    backgrounds: Optional[Tensor] = None,
    **kwargs,
) -> Tuple[Tensor, Tensor, Dict]:
    """Wrapper for Inria's rasterization backend.

    .. warning::
        This function exists for comparison purpose only. Only rendered image is
        returned.

    .. warning::
        Inria's CUDA backend has its own LICENSE, so this function should be used with
        the respect to the original LICENSE at:
        https://github.com/graphdeco-inria/diff-gaussian-rasterization

    """
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )

    assert eps2d == 0.3, "This is hard-coded in CUDA to be 0.3"
    C = len(viewmats)
    device = means.device
    channels = colors.shape[-1]

    render_colors = []
    for cid in range(C):
        FoVx = 2 * math.atan(width / (2 * Ks[cid, 0, 0].item()))
        FoVy = 2 * math.atan(height / (2 * Ks[cid, 1, 1].item()))
        tanfovx = math.tan(FoVx * 0.5)
        tanfovy = math.tan(FoVy * 0.5)

        world_view_transform = viewmats[cid].transpose(0, 1)
        projection_matrix = get_projection_matrix(
            znear=near_plane, zfar=far_plane, fovX=FoVx, fovY=FoVy, device=device
        ).transpose(0, 1)
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
        ).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        background = (
            backgrounds[cid]
            if backgrounds is not None
            else torch.zeros(3, device=device)
        )

        raster_settings = GaussianRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=background,
            scale_modifier=1.0,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=0 if sh_degree is None else sh_degree,
            campos=camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means2D = torch.zeros_like(means, requires_grad=True, device=device)

        render_colors_ = []
        for i in range(0, channels, 3):
            _colors = colors[..., i : i + 3]
            if _colors.shape[-1] < 3:
                pad = torch.zeros(
                    _colors.shape[0], 3 - _colors.shape[-1], device=device
                )
                _colors = torch.cat([_colors, pad], dim=-1)
            _render_colors_, radii = rasterizer(
                means3D=means,
                means2D=means2D,
                shs=_colors if colors.dim() == 3 else None,
                colors_precomp=_colors if colors.dim() == 2 else None,
                opacities=opacities[:, None],
                scales=scales,
                rotations=quats,
                cov3D_precomp=None,
            )
            if _colors.shape[-1] < 3:
                _render_colors_ = _render_colors_[:, :, : _colors.shape[-1]]
            render_colors_.append(_render_colors_)
        render_colors_ = torch.cat(render_colors_, dim=-1)

        render_colors_ = render_colors_.permute(1, 2, 0)  # [H, W, 3]

        render_colors.append(render_colors_)
    render_colors = torch.stack(render_colors, dim=0)
    return render_colors, None, {}


###### 2DGS ######
def rasterization_2dgs(
    means: Tensor,
    quats: Tensor,
    scales: Tensor,
    opacities: Tensor,
    colors: Tensor,
    viewmats: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    packed: bool = False,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    distloss: bool = False,
    depth_mode: Literal["expected", "median"] = "expected",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
    """Rasterize a set of 2D Gaussians (N) to a batch of image planes (C).

    This function supports a handful of features, similar to the :func:`rasterization` function.

    .. warning::
        This function is currently not differentiable w.r.t. the camera intrinsics `Ks`.

    Args:
        means: The 3D centers of the Gaussians. [N, 3]
        quats: The quaternions of the Gaussians (wxyz convension). It's not required to be normalized. [N, 4]
        scales: The scales of the Gaussians. [N, 3]
        opacities: The opacities of the Gaussians. [N]
        colors: The colors of the Gaussians. [(C,) N, D] or [(C,) N, K, 3] for SH coefficients.
        viewmats: The world-to-cam transformation of the cameras. [C, 4, 4]
        Ks: The camera intrinsics. [C, 3, 3]
        width: The width of the image.
        height: The height of the image.
        near_plane: The near plane for clipping. Default is 0.01.
        far_plane: The far plane for clipping. Default is 1e10.
        radius_clip: Gaussians with 2D radius smaller or equal than this value will be
            skipped. This is extremely helpful for speeding up large scale scenes.
            Default is 0.0.
        eps2d: An epsilon added to the egienvalues of projected 2D covariance matrices.
            This will prevents the projected GS to be too small. For example eps2d=0.3
            leads to minimal 3 pixel unit. Default is 0.3.
        sh_degree: The SH degree to use, which can be smaller than the total
            number of bands. If set, the `colors` should be [(C,) N, K, 3] SH coefficients,
            else the `colors` should [(C,) N, D] post-activation color values. Default is None.
        packed: Whether to use packed mode which is more memory efficient but might or
            might not be as fast. Default is True.
        tile_size: The size of the tiles for rasterization. Default is 16.
            (Note: other values are not tested)
        backgrounds: The background colors. [C, D]. Default is None.
        render_mode: The rendering mode. Supported modes are "RGB", "D", "ED", "RGB+D",
            and "RGB+ED". "RGB" renders the colored image, "D" renders the accumulated depth, and
            "ED" renders the expected depth. Default is "RGB".
        sparse_grad (Experimental): If true, the gradients for {means, quats, scales} will be stored in
            a COO sparse layout. This can be helpful for saving memory. Default is False.
        absgrad: If true, the absolute gradients of the projected 2D means
            will be computed during the backward pass, which could be accessed by
            `meta["means2d"].absgrad`. Default is False.
        channel_chunk: The number of channels to render in one go. Default is 32.
            If the required rendering channels are larger than this value, the rendering
            will be done looply in chunks.
        distloss: If true, use distortion regularization to get better geometry detail.
        depth_mode: render depth mode. Choose from expected depth and median depth.

    Returns:
        A tuple:

        **render_colors**: The rendered colors. [C, height, width, X].
        X depends on the `render_mode` and input `colors`. If `render_mode` is "RGB",
        X is D; if `render_mode` is "D" or "ED", X is 1; if `render_mode` is "RGB+D" or
        "RGB+ED", X is D+1.

        **render_alphas**: The rendered alphas. [C, height, width, 1].

        **render_normals**: The rendered normals. [C, height, width, 3].

        **surf_normals**: surface normal from depth. [C, height, width, 3]

        **render_distort**: The rendered distortions. [C, height, width, 1].
        L1 version, different from L2 version in 2DGS paper.

        **render_median**: The rendered median depth. [C, height, width, 1].

        **meta**: A dictionary of intermediate results of the rasterization.

    Examples:

    .. code-block:: python

        >>> # define Gaussians
        >>> means = torch.randn((100, 3), device=device)
        >>> quats = torch.randn((100, 4), device=device)
        >>> scales = torch.rand((100, 3), device=device) * 0.1
        >>> colors = torch.rand((100, 3), device=device)
        >>> opacities = torch.rand((100,), device=device)
        >>> # define cameras
        >>> viewmats = torch.eye(4, device=device)[None, :, :]
        >>> Ks = torch.tensor([
        >>>    [300., 0., 150.], [0., 300., 100.], [0., 0., 1.]], device=device)[None, :, :]
        >>> width, height = 300, 200
        >>> # render
        >>> colors, alphas, normals, surf_normals, distort, median_depth, meta = rasterization_2dgs(
        >>>    means, quats, scales, opacities, colors, viewmats, Ks, width, height
        >>> )
        >>> print (colors.shape, alphas.shape)
        torch.Size([1, 200, 300, 3]) torch.Size([1, 200, 300, 1])
        >>> print (normals.shape, surf_normals.shape)
        torch.Size([1, 200, 300, 3]) torch.Size([1, 200, 300, 3])
        >>> print (distort.shape, median_depth.shape)
        torch.Size([1, 200, 300, 1]) torch.Size([1, 200, 300, 1])
        >>> print (meta.keys())
        dict_keys(['camera_ids', 'gaussian_ids', 'radii', 'means2d', 'depths', 'ray_transforms',
        'opacities', 'normals', 'tile_width', 'tile_height', 'tiles_per_gauss', 'isect_ids',
        'flatten_ids', 'isect_offsets', 'width', 'height', 'tile_size', 'n_cameras', 'render_distort',
        'gradient_2dgs'])

    """

    N = means.shape[0]
    C = viewmats.shape[0]
    assert means.shape == (N, 3), means.shape
    assert quats.shape == (N, 4), quats.shape
    assert scales.shape == (N, 3), scales.shape
    assert opacities.shape == (N,), opacities.shape
    assert viewmats.shape == (C, 4, 4), viewmats.shape
    assert Ks.shape == (C, 3, 3), Ks.shape
    assert render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED"], render_mode
    if distloss:
        assert render_mode in [
            "D",
            "ED",
            "RGB+D",
            "RGB+ED",
        ], f"distloss requires depth rendering, render_mode should be D, ED, RGB+D, RGB+ED, but got {render_mode}"

    if sh_degree is None:
        # treat colors as post-activation values
        # colors should be in shape [N, D] or (C, N, D) (silently support)
        assert (colors.dim() == 2 and colors.shape[0] == N) or (
            colors.dim() == 3 and colors.shape[:2] == (C, N)
        ), colors.shape
    else:
        # treat colors as SH coefficients. Allowing for activating partial SH bands
        assert (
            colors.dim() == 3 and colors.shape[0] == N and colors.shape[2] == 3
        ), colors.shape
        assert (sh_degree + 1) ** 2 <= colors.shape[1], colors.shape

    # Compute Ray-Splat intersection transformation.
    proj_results = fully_fused_projection_2dgs(
        means,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        packed,
        sparse_grad,
    )

    if packed:
        (
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            ray_transforms,
            normals,
        ) = proj_results
        opacities = opacities[gaussian_ids]
    else:
        radii, means2d, depths, ray_transforms, normals = proj_results
        opacities = opacities.repeat(C, 1)
        camera_ids, gaussian_ids = None, None

    densify = torch.zeros_like(
        means2d, dtype=means.dtype, requires_grad=True, device="cuda"
    )
    # Identify intersecting tiles
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=packed,
        n_cameras=C,
        camera_ids=camera_ids,
        gaussian_ids=gaussian_ids,
    )
    isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height)

    # TODO: SH also suport N-D.
    # Compute the per-view colors
    if not (
        colors.dim() == 3 and sh_degree is None
    ):  # silently support [C, N, D] color.
        colors = (
            colors[gaussian_ids] if packed else colors.expand(C, *([-1] * colors.dim()))
        )  # [nnz, D] or [C, N, 3]
    else:
        if packed:
            colors = colors[camera_ids, gaussian_ids, :]
    if sh_degree is not None:  # SH coefficients
        camtoworlds = torch.inverse(viewmats)
        if packed:
            dirs = means[gaussian_ids, :] - camtoworlds[camera_ids, :3, 3]
        else:
            dirs = means[None, :, :] - camtoworlds[:, None, :3, 3]
        colors = spherical_harmonics(
            sh_degree, dirs, colors, masks=radii > 0
        )  # [nnz, D] or [C, N, 3]
        # make it apple-to-apple with Inria's CUDA Backend.
        colors = torch.clamp_min(colors + 0.5, 0.0)

    # Rasterize to pixels
    if render_mode in ["RGB+D", "RGB+ED"]:
        colors = torch.cat((colors, depths[..., None]), dim=-1)
        # backgrounds = torch.cat((backgrounds, torch.zeros((C, 1), device="cuda")), dim=-1)
    elif render_mode in ["D", "ED"]:
        colors = depths[..., None]
    else:  # RGB
        pass

    (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
        render_median,
    ) = rasterize_to_pixels_2dgs(
        means2d,
        ray_transforms,
        colors,
        opacities,
        normals,
        densify,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        packed=packed,
        absgrad=absgrad,
        distloss=distloss,
    )
    render_normals_from_depth = None
    if render_mode in ["ED", "RGB+ED"]:
        # normalize the accumulated depth to get the expected depth
        render_colors = torch.cat(
            [
                render_colors[..., :-1],
                render_colors[..., -1:] / render_alphas.clamp(min=1e-10),
            ],
            dim=-1,
        )
    if render_mode in ["RGB+ED", "RGB+D"]:
        # render_depths = render_colors[..., -1:]
        if depth_mode == "expected":
            depth_for_normal = render_colors[..., -1:]
        elif depth_mode == "median":
            depth_for_normal = render_median

        render_normals_from_depth = depth_to_normal(
            depth_for_normal, torch.linalg.inv(viewmats), Ks
        ).squeeze(0)

    meta = {
        "camera_ids": camera_ids,
        "gaussian_ids": gaussian_ids,
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "ray_transforms": ray_transforms,
        "opacities": opacities,
        "normals": normals,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "tiles_per_gauss": tiles_per_gauss,
        "isect_ids": isect_ids,
        "flatten_ids": flatten_ids,
        "isect_offsets": isect_offsets,
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "n_cameras": C,
        "render_distort": render_distort,
        "gradient_2dgs": densify,  # This holds the gradient used for densification for 2dgs
    }

    render_normals = torch.einsum(
        "...ij,...hwj->...hwi", torch.linalg.inv(viewmats)[..., :3, :3], render_normals
    )

    return (
        render_colors,
        render_alphas,
        render_normals,
        render_normals_from_depth,
        render_distort,
        render_median,
        meta,
    )


def rasterization_2dgs_inria_wrapper(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4]
    scales: Tensor,  # [N, 3]
    opacities: Tensor,  # [N]
    colors: Tensor,  # [N, D] or [N, K, 3]
    viewmats: Tensor,  # [C, 4, 4]
    Ks: Tensor,  # [C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    backgrounds: Optional[Tensor] = None,
    depth_ratio: int = 0,
    **kwargs,
) -> Tuple[Tuple, Dict]:
    """Wrapper for 2DGS's rasterization backend which is based on Inria's backend.

    Install the 2DGS rasterization backend from
        https://github.com/hbb1/diff-surfel-rasterization

    Credit to Jeffrey Hu https://github.com/jefequien

    """
    from diff_surfel_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )

    assert eps2d == 0.3, "This is hard-coded in CUDA to be 0.3"
    C = len(viewmats)
    device = means.device
    channels = colors.shape[-1]

    # rasterization from inria does not do normalization internally
    quats = F.normalize(quats, dim=-1)  # [N, 4]
    scales = scales[:, :2]  # [N, 2]

    render_colors = []
    for cid in range(C):
        FoVx = 2 * math.atan(width / (2 * Ks[cid, 0, 0].item()))
        FoVy = 2 * math.atan(height / (2 * Ks[cid, 1, 1].item()))
        tanfovx = math.tan(FoVx * 0.5)
        tanfovy = math.tan(FoVy * 0.5)

        world_view_transform = viewmats[cid].transpose(0, 1)
        projection_matrix = get_projection_matrix(
            znear=near_plane, zfar=far_plane, fovX=FoVx, fovY=FoVy, device=device
        ).transpose(0, 1)
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
        ).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        background = (
            backgrounds[cid]
            if backgrounds is not None
            else torch.zeros(3, device=device)
        )

        raster_settings = GaussianRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=background,
            scale_modifier=1.0,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=0 if sh_degree is None else sh_degree,
            campos=camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means2D = torch.zeros_like(means, requires_grad=True, device=device)

        render_colors_ = []
        for i in range(0, channels, 3):
            _colors = colors[..., i : i + 3]
            if _colors.shape[-1] < 3:
                pad = torch.zeros(
                    _colors.shape[0], 3 - _colors.shape[-1], device=device
                )
                _colors = torch.cat([_colors, pad], dim=-1)
            _render_colors_, radii, allmap = rasterizer(
                means3D=means,
                means2D=means2D,
                shs=_colors if colors.dim() == 3 else None,
                colors_precomp=_colors if colors.dim() == 2 else None,
                opacities=opacities[:, None],
                scales=scales,
                rotations=quats,
                cov3D_precomp=None,
            )
            if _colors.shape[-1] < 3:
                _render_colors_ = _render_colors_[:, :, : _colors.shape[-1]]
            render_colors_.append(_render_colors_)
        render_colors_ = torch.cat(render_colors_, dim=-1)

        render_colors_ = render_colors_.permute(1, 2, 0)  # [H, W, 3]
        render_colors.append(render_colors_)
    render_colors = torch.stack(render_colors, dim=0)

    # additional maps
    allmap = allmap.permute(1, 2, 0).unsqueeze(0)  # [1, H, W, C]
    render_depth_expected = allmap[..., 0:1]
    render_alphas = allmap[..., 1:2]
    render_normal = allmap[..., 2:5]
    render_depth_median = allmap[..., 5:6]
    render_dist = allmap[..., 6:7]

    render_normal = render_normal @ (world_view_transform[:3, :3].T)
    render_depth_expected = render_depth_expected / render_alphas
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # render_depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1;
    # for unbounded scene, use expected depth, i.e., depth_ratio = 0, to reduce disk aliasing.
    render_depth = (
        render_depth_expected * (1 - depth_ratio) + (depth_ratio) * render_depth_median
    )

    normals_surf = depth_to_normal(render_depth, torch.linalg.inv(viewmats), Ks)
    normals_surf = normals_surf * (render_alphas).detach()

    render_colors = torch.cat([render_colors, render_depth], dim=-1)

    meta = {
        "normals_rend": render_normal,
        "normals_surf": normals_surf,
        "render_distloss": render_dist,
        "means2d": means2D,
        "width": width,
        "height": height,
        "radii": radii.unsqueeze(0),
        "n_cameras": C,
        "gaussian_ids": None,
    }
    return (render_colors, render_alphas), meta
