Migrate from gsplat v0.1.11
===================================

.. currentmodule:: gsplat

`v1.0.0` is a major release that includes a huge API change. So this document will help 
you to migrate to `v0.1.11` from `v1.0.0` and enjoys the latest and greatest. The APIs 
of `v0.1.11` are available at `here <https://docs.gsplat.studio/versions/0.1.11/>`_. Note
you can still call the old APIs in `v1.0.0` but they are deprecated and will be removed in
the future.

Below we demonstrate the API changes on a couple of use cases.

Basic Usage
------------------

In `v0.1.11`, a basic rasterization workflow is:

.. code-block:: python
    
    from gsplat import project_gaussians, rasterize_gaussians

    means2d, depths, radii, conics, _, num_tiles_hit, _ = project_gaussians(
        means3d=means, # [N, 3]
        scales=scales, # [N, 3]
        glob_scale=1.0, 
        quats=quats, # [N, 4]
        viewmat=viewmat, # [4, 4]
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        img_height=height,
        img_width=width,
        block_width=tile_size,
        clip_thresh=near_plane,
    )

    renders, alphas = rasterize_gaussians(
        xys=means2d, # [N, 2]
        depths=depths, # [N, 1]
        radii=radii, # [N, 1]
        conics=conics, # [N, 3]
        num_tiles_hit=num_tiles_hit,
        colors=colors, # [N, 3]
        opacity=opacities, # [N, 1]
        img_height=height,
        img_width=width,
        block_width=tile_size,
        background=background, # [3]
        return_alpha=True,
    )

In `v1.0.0`, the equivalent code is:

.. code-block:: python

    from gsplat import rasterization

    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0., 0., 1.]], device=device)
    renders, alphas, meta = rasterization(
        means=means, # [N, 3]
        quats=quats, # [N, 4]
        scales=scales, # [N, 3]
        opacities=opacities.squeeze(-1), # [N]
        colors=colors, # [N, 3]
        viewmats=viewmat[None, ...], # [1, 4, 4]
        Ks=K[None, ...], # [1, 3, 3]
        width=width,
        height=height,
    )
    renders = renders[0]  # [height, width, 3]
    alphas = alphas[0]  # [height, width]

    # The intermediate results from fully_fused_projection can be accessed via meta, e.g.,
    means2d = meta['means2d'][0]  # [N, 2]

Color as Spherical Harmonics
----------------------------

In `v0.1.11`, user needs to explicitly converting spherical harmonics coefficients to RGB,
before passing it into `rasterize_gaussians`:

.. code-block:: python

    from gsplat import spherical_harmonics

    sh_degree: int = ... # the amount of bands activated
    sh_coeffs: Tensor = ... # [N, K, 3]

    viewdirs = means - camtoworld[:3, 3] # [N, 3]
    colors = spherical_harmonics(sh_degree, viewdirs, colors)  # [N, 3]

In `v1.0.0`, the SH to RGB conversion is handled automatically in :func:`rasterization`. 
The equivalent code is:

.. code-block:: python

    from gsplat import rasterization

    sh_degree: int = ... # the amount of bands activated
    sh_coeffs: Tensor = ... # [N, K, 3]

    renders, alphas, meta = rasterization(
        ...
        colors=sh_coeffs, # [N, K, 3]
        sh_degree=sh_degree,
        ...
    )


Depth Rendering
----------------------------

In `v0.1.11`, rendering the depth map could be achieved by passing the projected `depths` 
from `project_gaussians` to `rasterize_gaussians`:

.. code-block:: python
    
    from gsplat import project_gaussians, rasterize_gaussians

    ..., depths, ... = project_gaussians(...)

    accumulated_depth, alphas = rasterize_gaussians(
        ...
        colors=depths, # [N, 1]
        return_alpha=True,
        ...
    )
    expected_depth = accumulated_depth / alphas.clamp_min(1e-10)

In `v1.0.0`, the depth rendering is simplified to a single argument. The equivalent code is:

.. code-block:: python

    from gsplat import rasterization

    expected_depth, alphas, meta = rasterization(
        ...
        render_mode="ED", # render expected depth only
        ...
    )

    accumulated_depth, alphas, meta = rasterization(
        ...
        render_mode="D", # render accumulated depth only
        ...
    )

    rgbd, alphas, meta = rasterization(
        ...
        render_mode="RGB+ED", # render color with expected depth
        ...
    )

