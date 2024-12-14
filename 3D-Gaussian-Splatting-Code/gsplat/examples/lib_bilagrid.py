# # Copyright 2024 Yuehao Wang (https://github.com/yuehaowang). This part of code is borrowed form ["Bilateral Guided Radiance Field Processing"](https://bilarfpro.github.io/).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This is a standalone PyTorch implementation of 3D bilateral grid and CP-decomposed 4D bilateral grid.
To use this module, you can download the "lib_bilagrid.py" file and simply put it in your project directory.

For the details, please check our research project: ["Bilateral Guided Radiance Field Processing"](https://bilarfpro.github.io/).

#### Dependencies

In addition to PyTorch and Numpy, please install [tensorly](https://github.com/tensorly/tensorly).
We have tested this module on Python 3.9.18, PyTorch 2.0.1 (CUDA 11), tensorly 0.8.1, and Numpy 1.25.2.

#### Overview

- For bilateral guided training, you need to construct a `BilateralGrid` instance, which can hold multiple bilateral grids
  for input views. Then, use `slice` function to obtain transformed RGB output and the corresponding affine transformations.

- For bilateral guided finishing, you need to instantiate a `BilateralGridCP4D` object and use `slice4d`.

#### Examples

- Bilateral grid for approximating ISP:
    <a target="_blank" href="https://colab.research.google.com/drive/1tx2qKtsHH9deDDnParMWrChcsa9i7Prr?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

- Low-rank 4D bilateral grid for MR enhancement:
    <a target="_blank" href="https://colab.research.google.com/drive/17YOjQqgWFT3QI1vysOIH494rMYtt_mHL?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


Below is the API reference.

"""

import tensorly as tl
import torch
import torch.nn.functional as F
from torch import nn

tl.set_backend("pytorch")


def color_correct(
    img: torch.Tensor, ref: torch.Tensor, num_iters: int = 5, eps: float = 0.5 / 255
) -> torch.Tensor:
    """
    Warp `img` to match the colors in `ref_img` using iterative color matching.

    This function performs color correction by warping the colors of the input image
    to match those of a reference image. It uses a least squares method to find a
    transformation that maps the input image's colors to the reference image's colors.

    The algorithm iteratively solves a system of linear equations, updating the set of
    unsaturated pixels in each iteration. This approach helps handle non-linear color
    transformations and reduces the impact of clipping.

    Args:
        img (torch.Tensor): Input image to be color corrected. Shape: [..., num_channels]
        ref (torch.Tensor): Reference image to match colors. Shape: [..., num_channels]
        num_iters (int, optional): Number of iterations for the color matching process.
                                   Default is 5.
        eps (float, optional): Small value to determine the range of unclipped pixels.
                               Default is 0.5 / 255.

    Returns:
        torch.Tensor: Color corrected image with the same shape as the input image.

    Note:
        - Both input and reference images should be in the range [0, 1].
        - The function works with any number of channels, but typically used with 3 (RGB).
    """
    if img.shape[-1] != ref.shape[-1]:
        raise ValueError(
            f"img's {img.shape[-1]} and ref's {ref.shape[-1]} channels must match"
        )
    num_channels = img.shape[-1]
    img_mat = img.reshape([-1, num_channels])
    ref_mat = ref.reshape([-1, num_channels])

    def is_unclipped(z):
        return (z >= eps) & (z <= 1 - eps)  # z \in [eps, 1-eps].

    mask0 = is_unclipped(img_mat)
    # Because the set of saturated pixels may change after solving for a
    # transformation, we repeatedly solve a system `num_iters` times and update
    # our estimate of which pixels are saturated.
    for _ in range(num_iters):
        # Construct the left hand side of a linear system that contains a quadratic
        # expansion of each pixel of `img`.
        a_mat = []
        for c in range(num_channels):
            a_mat.append(img_mat[:, c : (c + 1)] * img_mat[:, c:])  # Quadratic term.
        a_mat.append(img_mat)  # Linear term.
        a_mat.append(torch.ones_like(img_mat[:, :1]))  # Bias term.
        a_mat = torch.cat(a_mat, dim=-1)
        warp = []
        for c in range(num_channels):
            # Construct the right hand side of a linear system containing each color
            # of `ref`.
            b = ref_mat[:, c]
            # Ignore rows of the linear system that were saturated in the input or are
            # saturated in the current corrected color estimate.
            mask = mask0[:, c] & is_unclipped(img_mat[:, c]) & is_unclipped(b)
            ma_mat = torch.where(mask[:, None], a_mat, torch.zeros_like(a_mat))
            mb = torch.where(mask, b, torch.zeros_like(b))
            w = torch.linalg.lstsq(ma_mat, mb, rcond=-1)[0]
            assert torch.all(torch.isfinite(w))
            warp.append(w)
        warp = torch.stack(warp, dim=-1)
        # Apply the warp to update img_mat.
        img_mat = torch.clip(torch.matmul(a_mat, warp), 0, 1)
    corrected_img = torch.reshape(img_mat, img.shape)
    return corrected_img


def bilateral_grid_tv_loss(model, config):
    """Computes total variations of bilateral grids."""
    total_loss = 0.0

    for bil_grids in model.bil_grids:
        total_loss += config.bilgrid_tv_loss_mult * total_variation_loss(
            bil_grids.grids
        )

    return total_loss


def color_affine_transform(affine_mats, rgb):
    """Applies color affine transformations.

    Args:
        affine_mats (torch.Tensor): Affine transformation matrices. Supported shape: $(..., 3, 4)$.
        rgb  (torch.Tensor): Input RGB values. Supported shape: $(..., 3)$.

    Returns:
        Output transformed colors of shape $(..., 3)$.
    """
    return (
        torch.matmul(affine_mats[..., :3], rgb.unsqueeze(-1)).squeeze(-1)
        + affine_mats[..., 3]
    )


def _num_tensor_elems(t):
    return max(torch.prod(torch.tensor(t.size()[1:]).float()).item(), 1.0)


def total_variation_loss(x):  # noqa: F811
    """Returns total variation on multi-dimensional tensors.

    Args:
        x (torch.Tensor): The input tensor with shape $(B, C, ...)$, where $B$ is the batch size and $C$ is the channel size.
    """
    batch_size = x.shape[0]
    tv = 0
    for i in range(2, len(x.shape)):
        n_res = x.shape[i]
        idx1 = torch.arange(1, n_res, device=x.device)
        idx2 = torch.arange(0, n_res - 1, device=x.device)
        x1 = x.index_select(i, idx1)
        x2 = x.index_select(i, idx2)
        count = _num_tensor_elems(x1)
        tv += torch.pow((x1 - x2), 2).sum() / count
    return tv / batch_size


def slice(bil_grids, xy, rgb, grid_idx):
    """Slices a batch of 3D bilateral grids by pixel coordinates `xy` and gray-scale guidances of pixel colors `rgb`.

    Supports 2-D, 3-D, and 4-D input shapes. The first dimension of the input is the batch size
    and the last dimension is 2 for `xy`, 3 for `rgb`, and 1 for `grid_idx`.

    The return value is a dictionary containing the affine transformations `affine_mats` sliced from bilateral grids and
    the output color `rgb_out` after applying the afffine transformations.

    In the 2-D input case, `xy` is a $(N, 2)$ tensor, `rgb` is  a $(N, 3)$ tensor, and `grid_idx` is a $(N, 1)$ tensor.
    Then `affine_mats[i]` can be obtained via slicing the bilateral grid indexed at `grid_idx[i]` by `xy[i, :]` and `rgb2gray(rgb[i, :])`.
    For 3-D and 4-D input cases, the behavior of indexing bilateral grids and coordinates is the same with the 2-D case.

    .. note::
        This function can be regarded as a wrapper of `color_affine_transform` and `BilateralGrid` with a slight performance improvement.
        When `grid_idx` contains a unique index, only a single bilateral grid will used during the slicing. In this case, this function will not
        perform tensor indexing to avoid data copy and extra memory
        (see [this](https://discuss.pytorch.org/t/does-indexing-a-tensor-return-a-copy-of-it/164905)).

    Args:
        bil_grids (`BilateralGrid`): An instance of $N$ bilateral grids.
        xy (torch.Tensor): The x-y coordinates of shape $(..., 2)$ in the range of $[0,1]$.
        rgb (torch.Tensor): The RGB values of shape $(..., 3)$ for computing the guidance coordinates, ranging in $[0,1]$.
        grid_idx (torch.Tensor): The indices of bilateral grids for each slicing. Shape: $(..., 1)$.

    Returns:
        A dictionary with keys and values as follows:
        ```
        {
            "rgb": Transformed RGB colors. Shape: (..., 3),
            "rgb_affine_mats": The sliced affine transformation matrices from bilateral grids. Shape: (..., 3, 4)
        }
        ```
    """

    sh_ = rgb.shape

    grid_idx_unique = torch.unique(grid_idx)
    if len(grid_idx_unique) == 1:
        # All pixels are from a single view.
        grid_idx = grid_idx_unique  # (1,)
        xy = xy.unsqueeze(0)  # (1, ..., 2)
        rgb = rgb.unsqueeze(0)  # (1, ..., 3)
    else:
        # Pixels are randomly sampled from different views.
        if len(grid_idx.shape) == 4:
            grid_idx = grid_idx[:, 0, 0, 0]  # (chunk_size,)
        elif len(grid_idx.shape) == 3:
            grid_idx = grid_idx[:, 0, 0]  # (chunk_size,)
        elif len(grid_idx.shape) == 2:
            grid_idx = grid_idx[:, 0]  # (chunk_size,)
        else:
            raise ValueError(
                "The input to bilateral grid slicing is not supported yet."
            )

    affine_mats = bil_grids(xy, rgb, grid_idx)
    rgb = color_affine_transform(affine_mats, rgb)

    return {
        "rgb": rgb.reshape(*sh_),
        "rgb_affine_mats": affine_mats.reshape(
            *sh_[:-1], affine_mats.shape[-2], affine_mats.shape[-1]
        ),
    }


class BilateralGrid(nn.Module):
    """Class for 3D bilateral grids.

    Holds one or more than one bilateral grids.
    """

    def __init__(self, num, grid_X=16, grid_Y=16, grid_W=8):
        """
        Args:
            num (int): The number of bilateral grids (i.e., the number of views).
            grid_X (int): Defines grid width $W$.
            grid_Y (int): Defines grid height $H$.
            grid_W (int): Defines grid guidance dimension $L$.
        """
        super(BilateralGrid, self).__init__()

        self.grid_width = grid_X
        """Grid width. Type: int."""
        self.grid_height = grid_Y
        """Grid height. Type: int."""
        self.grid_guidance = grid_W
        """Grid guidance dimension. Type: int."""

        # Initialize grids.
        grid = self._init_identity_grid()
        self.grids = nn.Parameter(grid.tile(num, 1, 1, 1, 1))  # (N, 12, L, H, W)
        """ A 5-D tensor of shape $(N, 12, L, H, W)$."""

        # Weights of BT601 RGB-to-gray.
        self.register_buffer("rgb2gray_weight", torch.Tensor([[0.299, 0.587, 0.114]]))
        self.rgb2gray = lambda rgb: (rgb @ self.rgb2gray_weight.T) * 2.0 - 1.0
        """ A function that converts RGB to gray-scale guidance in $[-1, 1]$."""

    def _init_identity_grid(self):
        grid = torch.tensor(
            [
                1.0,
                0,
                0,
                0,
                0,
                1.0,
                0,
                0,
                0,
                0,
                1.0,
                0,
            ]
        ).float()
        grid = grid.repeat(
            [self.grid_guidance * self.grid_height * self.grid_width, 1]
        )  # (L * H * W, 12)
        grid = grid.reshape(
            1, self.grid_guidance, self.grid_height, self.grid_width, -1
        )  # (1, L, H, W, 12)
        grid = grid.permute(0, 4, 1, 2, 3)  # (1, 12, L, H, W)
        return grid

    def tv_loss(self):
        """Computes and returns total variation loss on the bilateral grids."""
        return total_variation_loss(self.grids)

    def forward(self, grid_xy, rgb, idx=None):
        """Bilateral grid slicing. Supports 2-D, 3-D, 4-D, and 5-D input.
        For the 2-D, 3-D, and 4-D cases, please refer to `slice`.
        For the 5-D cases, `idx` will be unused and the first dimension of `xy` should be
        equal to the number of bilateral grids. Then this function becomes PyTorch's
        [`F.grid_sample`](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html).

        Args:
            grid_xy (torch.Tensor): The x-y coordinates in the range of $[0,1]$.
            rgb (torch.Tensor): The RGB values in the range of $[0,1]$.
            idx (torch.Tensor): The bilateral grid indices.

        Returns:
            Sliced affine matrices of shape $(..., 3, 4)$.
        """

        grids = self.grids
        input_ndims = len(grid_xy.shape)
        assert len(rgb.shape) == input_ndims

        if input_ndims > 1 and input_ndims < 5:
            # Convert input into 5D
            for i in range(5 - input_ndims):
                grid_xy = grid_xy.unsqueeze(1)
                rgb = rgb.unsqueeze(1)
            assert idx is not None
        elif input_ndims != 5:
            raise ValueError(
                "Bilateral grid slicing only takes either 2D, 3D, 4D and 5D inputs"
            )

        grids = self.grids
        if idx is not None:
            grids = grids[idx]
        assert grids.shape[0] == grid_xy.shape[0]

        # Generate slicing coordinates.
        grid_xy = (grid_xy - 0.5) * 2  # Rescale to [-1, 1].
        grid_z = self.rgb2gray(rgb)

        # print(grid_xy.shape, grid_z.shape)
        # exit()
        grid_xyz = torch.cat([grid_xy, grid_z], dim=-1)  # (N, m, h, w, 3)

        affine_mats = F.grid_sample(
            grids, grid_xyz, mode="bilinear", align_corners=True, padding_mode="border"
        )  # (N, 12, m, h, w)
        affine_mats = affine_mats.permute(0, 2, 3, 4, 1)  # (N, m, h, w, 12)
        affine_mats = affine_mats.reshape(
            *affine_mats.shape[:-1], 3, 4
        )  # (N, m, h, w, 3, 4)

        for _ in range(5 - input_ndims):
            affine_mats = affine_mats.squeeze(1)

        return affine_mats


def slice4d(bil_grid4d, xyz, rgb):
    """Slices a 4D bilateral grid by point coordinates `xyz` and gray-scale guidances of radiance colors `rgb`.

    Args:
        bil_grid4d (`BilateralGridCP4D`): The input 4D bilateral grid.
        xyz (torch.Tensor): The xyz coordinates with shape $(..., 3)$.
        rgb (torch.Tensor): The RGB values with shape $(..., 3)$.

    Returns:
        A dictionary with keys and values as follows:
        ```
        {
            "rgb": Transformed radiance RGB colors. Shape: (..., 3),
            "rgb_affine_mats": The sliced affine transformation matrices from the 4D bilateral grid. Shape: (..., 3, 4)
        }
        ```
    """

    affine_mats = bil_grid4d(xyz, rgb)
    rgb = color_affine_transform(affine_mats, rgb)

    return {"rgb": rgb, "rgb_affine_mats": affine_mats}


class _ScaledTanh(nn.Module):
    def __init__(self, s=2.0):
        super().__init__()
        self.scaler = s

    def forward(self, x):
        return torch.tanh(self.scaler * x)


class BilateralGridCP4D(nn.Module):
    """Class for low-rank 4D bilateral grids."""

    def __init__(
        self,
        grid_X=16,
        grid_Y=16,
        grid_Z=16,
        grid_W=8,
        rank=5,
        learn_gray=True,
        gray_mlp_width=8,
        gray_mlp_depth=2,
        init_noise_scale=1e-6,
        bound=2.0,
    ):
        """
        Args:
            grid_X (int): Defines grid width.
            grid_Y (int): Defines grid height.
            grid_Z (int): Defines grid depth.
            grid_W (int): Defines grid guidance dimension.
            rank (int): Rank of the 4D bilateral grid.
            learn_gray (bool): If True, an MLP will be learned to convert RGB colors to gray-scale guidances.
            gray_mlp_width (int): The MLP width for learnable guidance.
            gray_mlp_depth (int): The number of MLP layers for learnable guidance.
            init_noise_scale (float): The noise scale of the initialized factors.
            bound (float): The bound of the xyz coordinates.
        """
        super(BilateralGridCP4D, self).__init__()

        self.grid_X = grid_X
        """Grid width. Type: int."""
        self.grid_Y = grid_Y
        """Grid height. Type: int."""
        self.grid_Z = grid_Z
        """Grid depth. Type: int."""
        self.grid_W = grid_W
        """Grid guidance dimension. Type: int."""
        self.rank = rank
        """Rank of the 4D bilateral grid. Type: int."""
        self.learn_gray = learn_gray
        """Flags of learnable guidance is used. Type: bool."""
        self.gray_mlp_width = gray_mlp_width
        """The MLP width for learnable guidance. Type: int."""
        self.gray_mlp_depth = gray_mlp_depth
        """The MLP depth for learnable guidance. Type: int."""
        self.init_noise_scale = init_noise_scale
        """The noise scale of the initialized factors. Type: float."""
        self.bound = bound
        """The bound of the xyz coordinates. Type: float."""

        self._init_cp_factors_parafac()

        self.rgb2gray = None
        """ A function that converts RGB to gray-scale guidances in $[-1, 1]$.
        If `learn_gray` is True, this will be an MLP network."""

        if self.learn_gray:

            def rgb2gray_mlp_linear(layer):
                return nn.Linear(
                    self.gray_mlp_width,
                    self.gray_mlp_width if layer < self.gray_mlp_depth - 1 else 1,
                )

            def rgb2gray_mlp_actfn(_):
                return nn.ReLU(inplace=True)

            self.rgb2gray = nn.Sequential(
                *(
                    [nn.Linear(3, self.gray_mlp_width)]
                    + [
                        nn_module(layer)
                        for layer in range(1, self.gray_mlp_depth)
                        for nn_module in [rgb2gray_mlp_actfn, rgb2gray_mlp_linear]
                    ]
                    + [_ScaledTanh(2.0)]
                )
            )
        else:
            # Weights of BT601/BT470 RGB-to-gray.
            self.register_buffer(
                "rgb2gray_weight", torch.Tensor([[0.299, 0.587, 0.114]])
            )
            self.rgb2gray = lambda rgb: (rgb @ self.rgb2gray_weight.T) * 2.0 - 1.0

    def _init_identity_grid(self):
        grid = torch.tensor(
            [
                1.0,
                0,
                0,
                0,
                0,
                1.0,
                0,
                0,
                0,
                0,
                1.0,
                0,
            ]
        ).float()
        grid = grid.repeat([self.grid_W * self.grid_Z * self.grid_Y * self.grid_X, 1])
        grid = grid.reshape(self.grid_W, self.grid_Z, self.grid_Y, self.grid_X, -1)
        grid = grid.permute(4, 0, 1, 2, 3)  # (12, grid_W, grid_Z, grid_Y, grid_X)
        return grid

    def _init_cp_factors_parafac(self):
        # Initialize identity grids.
        init_grids = self._init_identity_grid()
        # Random noises are added to avoid singularity.
        init_grids = torch.randn_like(init_grids) * self.init_noise_scale + init_grids
        from tensorly.decomposition import parafac

        # Initialize grid CP factors
        _, facs = parafac(init_grids.clone().detach(), rank=self.rank)

        self.num_facs = len(facs)

        self.fac_0 = nn.Linear(facs[0].shape[0], facs[0].shape[1], bias=False)
        self.fac_0.weight = nn.Parameter(facs[0])  # (12, rank)

        for i in range(1, self.num_facs):
            fac = facs[i].T  # (rank, grid_size)
            fac = fac.view(1, fac.shape[0], fac.shape[1], 1)  # (1, rank, grid_size, 1)
            self.register_buffer(f"fac_{i}_init", fac)

            fac_resid = torch.zeros_like(fac)
            self.register_parameter(f"fac_{i}", nn.Parameter(fac_resid))

    def tv_loss(self):
        """Computes and returns total variation loss on the factors of the low-rank 4D bilateral grids."""

        total_loss = 0
        for i in range(1, self.num_facs):
            fac = self.get_parameter(f"fac_{i}")
            total_loss += total_variation_loss(fac)

        return total_loss

    def forward(self, xyz, rgb):
        """Low-rank 4D bilateral grid slicing.

        Args:
            xyz (torch.Tensor): The xyz coordinates with shape $(..., 3)$.
            rgb (torch.Tensor): The corresponding RGB values with shape $(..., 3)$.

        Returns:
            Sliced affine matrices with shape $(..., 3, 4)$.
        """
        sh_ = xyz.shape
        xyz = xyz.reshape(-1, 3)  # flatten (N, 3)
        rgb = rgb.reshape(-1, 3)  # flatten (N, 3)

        xyz = xyz / self.bound
        assert self.rgb2gray is not None
        gray = self.rgb2gray(rgb)
        xyzw = torch.cat([xyz, gray], dim=-1)  # (N, 4)
        xyzw = xyzw.transpose(0, 1)  # (4, N)
        coords = torch.stack([torch.zeros_like(xyzw), xyzw], dim=-1)  # (4, N, 2)
        coords = coords.unsqueeze(1)  # (4, 1, N, 2)

        coef = 1.0
        for i in range(1, self.num_facs):
            fac = self.get_parameter(f"fac_{i}") + self.get_buffer(f"fac_{i}_init")
            coef = coef * F.grid_sample(
                fac, coords[[i - 1]], align_corners=True, padding_mode="border"
            )  # [1, rank, 1, N]
        coef = coef.squeeze([0, 2]).transpose(0, 1)  # (N, rank) #type: ignore
        mat = self.fac_0(coef)
        return mat.reshape(*sh_[:-1], 3, 4)
