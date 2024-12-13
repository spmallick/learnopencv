Compression
===================================

.. currentmodule:: gsplat

`gsplat` provides handy APIs for compressing and decompressing the Gaussian parameters,
which can significantly reduce the storage / streaming cost. For example, using :class:`PngCompression`,
1 million Gaussians that are stored in 236 MB can be compressed to 16.5 MB, with only 0.5dB
PSNR loss (29.18dB to 28.65dB).

Similar to the :doc:`strategy`, our compression APIs also has a specific expectation on 
the format of the Gaussian parameters. It is designed to work with the Gaussians defined 
as either a Dict of
`torch.Tensor <https://pytorch.org/docs/stable/tensors.html>`_
with at least the following keys: {"means", "scales", "quats", "opacities", "sh0", "shN"}. 
On top of these attributes, an arbitrary number of extra attributes are supported.

The following code snippet is an example of how to use the compression approach in `gsplat`:

.. code-block:: python

    from gsplat import PngCompression

    splats: Dict[str, Tensor] = {
        "means": Tensor(N, 3), "scales": Tensor(N, 3), "quats": Tensor(N, 4), "opacities": Tensor(N),
        "sh0": Tensor(N, 1, 3), "shN": Tensor(N, 24, 3), "features1": Tensor(N, 128), "features2": Tensor(N, 64),
    }

    compression_method = PngCompression()
    # run compression and save the compressed files to compress_dir
    compression_method.compress(compress_dir, splats)
    # decompress the compressed files
    splats_c = compression_method.decompress(compress_dir)

Below is the API for the compression approaches we support in `gsplat`:

.. autoclass:: PngCompression
    :members:
