.. _profiling:

Profiling
===================================

.. currentmodule:: gsplat

`gsplat` is developed with efficiency in mind, while also supports trade-offs between 
memory consumption and speed. The :func:`rasterization` function has a few arguments
that would not affect the numerical results but could significantly impact the runtime and 
memory usage, including **packed** and **sparse_grad**:

- **packed**: If True, the rasterization process will be performed in a memory-efficient way,
  in which the intermidate tensors are packed into the sparse tensor layout. This could
  greatly reduce the memory usage when the scene is large and each camera only sees a 
  small portion of the scene. But this also introduces a small runtime overhead. It is recommended
  to set :code:`packed=True` when the scene is large and set :code:`packed=False` when the scene is small
  (relative to the camera frustum).

- **sparse_grad**: This argument is only effective when :code:`packed=True`. If True, in addition to  
  the intermidate tensors, the gradients will also be packed into a 
  `coo sparse tensor <https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html>`_ layout.
  This could further reduce the memory usage when training a large scene as the gradients of
  the Gaussian attributes are usually sparse. Note that in most cases, sparse gradients should be used together
  with a sparse optimizer, such as `torch.optim.SparseAdam <https://pytorch.org/docs/stable/generated/torch.optim.SparseAdam.html>`_. And currently we only supports 
  sparse gradients for part of the Gaussian attributes. See :func:`rasterization` 
  for more details.

Here we provide careful profiling of the performance with the different rasterization backends, along with 
the impact of the above arguments in `gsplat`. "Mem" denotes for
the amount of GPU memory allocated by the forward + backward rasterization process (excluding the input data),
which is calculated using the diff of 
`torch.cuda.max_memory_allocated() <https://pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html>`_ 
before and after the rasterization operation. Evaluations are conducted 
on a 24GB NVIDIA TITAN RTX GPU. (commit 8ea2ea3)


Render RGB Images
-------------------------------------

Batch size 1.

.. code-block:: bash

    python profiling/main.py --backend gsplat gsplat-legacy inria \
        --batch_size 1 --scene_grid 5 --channels 3

===========================  ========  =============  ==========  ==========  ==========
Backend                      Packed    Sparse Grad      Mem (GB)  FPS[fwd]    FPS[bwd]
===========================  ========  =============  ==========  ==========  ==========
gsplat v1.0.0                True      True             **0.35**  132.0       84.8
gsplat v1.0.0                True      False            **0.35**  160.8       88.4
gsplat v1.0.0                False     False                0.48  **171.8**   **97.1**
gsplat v0.1.11               n/a       n/a                  0.62  129.9       91.1
diff-gaussian-rasterization  n/a       n/a                  1.00  164.5       41.5
===========================  ========  =============  ==========  ==========  ==========

Batch size 4.

.. code-block:: bash

    python profiling/main.py --backend gsplat gsplat-legacy inria \
        --batch_size 4 --scene_grid 5 --channels 3

===========================  ========  =============  ==========  ============  ============
Backend                      Packed    Sparse Grad      Mem (GB)  FPS[fwd]      FPS[bwd]
===========================  ========  =============  ==========  ============  ============
gsplat v1.0.0                True      True             **1.41**  42.9 x 4      24.1 x 4
gsplat v1.0.0                True      False            **1.41**  43.9 x 4      24.0 x 4
gsplat v1.0.0                False     False                2.05  **46.1 x 4**  **25.5 x 4**
gsplat v0.1.11               n/a       n/a                  1.83  32.5 x 4      21.6 x 4
diff-gaussian-rasterization  n/a       n/a                  3.91  42.7 x 4      10.1 x 4
===========================  ========  =============  ==========  ============  ============


Render Feature Maps: 32 Channel
------------------------------------------

Batch size 1.

.. code-block:: bash

    python profiling/main.py --backend gsplat gsplat-legacy inria \
        --batch_size 1 --scene_grid 1 --channels 32

===========================  ========  =============  ==========  ==========  ==========
Backend                      Packed    Sparse Grad      Mem (GB)  FPS[fwd]    FPS[bwd]
===========================  ========  =============  ==========  ==========  ==========
gsplat v1.0.0                True      True             **0.61**  124.5       43.6
gsplat v1.0.0                True      False            **0.61**  157.0       **44.3**
gsplat v1.0.0                False     False            **0.61**  **168.4**   44.2
gsplat v0.1.11               n/a       n/a                  0.83  18.3        6.9
diff-gaussian-rasterization  n/a       n/a                  3.66  28.9        5.0
===========================  ========  =============  ==========  ==========  ==========

Batch size 4.

.. code-block:: bash

    python profiling/main.py --backend gsplat gsplat-legacy inria \
        --batch_size 4 --scene_grid 1 --channels 32

===========================  ========  =============  ==========  ============  ============
Backend                      Packed    Sparse Grad      Mem (GB)  FPS[fwd]      FPS[bwd]
===========================  ========  =============  ==========  ============  ============
gsplat v1.0.0                True      True             **2.45**  36.8 x 4      **10.9 x 4**
gsplat v1.0.0                True      False            **2.45**  40.4 x 4      **10.9 x 4**
gsplat v1.0.0                False     False                2.48  **42.1 x 4**  **10.9 x 4**
gsplat v0.1.11               n/a       n/a                  3.28  4.5 x 4       1.7 x 4
diff-gaussian-rasterization  n/a       n/a                 14.52  7.1 x 4       1.2 x 4
===========================  ========  =============  ==========  ============  ============

Render a Large Scene
------------------------------------------

49M Gaussians.

.. code-block:: bash

    python profiling/main.py --backend gsplat gsplat-legacy inria \
        --batch_size 1 --scene_grid 21 --channels 3

===========================  ========  =============  ==========  ==========  ==========
Backend                      Packed    Sparse Grad      Mem (GB)  FPS[fwd]    FPS[bwd]
===========================  ========  =============  ==========  ==========  ==========
gsplat v1.0.0                True      True             **1.44**  53.7        **39.2**
gsplat v1.0.0                True      False                3.08  **62.1**    34.6
gsplat v1.0.0                False     False                5.67  59.2        37.5
gsplat v0.1.11               n/a       n/a                  9.86  23.8        21.1
diff-gaussian-rasterization  n/a       n/a                 10.84  38.3        18.8
===========================  ========  =============  ==========  ==========  ==========

107M Gaussians.

.. code-block:: bash

    python profiling/main.py --backend gsplat gsplat-legacy inria \
        --batch_size 1 --scene_grid 31 --channels 3

===========================  ========  =============  ==========  ==========  ==========
Backend                      Packed    Sparse Grad      Mem (GB)  FPS[fwd]    FPS[bwd]
===========================  ========  =============  ==========  ==========  ==========
gsplat v1.0.0                True      True             **2.31**  45.1        **38.4**
gsplat v1.0.0                True      False                6.11  **47.3**    28.9
gsplat v1.0.0                False     False               12.17  39.3        25.8
gsplat v0.1.11               n/a       n/a                  OOM   OOM         OOM
diff-gaussian-rasterization  n/a       n/a                  OOM   OOM         OOM
===========================  ========  =============  ==========  ==========  ==========