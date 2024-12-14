Fit a COLMAP Capture
========================================

.. currentmodule:: gsplat

The :code:`examples/simple_trainer.py default` script allows you train a 
`3D Gaussian Splatting <https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/>`_ 
model for novel view synthesis, on a COLMAP processed capture. This script follows the
exact same logic with the `official implementation 
<https://github.com/graphdeco-inria/gaussian-splatting>`_ and we have verified it to be 
able to reproduce the metrics in the paper, with much better training speed and memory 
footprint. See :doc:`../tests/eval` for more details on the comparison.

Simply run the script under `examples/`:

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default \
        --data_dir data/360_v2/garden/ --data_factor 4 \
        --result_dir ./results/garden

It also supports a browser based viewer for real-time rendering, powered by 
`Viser <https://github.com/nerfstudio-project/viser>`_ and 
`nerfview <https://github.com/hangg7/nerfview>`_.

.. raw:: html

    <video class="video" autoplay="" loop="" muted="" playsinline="", width="100%", height="auto">
        <source src="../_static/viewer_garden_480p.mp4" type="video/mp4">
        Your browser does not support the video tag.
    </video>