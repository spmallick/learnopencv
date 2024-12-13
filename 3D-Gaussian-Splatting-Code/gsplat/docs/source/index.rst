gsplat
===================================

.. image:: assets/training.gif
    :width: 800
    :alt: Example training image

Overview
--------

*gsplat* is an open-source library for CUDA-accelerated differentiable rasterization of 
3D gaussians with Python bindings. It is inspired by the SIGGRAPH paper "3D Gaussian Splatting for 
Real-Time Rendering of Radiance Fields" :cite:p:`kerbl3Dgaussians`, but we've made *gsplat* even 
faster, more memory efficient, and with a growing list of new features!

* *gsplat* is developed with efficiency in mind. Comparing to the `official implementation <https://github.com/graphdeco-inria/gaussian-splatting>`_, 
  *gsplat* enables up to **4x less training memory footprint**, and up to **15% less training time** on Mip-NeRF 360 captures, and potential more on larger scenes. See :doc:`tests/eval` for details.

* *gsplat* is designed to **support extremely large scene rendering**, which is magnitudes 
  faster than the official CUDA backend `diff-gaussian-rasterization <https://github.com/graphdeco-inria/diff-gaussian-rasterization>`_. See :doc:`examples/large_scale` for an example.

* *gsplat* offers many extra features, including **batch rasterization**,  
  **N-D feature rendering**, **depth rendering**, **sparse gradient**, 
  **multi-GPU distributed rasterization**
  etc. See :doc:`apis/rasterization` for details.

* *gsplat* is equipped with the **latest and greatest** 3D Gaussian Splatting techniques, 
  including `absgrad <https://ty424.github.io/AbsGS.github.io/>`_, 
  `anti-aliasing <https://niujinshuchong.github.io/mip-splatting/>`_,
  `3DGS-MCMC <https://ubc-vision.github.io/3dgs-mcmc/>`_ etc. And more to come!


.. raw:: html
   
   <div style="position:relative; padding-bottom:56.25%; height:0; width:100%">
      <iframe style="position:absolute; top:0; left:0; width:100%; height:100%" src="https://www.youtube.com/embed/d5vOUYm5k34?si=fQBdwPgoi0FZOSyD" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
   </div>


Installation
------------

*gsplat* is available on PyPI and can be installed with pip:

.. code-block:: bash

    pip install gsplat

To get the latest features, it can also be installed from source:

.. code-block:: bash

    pip install git+https://github.com/nerfstudio-project/gsplat

Contributing
------------

This repository was born from the curiosity of people on the Nerfstudio team trying to 
understand a new rendering technique. We welcome contributions of any kind and are open 
to feedback, bug-reports, and improvements to help expand the capabilities of this software.

This project is developed by the following wonderful contributors (unordered):

- `Angjoo Kanazawa <https://people.eecs.berkeley.edu/~kanazawa/>`_ (UC Berkeley): Advisor.
- `Matthew Tancik <https://www.matthewtancik.com/about-me>`_ (Luma AI): Advisor.
- `Vickie Ye <https://people.eecs.berkeley.edu/~vye/>`_ (UC Berkeley): Project lead. v0.1 lead.
- `Matias Turkulainen <https://maturk.github.io/>`_ (Aalto University): Core developer.
- `Ruilong Li <https://www.liruilong.cn/>`_ (UC Berkeley): Core developer. v1.0 lead.
- `Justin Kerr <https://kerrj.github.io/>`_ (UC Berkeley): Core developer.
- `Brent Yi <https://github.com/brentyi>`_ (UC Berkeley): Core developer.
- `Zhuoyang Pan <https://panzhy.com/>`_ (ShanghaiTech University): Core developer.
- `Jianbo Ye <http://www.jianboye.org/>`_ (Amazon): Core developer.


Links
-----

.. toctree::    
   :glob:
   :maxdepth: 1
   :caption: Examples

   examples/*


.. toctree::    
   :glob:
   :maxdepth: 1
   :caption: Conventions

   conventions/*


.. toctree::    
   :glob:
   :maxdepth: 1
   :caption: Python API

   apis/*


.. toctree::    
   :glob:
   :maxdepth: 1
   :caption: Tests

   tests/*

.. toctree::    
   :glob:
   :maxdepth: 1
   :caption: Migration

   migration/*


Citations
-------------
.. bibliography::
   :style: unsrt
   :filter: docname in docnames
