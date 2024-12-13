Rasterization
===================================

3DGS
------

.. currentmodule:: gsplat

Given a set of 3D gaussians parametrized by means :math:`\mu \in \mathbb{R}^3`, covariances 
:math:`\Sigma \in \mathbb{R}^{3 \times 3}`, colors :math:`c`, and opacities :math:`o`, we first 
compute their projected means :math:`\mu' \in \mathbb{R}^2` and covariances 
:math:`\Sigma' \in \mathbb{R}^{2 \times 2}` on the image planes. Then we sort each gaussian such 
that all gaussians within the bounds of a tile are grouped and sorted by increasing 
depth :math:`z`, and then render each pixel within the tile with alpha-compositing. 

Note, the 3D covariances are reparametrized with a scaling matrix 
:math:`S = \text{diag}(\mathbf{s}) \in \mathbb{R}^{3 \times 3}` represented by a 
scale vector :math:`s \in \mathbb{R}^3`, and a rotation matrix 
:math:`R \in \mathbb{R}^{3 \times 3}` represented by a rotation 
quaternion :math:`q \in \mathcal{R}^4`:

.. math::
   
   \Sigma = RSS^{T}R^{T}

The projection of 3D Gaussians is approximated with the Jacobian of the perspective 
projection equation:

.. math::

    J = \begin{bmatrix}
        f_{x}/z & 0 & -f_{x} t_{x}/z^{2} \\
        0 & f_{y}/z & -f_{y} t_{y}/z^{2} \\
        0 & 0 & 0
    \end{bmatrix}

.. math::

    \Sigma' = J W \Sigma W^{T} J^{T}

Where :math:`[W | t]` is the world-to-camera transformation matrix, and :math:`f_{x}, f_{y}`
are the focal lengths of the camera.

.. autofunction:: rasterization

2DGS
------

Given a set of 2D gaussians parametrized by means :math:`\mu \in \mathbb{R}^3`, two principal tangent vectors 
embedded as the first two columns of a rotation matrix :math:`R \in \mathbb{R}^{3\times3}`, and a scale matrix :math:`S \in R^{3\times3}`
representing the scaling along the two principal tangential directions, we first transforms pixels into splats' local tangent frame 
by :math:`(WH)^{-1} \in \mathbb{R}^{4\times4}` and compute weights via ray-splat intersection. Then we follow the sort and rendering similar to 3DGS.  

Note that H is the  transformation from splat's local tangent plane :math:`\{u, v\}` into world space

.. math:: 
    
    H = \begin{bmatrix}
        RS & \mu \\
        0 & 1
    \end{bmatrix}

and :math:`W \in \mathbb{R}^{4\times4}` is the transformation matrix from world space to image space.


Splatting is done via ray-splat plane intersection. Each pixel is considered as a x-plane :math:`h_{x}=(-1, 0, 0, x)^{T}`
and a y-plane :math:`h_{y}=(0, -1, 0, y)^{T}`, and the intersection between a splat and the pixel :math:`p=(x, y)` is defined 
as the intersection bwtween x-plane, y-plane, and the splat's tangent plane. We first transform :math:`h_{x}` to :math:`h_{u}` and :math:`h_{y}`
to :math:`h_{v}` in splat's tangent frame via the inverse transformation :math:`(WH)^{-1}`. As the intersection point should fall on :math:`h_{u}` and :math:`h_{v}`, we have an efficient
solution:

.. math::
    u(p) = \frac{h^{2}_{u}h^{4}_{v}-h^{4}_{u}h^{2}_{v}}{h^{1}_{u}h^{2}_{v}-h^{2}_{u}h^{1}_{v}}, 
    v(p) = \frac{h^{4}_{u}h^{1}_{v}-h^{1}_{u}h^{4}_{v}}{h^{1}_{u}h^{2}_{v}-h^{2}_{u}h^{1}_{v}}

.. autofunction:: rasterization_2dgs