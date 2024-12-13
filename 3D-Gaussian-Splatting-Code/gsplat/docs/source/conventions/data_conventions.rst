Data Conventions
===================================

.. currentmodule:: gsplat

Here we explain the various data conventions used in our repo.

Rotation Convention
-------------------
We represent rotations with four dimensional vectors :math:`q = (w,x,y,z)` such that the 3x3 :math:`SO(3)` rotation matrix is defined by:

.. math::

    R = \begin{bmatrix}
        1 - 2 \left( y^2 + z^2 \right) & 2 \left( x y - w z \right) & 2 \left( x z + w y \right) \\
        2 \left( x y + w z \right) & 1 - 2 \left( x^2 + z^2 \right) & 2 \left( y z - w x \right) \\
        2 \left( x z - w y \right) & 2 \left( y z + w x \right) & 1 - 2 \left( x^2 + y^2 \right) \\
        \end{bmatrix}

View Matrix
---------------------------------
We refer to the `view matrix` :math:`W` as the world to camera frame transformation (referred to as `w2c` in some sources) that maps
3D world points :math:`(x,y,z)_{world}` to 3D camera points :math:`(x,y,z)_{cam}` where :math:`z_{cam}` is the relative depth to the camera center.