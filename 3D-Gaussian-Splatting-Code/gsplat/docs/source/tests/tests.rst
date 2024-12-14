Tests
===================================

Testing and verifying CUDA implementations
--------------------------------------------

The `tests/` folder provides automatic test scripts that are ran to verify that the CUDA implementations agree with native PyTorch ones.
They are also ran with any pull-requests into main branch on our github repository.

The tests include: 

.. code-block:: bash
    :caption: tests/

    ./test_basic.py