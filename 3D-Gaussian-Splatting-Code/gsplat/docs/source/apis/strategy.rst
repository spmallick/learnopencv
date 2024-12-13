Densification
===================================

.. currentmodule:: gsplat

In `gsplat`, we abstract out the densification and pruning process of the Gaussian 
training into a strategy. A strategy is a class that defines how the Gaussian parameters
(along with their optimizers) should be updated (splitting, pruning, etc.) during the training. 

An example of the training workflow using :class:`DefaultStrategy` is like:

.. code-block:: python

    from gsplat import DefaultStrategy, rasterization

    # Define Gaussian parameters and optimizers
    params: Dict[str, torch.nn.Parameter] | torch.nn.ParameterDict = ...
    optimizers: Dict[str, torch.optim.Optimizer] = ...

    # Initialize the strategy
    strategy = DefaultStrategy()

    # Check the sanity of the parameters and optimizers
    strategy.check_sanity(params, optimizers)

    # Initialize the strategy state
    strategy_state = strategy.initialize_state()

    # Training loop
    for step in range(1000):
        # Forward pass
        render_image, render_alpha, info = rasterization(...)

        # Pre-backward step
        strategy.step_pre_backward(params, optimizers, strategy_state, step, info)

        # Compute the loss
        loss = ...

        # Backward pass
        loss.backward()

        # Post-backward step
        strategy.step_post_backward(params, optimizers, strategy_state, step, info)

A strategy will inplacely update the Gaussian parameters as well as the optimizers,
so it has a specific expectation on the format of the parameters and the optimizers.
It is designed to work with the Guassians defined as either a Dict of
`torch.nn.Parameter <https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html>`_
or a
`torch.nn.ParameterDict <https://pytorch.org/docs/stable/generated/torch.nn.ParameterDict.html>`_
with at least the following keys: {"means", "scales", "quats", "opacities"}. On top of these attributes,
an arbitrary  number of extra attributes are supported. Besides the parameters, it also
expects a Dict of `torch.optim.Optimizer <https://pytorch.org/docs/stable/optim.html>`_
with the same keys as the parameters, and each optimizer should correspond to only
one learnable parameter.

For example, the following is a valid format for the parameters and the optimizers
that can be used with our strategies:

.. code-block:: python

    N = 100
    params = torch.nn.ParameterDict{
        "means": Tensor(N, 3), "scales": Tensor(N), "quats": Tensor(N, 4), "opacities": Tensor(N),
        "colors": Tensor(N, 25, 3), "features1": Tensor(N, 128), "features2": Tensor(N, 64),
    }
    optimizers = {k: torch.optim.Adam([p], lr=1e-3) for k, p in params.keys()}

Below are the strategies that are currently implemented in `gsplat`:

.. autoclass:: DefaultStrategy
    :members:

.. autoclass:: MCMCStrategy
    :members:
