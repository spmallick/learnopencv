from dataclasses import dataclass
from typing import Dict, Union

import torch


@dataclass
class Strategy:
    """Base class for the GS densification strategy.

    This class is an base class that defines the interface for the GS
    densification strategy.
    """

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Sanity check for the parameters and optimizers."""
        trainable_params = set(
            [name for name, param in params.items() if param.requires_grad]
        )
        assert trainable_params == set(optimizers.keys()), (
            "trainable parameters and optimizers must have the same keys, "
            f"but got {trainable_params} and {optimizers.keys()}"
        )

        for optimizer in optimizers.values():
            assert len(optimizer.param_groups) == 1, (
                "Each optimizer must have exactly one param_group, "
                "that cooresponds to each parameter, "
                f"but got {len(optimizer.param_groups)}"
            )

    def step_pre_backward(
        self,
        *args,
        **kwargs,
    ):
        """Callback function to be executed before the `loss.backward()` call."""
        pass

    def step_post_backward(
        self,
        *args,
        **kwargs,
    ):
        """Callback function to be executed after the `loss.backward()` call."""
        pass
