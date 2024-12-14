import os
from typing import Any, Callable, List, Optional, Union

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as distF
from torch import Tensor


def all_gather_int32(
    world_size: int, value: Union[int, Tensor], device: Optional[torch.device] = None
) -> List[int]:
    """Gather an 32-bit integer from all ranks.

    .. note::
        This implementation is faster than using `torch.distributed.all_gather_object`.

    .. note::
        This function is not differentiable to the input tensor.

    Args:
        world_size: The total number of ranks.
        value: The integer to gather. Could be a scalar or a tensor.
        device: Only required if `value` is a scalar. The device to put the tensor on.

    Returns:
        A list of integers, where the i-th element is the value from the i-th rank.
        Could be a list of scalars or tensors based on the input `value`.
    """
    if world_size == 1:
        return [value]

    # move to CUDA
    if isinstance(value, int):
        assert device is not None, "device is required for scalar input"
        value_tensor = torch.tensor(value, dtype=torch.int, device=device)
    else:
        value_tensor = value
    assert value_tensor.is_cuda, "value should be on CUDA"

    # gather
    collected = torch.empty(
        world_size, dtype=value_tensor.dtype, device=value_tensor.device
    )
    dist.all_gather_into_tensor(collected, value_tensor)

    if isinstance(value, int):
        # return as list of integers on CPU
        return collected.tolist()
    else:
        # return as list of single-element tensors
        return collected.unbind()


def all_to_all_int32(
    world_size: int,
    values: List[Union[int, Tensor]],
    device: Optional[torch.device] = None,
) -> List[int]:
    """Exchange 32-bit integers between all ranks in a many-to-many fashion.

    .. note::
        This function is not differentiable to the input tensors.

    Args:
        world_size: The total number of ranks.
        values: A list of integers to exchange. Could be a list of scalars or tensors.
            Should have the same length as `world_size`.
        device: Only required if `values` contains scalars. The device to put the tensors on.

    Returns:
        A list of integers. Could be a list of scalars or tensors based on the input `values`.
        Have the same length as `world_size`.
    """
    if world_size == 1:
        return values

    assert (
        len(values) == world_size
    ), "The length of values should be equal to world_size"

    if any(isinstance(v, int) for v in values):
        assert device is not None, "device is required for scalar input"

    # move to CUDA
    values_tensor = [
        (torch.tensor(v, dtype=torch.int, device=device) if isinstance(v, int) else v)
        for v in values
    ]

    # all_to_all
    collected = [torch.empty_like(v) for v in values_tensor]
    dist.all_to_all(collected, values_tensor)

    # return as a list of integers or tensors, based on the input
    return [
        v.item() if isinstance(tensor, int) else v
        for v, tensor in zip(collected, values)
    ]


def all_gather_tensor_list(world_size: int, tensor_list: List[Tensor]) -> List[Tensor]:
    """Gather a list of tensors from all ranks.

    .. note::
        This function expects the tensors in the `tensor_list` to have the same shape
        and data type across all ranks.

    .. note::
        This function is differentiable to the tensors in `tensor_list`.

    .. note::
        For efficiency, this function internally concatenates the tensors in `tensor_list`
        and performs a single gather operation. Thus it requires all tensors in the list
        to have the same first-dimension size.

    Args:
        world_size: The total number of ranks.
        tensor_list: A list of tensors to gather. The size of the first dimension of all
            the tensors in this list should be the same. The rest dimensions can be
            arbitrary. Shape: [(N, *), (N, *), ...]

    Returns:
        A list of tensors gathered from all ranks, where the i-th element is corresponding
        to the i-th tensor in `tensor_list`. The returned tensors have the shape
        [(N * world_size, *), (N * world_size, *), ...]

    Examples:

    .. code-block:: python

        >>> # on rank 0
        >>> # tensor_list = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        >>> # on rank 1
        >>> # tensor_list = [torch.tensor([7, 8, 9]), torch.tensor([10, 11, 12])]
        >>> collected = all_gather_tensor_list(world_rank, world_size, tensor_list)
        >>> # on both ranks
        >>> # [torch.tensor([1, 2, 3, 7, 8, 9]), torch.tensor([4, 5, 6, 10, 11, 12])]

    """
    if world_size == 1:
        return tensor_list

    N = len(tensor_list[0])
    for tensor in tensor_list:
        assert len(tensor) == N, "All tensors should have the same first dimension size"

    # concatenate tensors and record their sizes
    data = torch.cat([t.reshape(N, -1) for t in tensor_list], dim=-1)
    sizes = [t.numel() // N for t in tensor_list]

    if data.requires_grad:
        # differentiable gather
        collected = distF.all_gather(data)
    else:
        # non-differentiable gather
        collected = [torch.empty_like(data) for _ in range(world_size)]
        torch.distributed.all_gather(collected, data)
    collected = torch.cat(collected, dim=0)

    # split the collected tensor and reshape to the original shape
    out_tensor_tuple = torch.split(collected, sizes, dim=-1)
    out_tensor_list = []
    for out_tensor, tensor in zip(out_tensor_tuple, tensor_list):
        out_tensor = out_tensor.view(-1, *tensor.shape[1:])  # [N * world_size, *]
        out_tensor_list.append(out_tensor)
    return out_tensor_list


def all_to_all_tensor_list(
    world_size: int,
    tensor_list: List[Tensor],
    splits: List[Union[int, Tensor]],
    output_splits: Optional[List[Union[int, Tensor]]] = None,
) -> List[Tensor]:
    """Split and exchange tensors between all ranks in a many-to-many fashion.

    Args:
        world_size: The total number of ranks.
        tensor_list: A list of tensors to split and exchange. The size of the first
            dimension of all the tensors in this list should be the same. The rest
            dimensions can be arbitrary. Shape: [(N, *), (N, *), ...]
        splits: A list of integers representing the number of elements to send to each
            rank. It will be used to split the tensor in the `tensor_list`.
            The sum of the elements in this list should be equal to N. The size of this
            list should be equal to the `world_size`.
        output_splits: Splits of the output tensors. Could be pre-calculated via
            `all_to_all_int32(world_size, splits)`. If not provided, it will
            be calculated internally.

    Returns:
        A list of tensors exchanged between all ranks, where the i-th element is
        corresponding to the i-th tensor in `tensor_list`. Note the shape of the
        returned tensors might be different from the input tensors, depending on the
        splits.

    Examples:

    .. code-block:: python

        >>> # on rank 0
        >>> # tensor_list = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
        >>> # splits = [2, 1]

        >>> # on rank 1
        >>> # tensor_list = [torch.tensor([7, 8]), torch.tensor([9, 10])]
        >>> # splits = [1, 1]

        >>> collected = all_to_all_tensor_list(world_rank, world_size, tensor_list, splits)

        >>> # on rank 0
        >>> # [torch.tensor([1, 2, 7]), torch.tensor([4, 5, 9])]
        >>> # on rank 1
        >>> # [torch.tensor([3, 8]), torch.tensor([6, 10])]

    """
    if world_size == 1:
        return tensor_list

    N = len(tensor_list[0])
    for tensor in tensor_list:
        assert len(tensor) == N, "All tensors should have the same first dimension size"

    assert (
        len(splits) == world_size
    ), "The length of splits should be equal to world_size"

    # concatenate tensors and record their sizes
    data = torch.cat([t.reshape(N, -1) for t in tensor_list], dim=-1)
    sizes = [t.numel() // N for t in tensor_list]

    # all_to_all
    if output_splits is not None:
        collected_splits = output_splits
    else:
        collected_splits = all_to_all_int32(world_size, splits, device=data.device)
    collected = [
        torch.empty((l, *data.shape[1:]), dtype=data.dtype, device=data.device)
        for l in collected_splits
    ]
    # torch.split requires tuple of integers
    splits = [s.item() if isinstance(s, Tensor) else s for s in splits]
    if data.requires_grad:
        # differentiable all_to_all
        distF.all_to_all(collected, data.split(splits, dim=0))
    else:
        # non-differentiable all_to_all
        torch.distributed.all_to_all(collected, list(data.split(splits, dim=0)))
    collected = torch.cat(collected, dim=0)

    # split the collected tensor and reshape to the original shape
    out_tensor_tuple = torch.split(collected, sizes, dim=-1)
    out_tensor_list = []
    for out_tensor, tensor in zip(out_tensor_tuple, tensor_list):
        out_tensor = out_tensor.view(-1, *tensor.shape[1:])
        out_tensor_list.append(out_tensor)
    return out_tensor_list


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def _distributed_worker(
    world_rank: int,
    world_size: int,
    fn: Callable,
    args: Any,
    local_rank: Optional[int] = None,
    verbose: bool = False,
) -> bool:
    if local_rank is None:  # single Node
        local_rank = world_rank
    if verbose:
        print("Distributed worker: %d / %d" % (world_rank + 1, world_size))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend="nccl", world_size=world_size, rank=world_rank
        )
        # Dump collection that participates all ranks.
        # This initializes the communicator required by `batch_isend_irecv`.
        # See: https://github.com/pytorch/pytorch/pull/74701
        _ = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(_, 0)
    fn(local_rank, world_rank, world_size, args)
    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    if verbose:
        print("Job Done for worker: %d / %d" % (world_rank + 1, world_size))
    return True


def cli(fn: Callable, args: Any, verbose: bool = False) -> bool:
    """Wrapper to run a function in a distributed environment.

    The function `fn` should have the following signature:

    ```python
    def fn(local_rank: int, world_rank: int, world_size: int, args: Any) -> None:
        pass
    ```

    Usage:

    ```python
    # Launch with "CUDA_VISIBLE_DEVICES=0,1,2,3 python my_script.py"
    if __name__ == "__main__":
        cli(fn, None, verbose=True)
    ```
    """
    assert torch.cuda.is_available(), "CUDA device is required!"
    if "OMPI_COMM_WORLD_SIZE" in os.environ:  # multi-node
        local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])  # dist.get_world_size()
        world_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])  # dist.get_rank()
        return _distributed_worker(
            world_rank, world_size, fn, args, local_rank, verbose
        )

    world_size = torch.cuda.device_count()
    distributed = world_size > 1

    if distributed:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(_find_free_port())
        process_context = torch.multiprocessing.spawn(
            _distributed_worker,
            args=(world_size, fn, args, None, verbose),
            nprocs=world_size,
            join=False,
        )
        try:
            process_context.join()
        except KeyboardInterrupt:
            # this is important.
            # if we do not explicitly terminate all launched subprocesses,
            # they would continue living even after this main process ends,
            # eventually making the OD machine unusable!
            for i, process in enumerate(process_context.processes):
                if process.is_alive():
                    if verbose:
                        print("terminating process " + str(i) + "...")
                    process.terminate()
                process.join()
                if verbose:
                    print("process " + str(i) + " finished")
        return True
    else:
        return _distributed_worker(0, 1, fn=fn, args=args)
