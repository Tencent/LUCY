import torch
import random
import numpy as np
import collections
from transformers import set_seed


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

def rank0_print(local_rank, *args):
    if local_rank == 0:
        print(*args)

# from https://github.com/facebookresearch/fairseq/blob/main/fairseq/utils.py
def apply_to_sample(f, sample):
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, collections.OrderedDict):
            # OrderedDict has attributes that needs to be preserved
            od = collections.OrderedDict(
                (key, _apply(value)) for key, value in x.items()
            )
            od.__dict__ = x.__dict__
            return od
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample, device=None):
    device = device or torch.cuda.current_device()

    def _move_to_cuda(tensor):
        # non_blocking is ignored if tensor is not pinned, so we can always set
        # to True (see github.com/PyTorchLightning/pytorch-lightning/issues/620)
        return tensor.to(device=device, non_blocking=True)

    return apply_to_sample(_move_to_cuda, sample)

def move_to_cpu(sample):
    def _move_to_cpu(tensor):
        # PyTorch has poor support for half tensors (float16) on CPU.
        # Move any such tensors to float32.
        if tensor.dtype in {torch.bfloat16, torch.float16}:
            tensor = tensor.to(dtype=torch.float32)
        return tensor.cpu()

    return apply_to_sample(_move_to_cpu, sample)
