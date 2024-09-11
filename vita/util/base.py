import torch
import random
import numpy as np
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
