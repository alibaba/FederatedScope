import random
import os
import numpy as np
import torch

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

from .model_aggregation import FedAvg, FlexLoRA, truncate
from .client_participation_scheduling import client_selection
from .client import GeneralClient
from .adaptive_peft import (seed_torch, tokenize, load_weight_local, distribute_weight_fast, modify_adapter,
                    distribute_weight, load_weight_SLoRA)