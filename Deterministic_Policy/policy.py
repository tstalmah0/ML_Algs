#
# Simple feedforword nueral network to solve MNST dataset
#
import torch
from torch import nn

# global var for controling level of output
PRINT_ALL = True

# get devise
devise = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
if PRINT_ALL:
    print(f"usinng devise: {devise}")