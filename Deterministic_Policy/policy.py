#
# Simple feedforword nueral network to solve MNST dataset
#
import torch
from torch import nn

# global var for controling level of output
PRINT_ALL = True

# get devise
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
if PRINT_ALL:
    print(f"usinng device: {device}")

# class for nuneral network
class NeuralNetwork(nn.Module):
    # initialize module
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
        )

    # function for forword pass
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
# create instance of model and move it to devise
model = NeuralNetwork().to(device)
if PRINT_ALL:
    print(model)