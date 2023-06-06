# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim

from torch_poisson_solver.components.factory import DeviceFactory
from torch_poisson_solver.data_structures import TwoDimensionalField

class Optimizer:
    def __init__(self, device: DeviceFactory, field: TwoDimensionalField):
        self.device = device
        self.field = field
        

# Define the input and target matrices
x = torch.randn(100, 10) # a 100 x 10 matrix
y = torch.randn(100, 100) # a 100 x 100 matrix

# Define the matrix to be optimized
epsilon = nn.Linear(in_features=100, out_features=100)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epsilon.to(device)
y.to(device)
x.to(device)

# Define the loss function and the optimizer
loss_fn = nn.MSELoss()
optimizer = optim.SGD(epsilon.parameters(), lr=0.01)

# Run the optimization loop
for epoch in range(100):
    # Zero the gradients
    optimizer.zero_grad()
    # Compute the product of epsilon and y
    output = epsilon(y)
    # Compute the loss
    loss = loss_fn(output, x)
    # Backpropagate the loss
    loss.backward()
    # Update the parameters
    optimizer.step()
    # Print the loss
    print(f"Epoch {epoch}, loss {loss.item()}")

