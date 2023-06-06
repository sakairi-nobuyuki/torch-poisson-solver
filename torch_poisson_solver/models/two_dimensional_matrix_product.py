# coding: utf-8

import torch

# Define the neural network
class Net(torch.nn.Module):
    def __init__(self):
        """
        Optimize two random matrices A and B with dimensions n x n on the GPU so that their product is minimized.

        Args:
            n (int): The dimensions of the matrices.

        Returns:
            torch.Tensor: The optimized matrix A.
        """
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(n * n, n * n)

    def forward(self, x):
        x = x.view(-1, n * n)
        x = self.fc1(x)
        x = x.view(-1, n, n)
        return x
