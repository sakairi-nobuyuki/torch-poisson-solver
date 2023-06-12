# coding: utf-8


import torch
import torch


from torch_poisson_solver.components.solver import SolverTemplate
from torch_poisson_solver.data_structures import SolverConfig

class NaivePoissonSolver(SolverTemplate):
    lr: float = 0.01
    f_loss: torch.nn = torch.nn.L1Loss()
    optimizer: torch.optim
    n_epoch = 100

    def __init__(self, config: SolverConfig = None):

        if config is not None:
            if not isinstance(config, SolverConfig):
                raise TypeError(f"{type(config)} is not an instance of SolverConfig")
            
            self.lr = config.learning_rate
            self.n_epoch = config.n_epoch
            if config.loss_type == "MAE":
                f_loss = torch.nn.L1Loss()


    def solve(self, laplacean: torch.Tensor, phi: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:

        self.optimizer = torch.optim.SGD([], lr=self.lr)
        for epoch in range(self.n_epoch):
            loss = self.f_loss(torch.matmul(laplacean, phi), rho)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return phi
