# coding: utf-8

from dataclasses import dataclass

@dataclass
class SolverConfig:
    learning_rate: float = 0.01
    loss_type: str = "MAE"
    optimizer_type: str = "SDG"
    n_epoch: int = 100