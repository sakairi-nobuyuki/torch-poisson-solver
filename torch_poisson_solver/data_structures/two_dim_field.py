# coding: utf-8

import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class TwoDimensionalField:
    phi: Optional[torch.Tensor] = None

    def __post_init__(self, length: int) -> None:
        self.phi = torch.randn(length, length)
        self.laplacean = torch.randn(length, length)
