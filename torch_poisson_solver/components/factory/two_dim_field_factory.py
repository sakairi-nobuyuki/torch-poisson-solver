# coding: utf-8

from typing import Dict, List, Union, Any
import torch
import numpy as np

from torch_poisson_solver.components.factory import TemplateFactory, DeviceFactory

class TwoDimFieldFactory(TemplateFactory):
    
    def __init__(self, length: int, type: str):
        self.length = length
        

    def create(self, *args: List[Any], **kwargs: Dict[str, Union[str, int]]) -> torch.Tensor:
        """Create field values such as differenciating operators in a matrix form, unknown variable or source term.

        expected form of kwargs:
            field_type: str = Optional["unknown_variable", "operator", "source"]
            boundary_condition_type: str = Optional["Neumann", "Dirichlet", "periodical"]


        Returns:
            torch.Tensor: Field value as a tensor
        """
        self.device = DeviceFactory().create()
        if kwargs["type"] == "unknown_variable":
            return torch.randn(self.length, self.length).to(self.device)
        elif kwargs["type"] == "laplacean":
            return self.create_laplacean()

    def create_laplacean(self):
        diag = np.diag(np.full(self.length, 1))
        array = -4.0 * diag + np.pad(diag, [(0, 1), (0, 0)], mode="constant") + np.pad(diag, [(0, 0), (0, 1)], mode="constant")
        return torch.from_numpy(array).to(self.device)
        