# coding: utf-8

from typing import Dict, List, Union, Any
import torch

from torch_poisson_solver.components.factory import TemplateFactory

class TwoDimFieldFactory(TemplateFactory):
    
    def __init__(self, length: int, type: str):
        self.length = length
        self.type = type
        

    def create(self, *args: List[Any], **kwargs: Dict[str, Union[str, int]]) -> torch.Tensor:
        """Create field values such as differenciating operators in a matrix form, unknown variable or source term.

        expected form of kwargs:
            field_type: str = Optional["unknown_variable", "operator", "source"]
            boundary_condition_type: str = Optional["Neumann", "Dirichlet", "periodical"]


        Returns:
            torch.Tensor: Field value as a tensor
        """
        if kwargs["type"] == "unknown_variable":
            return torch.randn(self.length, self.length)