# coding: utf-8

from typing import Dict, List, Union, Any, Optional
import torch
import numpy as np

from torch_poisson_solver.components.factory import TemplateFactory, DeviceFactory

class TwoDimFieldFactory(TemplateFactory):
    
    def __init__(self, length: int, type: str):
        self.length = length
        

    def create(self, *args: List[Any], **kwargs: Dict[str, Union[str, int]]) -> torch.Tensor:
        """Create field values such as differenciating operators in a matrix form, unknown variable or source term.

        expected form of kwargs:
            field_type: str = Optional["unknown_variable", "operator", "source", "mask"]
            mask_config:
                - type: str: ["random", "fixed"]
                  n: int: number of particles
                  w: int: width
                  h: int: height
                  x: int: x-directional particle center position 
                  y: int: y-directional particle center position
                  theta: float: angle
            boundary_condition_type: str = Optional["Neumann", "Dirichlet", "periodical"]

        Returns:
            torch.Tensor: Field value as a tensor
        """
        self.device = DeviceFactory().create()
        if kwargs["type"] == "unknown_variable":
            return torch.randn(self.length, self.length).to(self.device)
        elif kwargs["type"] == "laplacean":
            return self.create_laplacean()


    def create_laplacean(self) -> torch.Tensor:
        diag = np.diag(np.full(self.length, 1))
        array = -4.0 * diag + np.pad(diag, [(0, 1), (0, 0)], mode="constant") + np.pad(diag, [(0, 0), (0, 1)], mode="constant")
        return torch.from_numpy(array).to(self.device)
        
    def create_mask(self, config_list: List[Dict[str, Union[int, float]]]) -> torch.Tensor:
        """_summary_

        Args:
            config_list (List[Dict[str, Union[int, float]]]): _description_

        Returns:
            torch.Tensor: _description_
        """
        array = np.zeros((self.length, self.length))
        for config in config_list:
            array += self.create_particle_mask()
        pass

    def create_particle_mask(self, x: int, y: int, w: int, h: int, theta: float) -> np.ndarray:
        
        array = np.zeros((self.length, self.length))
        dx = int(w/2)
        dy = int(h/2)

        ## put particle on the array
        if (x - dx < 0 or x + dx > self.length + y - dy < 0 + y + dy > self.length):
            raise ValueError(f"Particle exceeds the calculation region: (x, dx, y, dy) = ({x}, {dx}, {y}, {dy}")
        else:
            array[x - int(w/2):x + int(w/2), y - int(h/2):y + int(h/2)] = 1
            return array