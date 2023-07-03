# coding: utf-8

import torch
import numpy as np
from typing import Dict, Any, Tuple

from . import TemplateScalarFieldTensorFactory
from ...data_structures import Config, ParticleFactoryConfig

class ParticleMaskFactory(TemplateScalarFieldTensorFactory):
    def __init__(self, config: ParticleFactoryConfig):
        if isinstance(config, ParticleFactoryConfig):
            self.config = config
            self.length = config.length
            self.margin = config.margin
        else:
            raise TypeError(f"Input config is not an instance of Config: {config}")
        
        if self.config.type == "custom":
            if self.config.custom_particle_distribution_config is None:
                raise ValueError(f"Custom particle config is not in the config")
            if not isinstance(self.config.custom_particle_distribution_config, list):
                raise TypeError(f"Custom particle config must be a list: {type(self.config.custom_particle_distribution_config)}")
            self.create_mask = self.create_custom_mask
            self.create_potential_field = self.create_custom_potential
        else:
            raise NotImplementedError(f"{self.config.type} is not implemented")

    def create(self, mask_type: str) -> torch.Tensor:
        if mask_type == "mask":
            array =  self.create_mask()
        elif mask_type == "potential":
            array = self.create_potential_field()
        else:
            raise NotImplementedError(f"{mask_type} is not Implemented")
        
        return torch.from_numpy(array)

    def create_custom_mask(self, *args: Tuple[Any], **kwargs: Dict[str, Any]):
        array = np.zeros((self.length, self.length))

        for config in self.config.custom_particle_distribution_config:
            tmp_array = self.put_2d_mask(array, config.x, config.y, config.w, config.h, config.theta)
            array += tmp_array
        
        return array
    
    def put_2d_mask(self, array: np.ndarray, x: int, y: int, w: int, h: int, theta: float) -> np.ndarray:
        """Put a 2D particle mask onto a filed.

        Args:
            array (np.ndarray): A given field to put a particle.
            x (int): position X.
            y (int): position Y.
            w (int): width.
            h (int): height.
            theta (float): Agnle.

        Raises:
            ValueError: 
            ValueError: 

        Returns:
            np.ndarray: A field where a new particle is newly allocated.
        """
        tmp_array = np.zeros((self.length, self.length))
        dx = int(w / 2)
        dy = int(h / 2)

        ## put particle on the array
        ## TODO: Margin should be considered
        self.__validate_particle_pos(x, dx, y, dy, self.length)

        tmp_array[x - int(w/2): x + int(w/2), y - int(h/2): y + int(h/2)] = 1
    
        ##TODO: rotate array

        self.__validate_particle_overlapping(array + tmp_array)
        
        return tmp_array
    
    def create_custom_potential(self, *args: Tuple[Any], **kwargs: Dict[str, Any]) -> np.ndarray:
        array = np.zeros((self.length, self.length))
        
        for config in self.config.custom_particle_distribution_config:
            tmp_array = self.put_2d_sruface_potential(array, config.x, config.y, config.w, config.h, config.theta, config.abs_electric_potential)
            array += tmp_array

        return array


    def put_2d_sruface_potential(self, array: np.ndarray, x: int, y: int, w: int, h: int, theta: float, abs_electric_potential: float) -> np.ndarray:
        tmp_array = np.zeros((self.length, self.length))
        dx = int(w / 2)
        dy = int(h / 2)

        ## put particle on the array
        ## TODO: Margin should be considered
        self.__validate_particle_pos(x, dx, y, dy, self.length)

        tmp_array[x - int(w/2): x - int(w/6), y - int(h/2): y + int(h/2)] = 1
        tmp_array[x - int(w/6): x + int(w/6), y - int(h/2): y + int(h/2)] = -1
        tmp_array[x + int(w/6): x + int(w/2), y - int(h/2): y + int(h/2)] = 1
 
        ##TODO: rotate array

        self.__validate_particle_overlapping(array + tmp_array)
        
        return tmp_array

    def __validate_particle_pos(self, x: int, dx: int, y: int, dy: int, length: int) -> None:
        if (x - dx < 0 or x + dx > length + y - dy < 0 + y + dy > length):
            raise ValueError(f"Particle exceeds the calculation region: (x, dx, y, dy) = ({x}, {dx}, {y}, {dy}")
        
    def __validate_particle_overlapping(self, array: np.ndarray) -> None:
        if np.max(array) > 1:
            raise ValueError("Overlapping particles")