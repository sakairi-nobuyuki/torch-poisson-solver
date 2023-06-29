# coding: utf-8

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
            self.create_ = self.create_custom_mask
        else:
            raise NotImplementedError(f"{self.config.type} is not implemented")

    def create(self, *args: Tuple[Any], **kwargs: Dict[str, Any]) -> np.ndarray:
        return self.create_(kwargs)

    def create_custom_mask(self, *args: Tuple[Any], **kwargs: Dict[str, Any]):
        array = np.zeros((self.length, self.length))
        for config in self.config.custom_particle_distribution_config:
            tmp_array = np.zeros((self.length, self.length))
            dx = int(config.w / 2)
            dy = int(config.h / 2)

            ## put particle on the array
            if (config.x - dx < 0 or config.x + dx > self.length + config.y - dy < 0 + config.y + dy > self.length):
                raise ValueError(f"Particle exceeds the calculation region: (x, dx, y, dy) = ({config.x}, {dx}, {config.y}, {dy}")
            else:
                tmp_array[config.x - int(config.w/2):config.x + int(config.w/2), 
                          config.y - int(config.h/2):config.y + int(config.h/2)] = 1
            array += tmp_array
            if np.max(array) > 1:
                raise ValueError("Overlapping particles")
        return array
    
    def put_2d_mask(self, array: np.ndarray, x: int, y: int, w: int, h: int, theta: float) -> np.ndarray:
        tmp_array = np.zeros((self.length, self.length))
        dx = int(w / 2)
        dy = int(h / 2)

        ## put particle on the array
        ## TODO: Margin should be considered
        if (x - dx < 0 or x + dx > self.length + y - dy < 0 + y + dy > self.length):
            raise ValueError(f"Particle exceeds the calculation region: (x, dx, y, dy) = ({x}, {dx}, {y}, {dy}")
        else:
            tmp_array[x - int(w/2): x + int(w/2), 
                        y - int(h/2): y + int(h/2)] = 1
        
        ##TODO: rotate array

        array += tmp_array


        if np.max(array) > 1:
            raise ValueError("Overlapping particles")
        
        return array