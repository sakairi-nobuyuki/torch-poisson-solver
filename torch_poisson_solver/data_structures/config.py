# coding: utf-8

from dataclasses import dataclass
from typing import Optional, List

from torch_poisson_solver.components.operators import create_instance_from_dict

# from . import SolverConfig, TwoDimParticleDistributionCreatorConfig

"""
    dimension: int: "Dimensionality of the solver should solve. 2 or 3."
    length: int: Size of calculation
    solver: SolverConfig:
        learning_rate: float = 0.01
        loss_type: str = "MAE"
        optimizer_type: str = "SDG"
        n_epoch: int = 100
    particle_factory: ParticleFaactoryConfig
        type: str: "random" or "custom"
        custon_particle_distribution_config: List[CustomParticleDistributionConfig]: "Effective when type is 'custom'"
            - x: int: "X-directional position"
              y: int: "Y-directional position"
              z: int: "Z-directional position"
              w: int: "particle width"
              h: int: "particle height"
              l: int: "particle length. Effective when dimensionality is 3."
              theta: float: "paticle angle."
              phi: float: "particle angle. Effective when dimensionality is 3."
              abs_electric_potential: float: "Electric potential of the particle with particle's electric charge"
        random_particle_distribution_config: RandomParticleDistributionConfig: "Effective when type is 'random'"
            n_particle: int: 10: "Number of particles"
            w: int: "Width of the particle."
            h: int: "Height of the particle."
            l: int: "Length of the particle."
            abs_electric_potential: float: "Electric potential of the particle with particle's electric charge"
"""

@dataclass
class RandomParticleDistributionConfig:
    n_particle: int = 10
    w: int = 10
    h: int = 100
    l: int = 100
    abs_electric_potential: float = 1.0

@dataclass
class CustomParticleDistributionConfig:
    x: int = 100
    y: int = 100
    z: int = 100
    w: int = 100
    h: int = 100
    l: int = 100
    theta: float = 0.0
    phi: float = 0.0
    abs_electric_potential: float = 1.0

@dataclass
class ParticleFaactoryConfig:
    type: str = "custom"
    custom_particle_distribution_config: Optional[List[CustomParticleDistributionConfig]] = None
    random_particle_distribution_config: Optional[RandomParticleDistributionConfig] = None

    def __post_init__(self) -> None:
        self.custom_particle_distribution_config = [
            create_instance_from_dict(CustomParticleDistributionConfig, conf) 
            for conf in self.custom_particle_distribution_config]

        self.random_particle_distribution_config = create_instance_from_dict(
            RandomParticleDistributionConfig, self.random_particle_distribution_config)
        

@dataclass
class SolverConfig:
    learning_rate: float = 0.01
    loss_type: str = "MAE"
    optimizer_type: str = "SDG"
    n_epoch: int = 100

@dataclass
class Config:
    dimension: int = 2
    length: int = 1000
    solver: SolverConfig = None
    particle_factory: ParticleFaactoryConfig = None

    def __post_init__(self):
        if self.solver is not None:
            self.solver = create_instance_from_dict(SolverConfig, self.solver)
        if self.particle_factory is not None:
            self.particle_factory = create_instance_from_dict(ParticleFaactoryConfig, self.particle_factory)            
    

    
