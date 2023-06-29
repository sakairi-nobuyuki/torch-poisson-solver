# coding: utf-8

from typing import Dict, Any, List

from torch_poisson_solver.data_structures.config import Config, ParticleFactoryConfig, CustomParticleDistributionConfig
from torch_poisson_solver.components.operators import create_instance_from_dict

class TestConfig:
    

    def test_init(self) -> None:
        """Testing initialization"""
        config = Config()

        assert isinstance(config, Config)

    def test_config_load(self, mock_config_dict: Dict[str, Any]) -> None:
        config = create_instance_from_dict(Config, mock_config_dict)
        assert isinstance(config, Config)
        assert config.length == mock_config_dict["length"]
        assert config.dimension == mock_config_dict["dimension"]
        print(config.solver, type(config.solver))
        #pprint(mock_config_dict)
        
        assert config.solver.learning_rate == mock_config_dict["solver"]["learning_rate"]
        assert config.particle_factory.type == mock_config_dict["particle_factory"]["type"]
        assert len(config.particle_factory.custom_particle_distribution_config) == len(mock_config_dict["particle_factory"]["custom_particle_distribution_config"])
        assert config.particle_factory.custom_particle_distribution_config[0].x == mock_config_dict["particle_factory"]["custom_particle_distribution_config"][0]["x"]
        assert config.particle_factory.custom_particle_distribution_config[0].y == mock_config_dict["particle_factory"]["custom_particle_distribution_config"][0]["y"]

    def test_config_load_custom_particle_factory(self, mock_config_dict_custom_particle_factory: Dict[str, Any]) -> None:
        assert isinstance(mock_config_dict_custom_particle_factory, Dict)
        config = create_instance_from_dict(Config, mock_config_dict_custom_particle_factory)
        assert isinstance(config, Config)
        assert isinstance(config.particle_factory, ParticleFactoryConfig)
        assert config.particle_factory.custom_particle_distribution_config[0].y == mock_config_dict_custom_particle_factory["particle_factory"]["custom_particle_distribution_config"][0]["y"]
        
        assert isinstance(config.particle_factory.custom_particle_distribution_config, List)
        assert config.particle_factory.length == config.length