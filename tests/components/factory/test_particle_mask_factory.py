# coding: utf-8

import pytest
import torch        
from typing import Dict, Any
import numpy as np

from torch_poisson_solver.components.factory import ParticleMaskFactory
from torch_poisson_solver.data_structures import ParticleFactoryConfig, CustomParticleDistributionConfig, Config
from torch_poisson_solver.components.operators import create_instance_from_dict

@pytest.fixture
def mock_custom_config() -> ParticleFactoryConfig:
    
    return ParticleFactoryConfig(dict(
        type="custom", length=10, margin=0,
        custom_particle_distribution_config=[CustomParticleDistributionConfig(x=5, y=5, z=0, w=3, h=5, l=0)]))


class TestParticleMaskFactoryInit:
    def test_init(self, mock_config_dict_custom_particle_factory: Dict[str, Any]) -> None:
        config =  create_instance_from_dict(Config, mock_config_dict_custom_particle_factory)
        
        assert isinstance(config, Config)
        assert isinstance(config.particle_factory, ParticleFactoryConfig)
        assert isinstance(config.particle_factory.custom_particle_distribution_config[0], CustomParticleDistributionConfig)
        factory = ParticleMaskFactory(config.particle_factory)
        assert isinstance(factory, ParticleMaskFactory)

    def test_create_single_custom_mask(self, mock_config_dict_custom_particle_factory: Dict[str, Any]) -> None:
        config =  create_instance_from_dict(Config, mock_config_dict_custom_particle_factory)
        factory = ParticleMaskFactory(config.particle_factory)
        tensor = factory.create("mask")
        assert isinstance(tensor, torch.Tensor)
        array = tensor.numpy()
        assert isinstance(array, np.ndarray)

        np.testing.assert_array_almost_equal(array, np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
            [0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
            [0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
            [0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
            [0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
            [0., 0., 1., 1., 1., 1., 1., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]), decimal=6)


    def test_create_single_custom_potential(self, mock_config_dict_custom_particle_factory: Dict[str, Any]) -> None:
        config =  create_instance_from_dict(Config, mock_config_dict_custom_particle_factory)
        factory = ParticleMaskFactory(config.particle_factory)
        tensor = factory.create("potential")
        assert isinstance(tensor, torch.Tensor)
        array = tensor.numpy()
        assert isinstance(array, np.ndarray)

        np.testing.assert_array_almost_equal(array, np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
            [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
            [ 0.,  0., -1., -1., -1., -1., -1., -1.,  0.,  0.],
            [ 0.,  0., -1., -1., -1., -1., -1., -1.,  0.,  0.],
            [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
            [ 0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
            [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]]), decimal=6)


    def test_create_single_custom_mask_witdth_assert(self, mock_config_dict_custom_particle_factory: Dict[str, Any]) -> None:
        mock_config_dict_custom_particle_factory["particle_factory"]["custom_particle_distribution_config"][0]["w"] = 3
        with pytest.raises(ValueError):
            config =  create_instance_from_dict(Config, mock_config_dict_custom_particle_factory)

    def test_validate_particle_overlapping(self):
        # TODO: testing a particle overlapping validation
        assert True