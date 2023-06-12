# coding: utf-8

from torch_poisson_solver.components.factory import DeviceFactory
import torch

class TestDeviceFactory:
    factory = DeviceFactory()
    def test_create(self) -> None:
        """Testing create factory
        """
        assert isinstance(self.factory, DeviceFactory)
        assert isinstance(self.factory.create(), torch.device)
