# coding: utf-8

from torch_poisson_solver.components.factory import TwoDimFieldFactory
import torch

class TestTwoDimFieldFactory:
    def test_init(self) -> None:
        factory = TwoDimFieldFactory(100)
        assert isinstance(factory, TwoDimFieldFactory)
        assert factory.length == 100
    def test_create(self) -> None:
        factory = TwoDimFieldFactory(100)
        phi = factory.create(**{"type":"unknown_variable"})
        assert isinstance(phi, torch.Tensor)
        assert phi.dim() == 2
        assert phi.size() == torch.Size([100, 100])

        laplacean = factory.create(**{"type": "laplacean"})
        assert isinstance(laplacean, torch.Tensor)
        assert laplacean.dim() == 2
        assert laplacean[0][0].item() == -4
        assert laplacean[1][0].item() == 1
        assert laplacean[2][0].item() == 0
        assert laplacean[0][1].item() == 1

