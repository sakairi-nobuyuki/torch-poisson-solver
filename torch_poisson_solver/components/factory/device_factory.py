# coding: utf-8

import torch
from torch_poisson_solver.components.factory import TemplateFactory

class DeviceFactory(TemplateFactory):
    
    def __init__(self) -> None:
        pass

    def create(self) -> torch.device:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")