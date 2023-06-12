# coding: utf-8

from typing import Any
from abc import ABCMeta, abstractmethod

class SolverTemplate(metaclass = ABCMeta):
    pass

    @abstractmethod
    def solve(self) -> Any:
        pass