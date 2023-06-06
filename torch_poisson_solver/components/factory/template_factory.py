# coding: utf-8

from typing import Any
from abc import ABCMeta, abstractmethod

class TemplateFactory(metaclass = ABCMeta):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def create(self, *args: Any, **kwargs: Any) -> Any:
        pass