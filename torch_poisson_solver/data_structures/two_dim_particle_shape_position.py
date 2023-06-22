# coding: utf-8

from dataclasses import dataclass

@dataclass
class ParticleShapePosition:
    """Configuration of particle shape and its position."""
    x: int = 100
    y: int = 100
    z: int = 100
    w: int = 10
    h: int = 100
    l: int = 100
    theta: float = 0.0
    phi: float = 0.0
    abs_electric_potential: float = 1.0
