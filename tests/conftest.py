# coding: utf-8

import pytest
from typing import Dict, Any

@pytest.fixture
def mock_config_dict() -> Dict[str, Any]:
    return {
        "dimension": 3,
        "length": 2000,
        "solver": {
            "learning_rate": 0.001,
            "loss_type": "MSE",
            "optimizer_type": "Adam",
            "n_epoch": 200,
        },
        "particle_factory": {
            "type": "custom",
            "custom_particle_distribution_config": [
                {
                    "x": -50, "y": -50, "z": -50,
                    "w": 100, "h": 100, "l": 100,
                    "theta": -45.0, "phi": -45.0,
                    "abs_electric_potential": -1.0,
                }, 
                {
                    "x": 50, "y": 50, "z": 50,
                    "w": 100, "h": 100, "l": 100,
                    "theta": -45.0, "phi": -45.0,
                    "abs_electric_potential": -1.0,
                } 
            ],
            "random_particle_distribution_config": {
                "n_particle": 20,
                "w": 20, "h": 200, "l": 200,
                "abs_electric_potential": -1.0,
            },
        },
    }