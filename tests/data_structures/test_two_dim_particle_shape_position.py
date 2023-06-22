# coding: utf-8


from torch_poisson_solver.data_structures import TwoDimParticleShapePosition

class TestTwoDimParticleDistributionConfig:
    """Testing dataclass for particle distribution config. """
    def test_two_dim_particle_shape_position(self) -> None:
        """Testing initialize with normal method.
        """
        conf = TwoDimParticleShapePosition(x=100, y=100, w=10, h=100)
        assert isinstance(conf, TwoDimParticleShapePosition)
    def test_two_dim_particle_shape_position_dict_input(self) -> None:
        """Testing to initialize the dataclass with a dict input"""
        conf_input_dict = {"x":100, "y":100, "w":10, "h":100}
        conf = TwoDimParticleShapePosition(**conf_input_dict)
        assert isinstance(conf, TwoDimParticleShapePosition)
        assert conf.x == conf_input_dict["x"]
        assert conf.y == conf_input_dict["y"]
        assert conf.w == conf_input_dict["w"]
        assert conf.h == conf_input_dict["h"]
