# cofing: utf-8

from typing import Dict, Type

def create_instance_from_dict(cls: Type, dict_: Dict) -> object:
    """
    Create an instance of a dataclass from a dictionary.

    Args:
        cls (Type): The dataclass type.
        dict_ (Dict): The dictionary containing the dataclass values.

    Returns:
        object: The dataclass instance.
    """
    return cls(**dict_)


