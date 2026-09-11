import importlib.util
from typing import TypeVar

T = TypeVar("T")


def get_obj(module_path: str, obj_name: str):
    """
    Load a configuration instance from a Python module.

    Args:
        module_path: Path to the Python module file
        cfg_var_name: Name of the variable to load from the module
        expected_type: Expected type of the configuration object

    Returns:
        Configuration object of type T

    Raises:
        ImportError: If module cannot be loaded
        AttributeError: If variable doesn't exist in module
        TypeError: If variable is not of expected type
    """
    spec = importlib.util.spec_from_file_location("config_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Look for the specified cfg variable in the module
    if not hasattr(module, obj_name):
        raise AttributeError(
            f"Module {module_path} must contain a '{obj_name}' variable"
        )

    return getattr(module, obj_name)
