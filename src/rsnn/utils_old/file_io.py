import copy
import os
from typing import Any

import dill as pickle


def save_object_to_file(obj:Any, path: str):
    """Save the object to a file.

    Args:
        obj (Any): the object to save.
        path (str): the path to the saving location.

    Raises:
        ValueError: if error saving the object
    """    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save the configuration to a file
    try:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    except (FileNotFoundError, PermissionError) as e:
        raise ValueError(f"Error saving object: {e}")
    
    
def load_object_from_file(path: str):
    """Load the object from a file.

    Args:
        path (str): the path to the loading location.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If number of neurons does not match.
        ValueError: If error loading the file.
    """

    # Check if the file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} not found")

    # Load the configuration from the file
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, PermissionError) as e:
        raise ValueError(f"Error loading network configuration: {e}")
    

def copy_object(obj:Any) -> Any:
    """Copy an object.

    Args:
        obj (Any): the object to copy.

    Returns:
        Any: the copied object.
    """
    return copy.deepcopy(obj)