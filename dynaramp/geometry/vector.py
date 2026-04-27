import numpy as np

def normalize(vec: np.ndarray) -> np.ndarray:
    """Normalize a vector"""
    try:
        return vec / np.linalg.norm(vec)
    except ZeroDivisionError:
        raise ValueError("Vector cannot be zero vector.")
