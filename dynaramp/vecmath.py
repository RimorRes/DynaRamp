from __future__ import annotations
import logging

import numpy as np

from .types import Vector, Matrix


logger = logging.getLogger(__name__)


def skew_sym_mat(vec: Vector) -> Matrix:
    """
    Returns the skew symmetric matrix of a 3D vector.
    The skew symmetric matrix is defined such that for any vector v,
    the cross-product with another vector w can be expressed as a matrix multiplication: v x w = S(v) * w,
    where S(v) is the skew symmetric matrix of v.
    :param vec:
    :return:
    """

    vec = np.array(vec)
    # Verify shape
    match vec.shape:
        # "Flat" vector
        case (3,):
            ssv = np.array([
                [0, -vec[2], vec[1]],
                [vec[2], 0, -vec[0]],
                [-vec[1], vec[0], 0]
            ])
        case (3,1):
            ssv = np.array([
                [0, -vec[2, 0], vec[1, 0]],
                [vec[2, 0], 0, -vec[0, 0]],
                [-vec[1, 0], vec[0, 0], 0]
            ])
        case _:
            err_msg = f"Input vector must be of shape (3,) or (3,1). Got shape {vec.shape}"
            logger.error(err_msg)
            raise ValueError(err_msg)

    return ssv