from __future__ import annotations

import numpy as np
from enum import Enum
from uuid import UUID

from typing import Tuple, TypeGuard

type EntityID = int | str | UUID | Enum


def is_entity_id(obj: object) -> TypeGuard[EntityID]:
    return isinstance(obj, (int, str, UUID, Enum))


type Vector = np.ndarray[Tuple[int,], np.dtype[np.float64]]
type VectorLike = Vector | Tuple[float, ...]
type Matrix = np.ndarray[Tuple[int, int], np.dtype[np.float64]]
