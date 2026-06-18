from __future__ import annotations

import numpy as np
from enum import Enum
from uuid import UUID

from typing import Tuple


type EntityID = int | str | UUID | Enum
type Vector = np.ndarray[tuple[int,], np.dtype[np.number]] | Tuple[float,...]
type Matrix = np.ndarray[tuple[int, int], np.dtype[np.number]]