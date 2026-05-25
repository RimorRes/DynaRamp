from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import networkx as nx


@dataclass
class Element:
    elem_id: int
    U: np.ndarray
    U_extract: Dict[int, np.ndarray] = field(default_factory=dict)
    H: Optional[np.ndarray] = None
    H_extract: Optional[Dict[int, np.ndarray]] = field(default_factory=dict)


@dataclass
class Connection:
    source: int
    sink: int
    input_slot: int = 1

@dataclass
class Boundary:
    elem_id: int
    free_dofs: List[int]
    fixed_dofs: List[int]

