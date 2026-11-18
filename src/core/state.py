"""State for motion planning"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class State:
    """Configuration represented by a vector of real values"""

    values: np.ndarray

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=np.float64)

        if self.values.ndim != 1:
            raise ValueError("State must be a one-dimensional vector.")

    @property
    def dimension(self) -> int:
        return self.values.size

    def copy(self) -> "State":
        return State(self.values.copy())

    def distance(self, other: "State") -> float:
        self._check_dimension(other)
        return float(np.linalg.norm(self.values - other.values))

    def _check_dimension(self, other: "State"):
        if self.dimension != other.dimension:
            raise ValueError("State dimensions do not match.")

    def __getitem__(self, index):
        return self.values[index]

    def __setitem__(self, index, value):
        self.values[index] = value

    def __len__(self):
        return self.dimension

    def __repr__(self):
        return f"State({self.values})"