from datetime import datetime
from typing import Optional

import numpy as np


class TrainingWindow:
    def __init__(self, training_samples: int = 20000, training_days: Optional[int] = None):
        self.training_samples = training_samples
        self.training_days = training_days
        self.first_day = None
        self.last_day = None

    def ends_before(self, timestamp: datetime) -> bool:
        if not self.training_days:
            return False
        day = timestamp.date()
        if self.first_day is None:
            self.first_day = day
        if (day - self.first_day).days >= self.training_days:
            return True
        self.last_day = day
        return False

    def ends_after(self, count: int) -> bool:
        return not self.training_days and count >= self.training_samples

    def describe(self, count: int) -> str:
        if self.training_days:
            return f"{count:,} vectors of {self.first_day} to {self.last_day}"
        return f"the first {count:,} vectors"


class Reservoir:
    def __init__(self, capacity: int, width: int, seed: int = 42):
        self.capacity = capacity
        self.rows = np.empty((capacity, width))
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def add(self, row: np.ndarray) -> None:
        if self.seen < self.capacity:
            self.rows[self.seen] = row
        else:
            j = self.rng.integers(0, self.seen + 1)
            if j < self.capacity:
                self.rows[j] = row
        self.seen += 1

    def sample(self) -> np.ndarray:
        return self.rows[: min(self.seen, self.capacity)]
