"""When a frozen baseline (z-score, Isolation Forest) stops collecting training data.

Two modes:
  * training_days = d: train on the first d calendar days of data. Training ends at the
    first vector of a later day, which is then already scored. With d = 1 the frozen
    models learn from the whole warm-up day, which the evaluation excludes anyway, instead
    of from its first minutes (the first 20,000 scored vectors covered only the opening
    auction and about 90 s of trading).
  * training_samples = n (when training_days is not set): train on the first n vectors, as
    before; the n-th vector completes training and is not scored.

Days are calendar days of the vector timestamps, which carry the exchange-local wall clock
labelled UTC, the same day boundaries the evaluation uses.
"""

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
        """Days mode: does this vector already belong to the scoring period?"""
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
        """Samples mode: has the count-th training vector completed training?"""
        return not self.training_days and count >= self.training_samples

    def describe(self, count: int) -> str:
        if self.training_days:
            return f"{count:,} vectors of {self.first_day} to {self.last_day}"
        return f"the first {count:,} vectors"


class Reservoir:
    """A uniform random sample of at most `capacity` rows of a stream (Algorithm R), so a
    model can be trained on a whole day of vectors in bounded memory."""

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
