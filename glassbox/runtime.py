"""Determinism and logging setup, called once at process start.

No simulation code may depend on wall-clock time or unseeded randomness —
see standing constraint #6 (determinism: same inputs -> byte-identical
outputs).
"""

from __future__ import annotations

import logging
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
