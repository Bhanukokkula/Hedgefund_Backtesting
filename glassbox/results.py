"""Typed results store: every reported metric is a {target, result} pair.

`target` is set once, up front, as a stated goal/expectation. `result` is
filled in later from an actual run. The store never lets `result` writes
touch `target` — that asymmetry is the point: this file is what lets the
portfolio website flip a project from roadmap to completed by filling in
`result` only, without anyone being able to quietly move the goalposts.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class Metric(BaseModel):
    target: str | float | int | bool | None = None
    result: str | float | int | bool | None = None
    note: str | None = None


class ResultsStore(BaseModel):
    metrics: dict[str, Metric] = {}

    def set_target(
        self, key: str, target: str | float | int | bool, note: str | None = None
    ) -> None:
        existing = self.metrics.get(key, Metric())
        existing.target = target
        if note is not None:
            existing.note = note
        self.metrics[key] = existing

    def set_result(
        self, key: str, result: str | float | int | bool, note: str | None = None
    ) -> None:
        """Set the observed result for `key`. Never touches `target`.

        Raises if `key` has no target set yet — results should land against
        a pre-declared target, not be invented after the fact.
        """
        if key not in self.metrics or self.metrics[key].target is None:
            raise KeyError(f"no target set for metric '{key}'; call set_target() first")
        self.metrics[key].result = result
        if note is not None:
            self.metrics[key].note = note

    @classmethod
    def load(cls, path: Path | str) -> ResultsStore:
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path) as f:
            return cls.model_validate(json.load(f))

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=2))
            f.write("\n")
