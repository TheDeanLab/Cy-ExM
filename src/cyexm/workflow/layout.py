from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RoundLayout:
    root: Path
    round: int

    @property
    def registration(self) -> Path:
        return self.root / "registration"

    @property
    def reference_crop(self) -> Path:
        return self.registration / "reference_crop_indices.json"

    @property
    def channels(self) -> Path:
        return self.root / "channels"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def log(self) -> Path:
        return self.logs / f"round-{self.round:03d}.log"

    def channel(self, source: str | PathLike[str]) -> Path:
        return self.channels / Path(source).name


@dataclass(frozen=True, slots=True)
class WorkflowLayout:
    root: Path

    @classmethod
    def for_dataset(
        cls,
        output_root: str | PathLike[str],
        dataset_id: str,
    ) -> WorkflowLayout:
        return cls(Path(output_root) / dataset_id)

    @property
    def plan_path(self) -> Path:
        return self.root / "scientific-plan.json"

    @property
    def workflow_runs(self) -> Path:
        return self.root / "workflow-runs"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def rounds(self) -> Path:
        return self.root / "rounds"

    def round(self, round_number: int) -> RoundLayout:
        return RoundLayout(
            root=self.rounds / f"round-{round_number:03d}",
            round=round_number,
        )

    def run_record(self, run_id: str) -> Path:
        return self.workflow_runs / f"{run_id}.json"

    def run_log(self, run_id: str) -> Path:
        return self.logs / f"{run_id}.log"


__all__ = ("RoundLayout", "WorkflowLayout")
