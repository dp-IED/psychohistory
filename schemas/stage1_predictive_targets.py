"""Schema for Stage1 predictive-coding rank-change targets."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class Stage1PredictiveTargetsManifest(BaseModel):
    manifest_version: str = "stage1_predictive_targets_v1"
    objective_version: str = "predictive_coding_rankchange_v1"
    embedding_version: str = Field(min_length=1)
    source_manifest_path: str = Field(min_length=1)
    row_count: int = Field(ge=0)
    horizons_days: list[int] = Field(min_length=1)
    horizon_weights: list[float] = Field(min_length=1)
    target_files: dict[str, str] = Field(
        description="Map from horizon days (stringified int) to .npy basename with float32 targets shape=(row_count,).",
    )
    actor_missing_rows: int = Field(ge=0, default=0)
    rows_without_first_seen: int = Field(ge=0, default=0)

    @model_validator(mode="after")
    def _validate_lengths(self) -> "Stage1PredictiveTargetsManifest":
        if len(self.horizons_days) != len(self.horizon_weights):
            raise ValueError("horizons_days and horizon_weights must have same length")
        if sorted(self.horizons_days) != self.horizons_days:
            raise ValueError("horizons_days must be sorted ascending")
        expected_keys = {str(h) for h in self.horizons_days}
        got_keys = set(self.target_files.keys())
        if expected_keys != got_keys:
            raise ValueError(f"target_files keys must equal horizons_days set: expected={expected_keys} got={got_keys}")
        return self


__all__ = ["Stage1PredictiveTargetsManifest"]
