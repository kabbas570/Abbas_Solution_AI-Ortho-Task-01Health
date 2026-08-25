"""Pydantic models for LLM-generated orthodontic movement plans."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


VALID_FDI = frozenset(
    [17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27]
    + [47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37]
)


class MovementPlan(BaseModel):
    """Validated, geometry-independent interpretation of one prescription."""

    model_config = ConfigDict(extra="forbid")

    goals: list[str] = Field(default_factory=list, description="Canonical clinical goals")
    move_teeth: list[int] = Field(default_factory=list, description="FDI teeth allowed to move")
    fixed_teeth: list[int] = Field(default_factory=list, description="FDI teeth that must stay fixed")
    movement_rationale: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence from 0 to 1")

    @field_validator("move_teeth", "fixed_teeth")
    @classmethod
    def validate_fdi_numbers(cls, teeth: list[int]) -> list[int]:
        if len(teeth) != len(set(teeth)):
            raise ValueError("tooth lists must not contain duplicates")
        invalid = sorted(set(teeth) - VALID_FDI)
        if invalid:
            raise ValueError(f"invalid FDI tooth numbers: {invalid}")
        return sorted(teeth)

    @field_validator("goals")
    @classmethod
    def validate_goals(cls, goals: list[str]) -> list[str]:
        cleaned = [goal.strip() for goal in goals if goal.strip()]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("goals must not contain duplicates")
        return cleaned

    @model_validator(mode="after")
    def teeth_cannot_move_and_be_fixed(self) -> "MovementPlan":
        overlap = set(self.move_teeth) & set(self.fixed_teeth)
        if overlap:
            raise ValueError(f"a tooth cannot be both moved and fixed: {sorted(overlap)}")
        expected_rationale = {str(tooth) for tooth in self.move_teeth}
        actual_rationale = set(self.movement_rationale)
        if actual_rationale != expected_rationale:
            missing = sorted(expected_rationale - actual_rationale)
            extra = sorted(actual_rationale - expected_rationale)
            raise ValueError(
                "movement rationale keys must exactly match moved teeth; "
                f"missing={missing}, extra={extra}"
            )
        if any(not reason.strip() for reason in self.movement_rationale.values()):
            raise ValueError("movement rationale values must not be empty")
        return self

    def as_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary using stable field names."""
        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")
        return self.dict()


def validate_movement_plan(payload: Any) -> MovementPlan:
    """Validate a decoded JSON payload on both Pydantic v1 and v2."""
    if hasattr(MovementPlan, "model_validate"):
        return MovementPlan.model_validate(payload)
    return MovementPlan.parse_obj(payload)