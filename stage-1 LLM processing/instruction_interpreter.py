"""Interpret orthodontic prescriptions with an OpenAI-compatible chat model.

The public API is deliberately small: :class:`InstructionInterpreter` validates every
model response with Pydantic, retries malformed responses, and can return either a model
or JSON text.  A client can be injected, which keeps parsing tests independent of a
network and of the OpenAI package.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Protocol

try:
    from .movement_plan_schema import MovementPlan, validate_movement_plan
except ImportError:
    from movement_plan_schema import MovementPlan, validate_movement_plan


class ChatCompletions(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class LLMClient(Protocol):
    chat: Any


SYNONYM_GROUPS: dict[str, tuple[str, ...]] = {
    "procline_lower_incisors": (
        "procline lower incisors", "flare lower incisors", "advance lower incisors",
        "labialize lower incisors",
    ),
    "align_arch": (
        "align arch", "straighten teeth", "resolve crowding", "correct rotations",
    ),
}


def canonicalize_instruction(text: str) -> list[str]:
    """Return canonical goals detected from common clinical synonyms."""
    low = re.sub(r"\s+", " ", text.lower()).strip()
    return [canonical for canonical, phrases in SYNONYM_GROUPS.items()
            if any(phrase in low for phrase in phrases)]


def _canonical_hints(text: str) -> str:
    goals = canonicalize_instruction(text)
    if not goals:
        return "No predefined synonym was detected; infer goals from the instruction."
    return "Detected canonical goals: " + ", ".join(goals) + "."


def _extract_response_text(response: Any) -> str:
    """Handle the object and dictionary response shapes used by compatible clients."""
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError):
        content = response["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item) for item in content
        )
    return str(content).strip()


def _json_object(text: str) -> dict[str, Any]:
    """Decode strict JSON, tolerating only an accidental markdown fence."""
    cleaned = text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("LLM output must be a JSON object")
    return value


def _normalize_payload(payload: dict[str, Any], instruction: str) -> dict[str, Any]:
    """Add deterministic canonical synonym goals before schema validation."""
    normalized = dict(payload)
    existing_goals = normalized.get("goals", [])
    if not isinstance(existing_goals, list):
        return normalized
    normalized["goals"] = list(dict.fromkeys(
        [*existing_goals, *canonicalize_instruction(instruction)]
    ))
    return normalized


def _jaw_of(fdi: int) -> str:
    return "upper" if fdi // 10 in (1, 2) else "lower"


def _region_of(fdi: int) -> str:
    return "anterior" if fdi % 10 <= 3 else "posterior"


def _tooth_rationale(fdi: int, goals: list[str]) -> str:
    """Give the few-shot examples a real (jaw/region-aware) cue instead of boilerplate."""
    goal = goals[0] if goals else "the prescription"
    return f"Coordinated {_jaw_of(fdi)}-arch {_region_of(fdi)} movement consistent with '{goal}'."


def load_training_examples(training_root: str | Path, max_examples: int = 8) -> list[dict[str, Any]]:
    """Extract compact, diverse few-shot examples from training case folders.

    Gold transforms provide moved/fixed teeth; canonical goals and rationale are derived
    from the prescription because the task data does not store those labels explicitly.
    Cases are chosen to cover distinct canonical-goal signatures (not just the first N
    alphabetically) so the few-shot set demonstrates varied move/fixed patterns.
    """
    if max_examples < 0:
        raise ValueError("max_examples must not be negative")
    root = Path(training_root)
    candidates: list[dict[str, Any]] = []
    for case_dir in sorted(path for path in root.glob("*") if path.is_dir()):
        instruction_path = case_dir / "instruction.txt"
        gold_path = case_dir / "gold_transforms.json"
        if not instruction_path.exists() or not gold_path.exists():
            continue
        try:
            instruction = instruction_path.read_text(encoding="utf-8").strip()
            transforms = json.loads(gold_path.read_text(encoding="utf-8"))["transforms"]
            moved = sorted(int(fdi) for fdi, value in transforms.items()
                           if any(abs(float(component)) > 1e-6 for component in value["t_mm"])
                           or abs(float(value["q"][3])) < 0.999999)
            all_teeth = sorted(int(fdi) for fdi in transforms)
            fixed = sorted(set(all_teeth) - set(moved))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        goals = canonicalize_instruction(instruction) or ["follow_prescription"]
        candidates.append({
            "goal_signature": tuple(sorted(goals)),
            "instruction": instruction,
            "output": MovementPlan(
                goals=goals,
                move_teeth=moved,
                fixed_teeth=fixed,
                movement_rationale={str(tooth): _tooth_rationale(tooth, goals) for tooth in moved},
                confidence=0.7,
            ).as_json_dict(),
        })

    examples: list[dict[str, Any]] = []
    seen_signatures: set[tuple[str, ...]] = set()
    for candidate in candidates:
        if candidate["goal_signature"] in seen_signatures:
            continue
        seen_signatures.add(candidate["goal_signature"])
        examples.append({"instruction": candidate["instruction"], "output": candidate["output"]})
        if len(examples) >= max_examples:
            return examples
    for candidate in candidates:
        if len(examples) >= max_examples:
            break
        entry = {"instruction": candidate["instruction"], "output": candidate["output"]}
        if entry not in examples:
            examples.append(entry)
    return examples


class InstructionInterpreter:
    """Call an OpenAI-compatible model and return a validated movement plan."""

    def __init__(self, client: LLMClient | None = None, model: str | None = None,
                 base_url: str | None = None, api_key: str | None = None,
                 max_retries: int = 3, few_shot_examples: list[dict[str, Any]] | None = None,
                 extra_body: dict[str, Any] | None = None) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        self.client = client or self._create_client(api_key=api_key, base_url=base_url)
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.max_retries = max_retries
        self.few_shot_examples = few_shot_examples or []
        self.extra_body = extra_body

    @staticmethod
    def _create_client(api_key: str | None, base_url: str | None) -> LLMClient:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise ImportError("Install the OpenAI client with `pip install openai`.") from error
        return OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"),
                      base_url=base_url or os.getenv("OPENAI_BASE_URL"))

    def interpret(self, instruction: str) -> MovementPlan:
        """Interpret instruction text, retrying invalid JSON or schema output."""
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must not be empty")
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt()},
        ]
        for example in self.few_shot_examples:
            messages.extend([
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": json.dumps(example["output"])},
            ])
        messages.append({"role": "user", "content": (
            f"Instruction:\n{instruction}\n\n{_canonical_hints(instruction)}"
        )})
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            if attempt:
                messages.append({"role": "user", "content":
                                 f"Return corrected JSON only. Validation error: {last_error}"})
            try:
                create_kwargs: dict[str, Any] = dict(
                    model=self.model, messages=messages, temperature=0,
                    response_format={"type": "json_object"},
                )
                if self.extra_body is not None:
                    create_kwargs["extra_body"] = self.extra_body
                response = self.client.chat.completions.create(**create_kwargs)
                payload = _json_object(_extract_response_text(response))
                return validate_movement_plan(_normalize_payload(payload, instruction))
            except (json.JSONDecodeError, TypeError, ValueError, KeyError, IndexError) as error:
                last_error = error
        raise ValueError(
            f"LLM failed to return a valid movement plan after {self.max_retries} attempts"
        ) from last_error

    def interpret_json(self, instruction: str) -> str:
        """Interpret instruction text and serialize the validated result as JSON only."""
        return json.dumps(self.interpret(instruction).as_json_dict(), separators=(",", ":"))

    @staticmethod
    def _system_prompt() -> str:
        return (Path(__file__).with_name("prompts").joinpath("instruction_interpreter_prompt.md")
                .read_text(encoding="utf-8"))


def interpret_instruction(instruction: str, **kwargs: Any) -> MovementPlan:
    """Convenience wrapper around :class:`InstructionInterpreter`."""
    return InstructionInterpreter(**kwargs).interpret(instruction)
