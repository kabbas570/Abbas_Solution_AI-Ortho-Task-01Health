"""Instruction text → InstructionConfig.

Two paths behind one function:
- `parse_instruction(text)` — deterministic keyword parser (offline; a starting point you
  are welcome to improve or replace).
- `parse_instruction(text, use_llm=True)` — Claude parse via the Anthropic SDK
  (`claude-opus-5`, schema-validated structured output). Requires ANTHROPIC_API_KEY; falls
  back to the keyword parser on any failure so pipelines never hard-block on the network.

The parsed config is logged with every plan (SaMD auditability) — see plan meta.
"""

from __future__ import annotations

import os
import re

from contract import (
    ALL_FDI,
    DEFAULT_CAPS,
    FAST_CAPS,
    SAFE_CAPS,
    InstructionConfig,
)

_MOLARS = [f for f in ALL_FDI if f % 10 in (6, 7)]

_FDI_TOKEN = re.compile(r"\b([1-4])\s*[.\-]?\s*([1-8])\b")
# clinical quadrant notation (UR6, ul5 …): UR=1, UL=2, LL=3, LR=4
_QUAD_TOKEN = re.compile(r"\b([ul])\s*([lr])\s*([1-8])\b")
_QUAD_MAP = {("u", "r"): 1, ("u", "l"): 2, ("l", "l"): 3, ("l", "r"): 4}


_PROTECT_VERB = re.compile(
    r"do not move|don't move|don't touch|do not touch|avoid moving|\bleave\b|\bkeep\b|\bhold\b"
    r"|must not|may not move|not be moved|no movement|ankylos|must stay|remain stable"
    r"|remain unchanged|no changes? to|should not change|stay put|stay in place|do not touch"
    r"|no adjustments? to|freeze\b|spare\b|off limits"
)

# whole-arch immobility: an (upper|lower) arch/teeth mention plus an unambiguous hold cue
# in the same clause — "do not move any upper teeth", "leave the lower arch untouched",
# "the patient has declined upper-arch treatment", "no lower tooth may move"
_ARCH_WORD = re.compile(r"\b(upper|lower)\b[^.;]{0,40}?\b(?:arch|teeth|tooth)\b")
_ARCH_HOLD_CUE = re.compile(
    r"untouched|do not move|don't move|must not|may not move|not be moved|no movement"
    r"|declined|decided against|remain stable|remain unchanged|no changes?|stay put"
    r"|stay in place|off limits"
)


def _sentences(low: str) -> list[str]:
    # split on ';', newlines, em-dash clauses, or '.' followed by whitespace/end —
    # never inside "1.6"-style FDI tokens
    return [s for s in re.split(r"[;\n—]|\.(?=\s|$)", low) if s.strip()]


def _tooth_tokens(sent: str):
    for m in _FDI_TOKEN.finditer(sent):
        yield int(m.group(1)) * 10 + int(m.group(2))
    for m in _QUAD_TOKEN.finditer(sent):
        yield _QUAD_MAP[(m.group(1), m.group(2))] * 10 + int(m.group(3))


def _protected_teeth(text: str) -> list[int]:
    protected: set[int] = set()
    for sent in _sentences(text.lower()):
        arch = _ARCH_WORD.search(sent)
        if arch and _ARCH_HOLD_CUE.search(sent):
            jaws = (1, 2) if arch.group(1) == "upper" else (3, 4)
            protected.update(f for f in ALL_FDI if f // 10 in jaws)
        if not _PROTECT_VERB.search(sent):
            continue
        for fdi in _tooth_tokens(sent):
            if fdi in ALL_FDI:
                protected.add(fdi)
        if "molar" in sent:
            protected.update(_MOLARS)
        if ("posterior" in sent or "back teeth" in sent) and "arch" not in sent:
            protected.update(f for f in ALL_FDI if f % 10 >= 4)
    return sorted(protected)


def _goals(text: str) -> list[dict]:
    low = text.lower()
    goals: list[dict] = []

    upper_only = re.search(r"upper arch only|only the upper", low) or (
        "upper" in low and re.search(r"leave the lower[^.;]*(untouched|alone|as is)", low)
    )
    lower_only = re.search(r"lower arch only|only the lower", low) or (
        "lower" in low and re.search(r"leave the upper[^.;]*(untouched|alone|as is)", low)
    )
    arch = "upper" if upper_only else "lower" if lower_only else "both"

    anterior_scope = bool(
        re.search(r"(anterior|front teeth|front-teeth|incisor)", low)
        and re.search(r"(keep|leave|hold)[^.;]*(posterior|back teeth|where they are)", low)
    )
    region = "anterior" if anterior_scope else "full"

    goals.append({"type": "align", "arch": arch, "region": region})
    if re.search(r"close [^.;]*spacing|close [^.;]*gaps|diastema", low):
        goals.append({"type": "close_spacing", "arch": arch, "region": region})
    return goals


def parse_instruction_keywords(text: str) -> InstructionConfig:
    low = text.lower()
    cfg = InstructionConfig()
    cfg.protected_teeth = _protected_teeth(text)
    cfg.goals = _goals(text)

    if re.search(r"priorit\w+ speed|as fast|quick|wedding|event in \d+ months", low):
        cfg.objective_priority = {"speed": 0.7, "safety": 0.2, "aesthetics": 0.1}
        cfg.max_translation_per_stage_mm = FAST_CAPS["max_translation_per_stage_mm"]
        cfg.max_rotation_per_stage_deg = FAST_CAPS["max_rotation_per_stage_deg"]
    elif re.search(r"safest|gentle|minimal per-stage|conservative", low):
        cfg.objective_priority = {"speed": 0.1, "safety": 0.8, "aesthetics": 0.1}
        cfg.max_translation_per_stage_mm = SAFE_CAPS["max_translation_per_stage_mm"]
        cfg.max_rotation_per_stage_deg = SAFE_CAPS["max_rotation_per_stage_deg"]
    else:
        cfg.max_translation_per_stage_mm = DEFAULT_CAPS["max_translation_per_stage_mm"]
        cfg.max_rotation_per_stage_deg = DEFAULT_CAPS["max_rotation_per_stage_deg"]

    m = re.search(r"(\d+)\s*refinement", low)
    if m:
        cfg.refinements = int(m.group(1))
    m = re.search(r"(?:within|budget of|at most)\s*(\d+)\s*(?:stages|aligners|trays)", low)
    if m:
        cfg.stage_budget = int(m.group(1))
    return cfg


def parse_instruction_llm(text: str) -> InstructionConfig:
    """Schema-validated parse with the Anthropic SDK. Raises on any failure."""
    import anthropic
    from pydantic import BaseModel, Field

    class Goal(BaseModel):
        type: str = Field(description="align | close_spacing | level | procline")
        arch: str = Field(description="upper | lower | both")
        region: str = Field(description="anterior | posterior | full")

    class ParsedConfig(BaseModel):
        speed: float = Field(description="objective weight 0..1")
        safety: float = Field(description="objective weight 0..1")
        aesthetics: float = Field(description="objective weight 0..1")
        protected_teeth: list[int] = Field(description="FDI numbers that must not move")
        goals: list[Goal]
        refinements: int = 0
        stage_budget: int | None = None

    client = anthropic.Anthropic()
    resp = client.messages.parse(
        model="claude-opus-5",
        max_tokens=2048,
        output_config={"effort": "low"},
        system=(
            "You convert a dentist's free-text aligner instruction into a strict JSON config. "
            "FDI notation: '1.6' means 16. 'Molars' = FDI unit digits 6-7 in all quadrants. "
            "Only mark teeth protected when the text says not to move them. Objective weights "
            "must sum to 1."
        ),
        messages=[{"role": "user", "content": text}],
        output_format=ParsedConfig,
    )
    p = resp.parsed_output
    caps = FAST_CAPS if p.speed >= 0.5 else SAFE_CAPS if p.safety >= 0.5 else DEFAULT_CAPS
    return InstructionConfig(
        objective_priority={"speed": p.speed, "safety": p.safety, "aesthetics": p.aesthetics},
        protected_teeth=[f for f in p.protected_teeth if f in ALL_FDI],
        goals=[g.model_dump() for g in p.goals] or InstructionConfig().goals,
        refinements=p.refinements,
        stage_budget=p.stage_budget,
        **caps,
    )


def parse_instruction(text: str, use_llm: bool | None = None) -> InstructionConfig:
    """Parse text into a config. use_llm=None → LLM iff ANTHROPIC_API_KEY is set."""
    if use_llm is None:
        use_llm = bool(os.environ.get("ANTHROPIC_API_KEY")) and (
            os.environ.get("TASKPACK_LLM_PARSER", "0") == "1"
        )
    if use_llm:
        try:
            return parse_instruction_llm(text)
        except Exception:
            pass  # network/schema failure → deterministic fallback
    return parse_instruction_keywords(text)
