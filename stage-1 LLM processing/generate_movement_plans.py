"""Interpret orthodontic instructions from a file and save validated movement plans.

Accepts either:
  - a JSON file containing a list of {"id": ..., "text": ...} entries
    (e.g. eval/*/instructions.json), or
  - a plain .txt file containing a single instruction (e.g. train/*/instruction.txt).

For each instruction, the configured LLM is called through
:class:`InstructionInterpreter`, its response is validated against the
``MovementPlan`` Pydantic schema, and the results are written as JSON.

Example:
    python appraoch_1.py --instructions-file ../eval/prod_0006/instructions.json \
        --output movement_plans.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("SSL_CERT_DIR", None)
os.environ.pop("REQUESTS_CA_BUNDLE", None)

from openai import OpenAI

try:
    from .instruction_interpreter import InstructionInterpreter, load_training_examples
except ImportError:
    from instruction_interpreter import InstructionInterpreter, load_training_examples

DEFAULT_BASE_URL = "https://llama-3-3-70b-instruct-thea.runai-inference.eu-bcm11.jnj.com/v1"
DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"
# Root folder processed when neither --root nor --instructions-file is given.
DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "eval"


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Interpret orthodontic instructions and save validated movement plans."
    )
    parser.add_argument(
        "--instructions-file",
        type=Path,
        help="Path to instructions.json (list of {id, text}) or a plain instruction .txt file. "
             "If omitted (and --root is not given), the whole train/ folder is processed via --root.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="Directory containing case subfolders (e.g. train/), each with an instruction.txt "
             "or instructions.json. Every case is processed and its plan is saved back into "
             "that same subfolder as movement_plan.json. Overrides --instructions-file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path; defaults to <instructions-file stem>_movement_plans.json next to the input. "
             "Ignored when --root is used.",
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        help="Training directory used for few-shot examples; defaults to ../train.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=8,
        help="Maximum number of training examples to include (default: 8).",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible endpoint.")
    parser.add_argument("--api-key", default="dummy", help="API key for the endpoint.")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum attempts for malformed or schema-invalid model output.",
    )
    return parser


def load_instructions(path: Path) -> list[dict[str, str]]:
    """Load one or more instructions from a .json list or a plain .txt file."""
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a JSON list of instruction entries")
        entries = []
        for index, item in enumerate(payload):
            text = item["text"]
            entries.append({"id": str(item.get("id", index)), "text": text})
        return entries
    text = path.read_text(encoding="utf-8").strip()
    # "i0" matches contract.case_instructions() for single-instruction train cases.
    return [{"id": "i0", "text": text}]


def find_case_instructions_file(case_dir: Path) -> Path | None:
    """Return the instructions.json or instruction.txt inside a case folder, if any."""
    for name in ("instructions.json", "instruction.txt"):
        candidate = case_dir / name
        if candidate.exists():
            return candidate
    return None


def interpret_all(interpreter: InstructionInterpreter,
                   instructions: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Interpret every instruction entry, collecting a plan or error per id."""
    results: list[dict[str, Any]] = []
    for entry in instructions:
        try:
            plan = interpreter.interpret(entry["text"])
            results.append({"id": entry["id"], "plan": plan.as_json_dict()})
        except ValueError as error:
            results.append({"id": entry["id"], "error": str(error)})
    return results


def process_root(interpreter: InstructionInterpreter, root: Path) -> None:
    """Interpret every case folder under root, saving movement_plan.json back into each."""
    case_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    for case_dir in case_dirs:
        instructions_file = find_case_instructions_file(case_dir)
        if instructions_file is None:
            continue
        instructions = load_instructions(instructions_file)
        results = interpret_all(interpreter, instructions)
        output = case_dir / "movement_plan.json"
        output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        sys.stdout.write(f"Wrote {len(results)} movement plan(s) to {output}\n")


def main(argv: list[str] | None = None) -> int:
    """Interpret instructions and write validated plans as JSON."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    try:
        training_root = args.training_root
        if training_root is None:
            # Few-shot examples need gold_transforms.json, which only exists in train/.
            training_root = Path(__file__).resolve().parent.parent / "train"
        examples = load_training_examples(training_root, args.max_examples)

        client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=300.0)
        interpreter = InstructionInterpreter(
            client=client,
            model=args.model,
            max_retries=args.max_retries,
            few_shot_examples=examples,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )

        if args.root is not None or args.instructions_file is None:
            root = args.root or DEFAULT_ROOT
            if not root.is_dir():
                raise ValueError(f"--root is not a directory: {root}")
            process_root(interpreter, root)
            return 0

        instructions_file = args.instructions_file
        if not instructions_file.exists():
            raise ValueError(f"--instructions-file does not exist: {instructions_file}")
        instructions = load_instructions(instructions_file)
        results = interpret_all(interpreter, instructions)

        output = args.output
        if output is None:
            output = instructions_file.with_name(
                f"{instructions_file.stem}_movement_plans.json"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        sys.stdout.write(f"Wrote {len(results)} movement plan(s) to {output}\n")
    except (OSError, ValueError, ImportError, KeyError) as error:
        parser.exit(1, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())