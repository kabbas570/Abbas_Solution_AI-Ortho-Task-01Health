# Stage-1 LLM Instruction Processing

This folder contains the Stage-1 instruction interpreter used to convert free-text orthodontic instructions into validated tooth-level movement intent.

The script reads each case instruction, calls an OpenAI-compatible LLM, validates the response with a Pydantic schema, and writes a `movement_plan.json` file back into each case folder. The generated file contains clinical goals, teeth expected to move, teeth expected to remain fixed, rationale for moved teeth, and a confidence score.

## Files

| File | Purpose |
|---|---|
| `generate_movement_plans.py` | Main batch runner for train/eval case folders. |
| `instruction_interpreter.py` | LLM client wrapper, prompt loading, response parsing, retries, and few-shot examples. |
| `movement_plan_schema.py` | Pydantic schema and validation rules for `MovementPlan`. |
| `prompts/instruction_interpreter_prompt.md` | System prompt used by the LLM to produce valid JSON. |

## Requirements

Install the runtime dependencies in your Python environment:

```bash
pip install openai pydantic
```

The script expects the project root to contain the task data folders:

```text
train/
eval/
```

Few-shot examples are loaded from `train/*/instruction.txt` and `train/*/gold_transforms.json`.

## Run

From the project root:

```bash
python "stage-1 LLM processing/generate_movement_plans.py" --root train
python "stage-1 LLM processing/generate_movement_plans.py" --root eval
```

From inside this folder:

```bash
python generate_movement_plans.py --root ../train
python generate_movement_plans.py --root ../eval
```

## Output

For each processed case folder, the script writes:

```text
movement_plan.json
```

Example structure:

```json
[
  {
    "id": "i0",
    "plan": {
      "goals": ["align_arch"],
      "move_teeth": [11, 12, 13],
      "fixed_teeth": [16, 26],
      "movement_rationale": {
        "11": "anterior alignment"
      },
      "confidence": 0.8
    }
  }
]
```

## Notes

- The LLM output is not trusted directly; it is validated by `MovementPlan` before saving.
- Invalid JSON or schema-invalid responses are retried.
- For single train instructions, the instruction id is saved as `i0` to match the task contract.
- The downstream model can use `movement_plan.json` as Stage-1 move/fixed instruction features.
