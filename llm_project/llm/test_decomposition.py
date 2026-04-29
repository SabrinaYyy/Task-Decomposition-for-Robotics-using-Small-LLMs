import json
import torch
import argparse
import re
from llm_functions import MyApproach


# I use the model/prompt pair that gave the best F1 score in my evaluation.
MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# This prompt is intentionally strict because the executor expects a small
# closed set of actions and a predictable JSON schema.
SYSTEM_BASE = """You are a robot planner. Decompose the given instruction into a sequence of robot actions.

AVAILABLE ACTIONS - you MUST use ONLY these, spelled exactly as shown:
  - move_pose
  - close_gripper
  - open_gripper

OUTPUT FORMAT - output ONLY a valid JSON object, no extra text, no markdown fences:
{
  "step_1": {
    "action": "<one action from the list above>",
    "object_a": "<object being manipulated, or null>",
    "object_b": "<target/reference object, or null>",
    "spatial_relation": "<spatial relation, or null>"
  },
  "step_2": { ... }
}

SEQUENCING RULES:
1. To pick up an object: FIRST move to the object, THEN close_gripper on it.
2. To place an object: FIRST pick it up, THEN move to the destination, THEN open_gripper.
3. Never go to the destination before picking the object up.
4. If the task only requires a push, use move steps without gripping.
5. If a field is not applicable, set it to null (not the string "null").
6. For "move X to the left/right of Y", output exactly:
   step_1 move_pose to X, step_2 close_gripper on X, step_3 move_pose with object_b Y and spatial_relation left_of/right_of, step_4 open_gripper.
7. Do not put spatial relations inside object names. Use object_b "cup" and spatial_relation "left_of", not object_b "left_of_cup"."""

ONE_SHOT_EXAMPLE = """
EXAMPLE:
Instruction: put the cube in the cup
Output JSON:
{
  "step_1": {"action": "move_pose", "object_a": null, "object_b": "cube", "spatial_relation": null},
  "step_2": {"action": "close_gripper", "object_a": "cube", "object_b": null, "spatial_relation": null},
  "step_3": {"action": "move_pose", "object_a": "cube", "object_b": "cup", "spatial_relation": "above"},
  "step_4": {"action": "open_gripper", "object_a": "cube", "object_b": "cup", "spatial_relation": "in"}
}"""


def parse_json_output(raw_output):
    """I keep parsing tolerant because small LLMs sometimes add fences or text."""
    clean_output = re.sub(r"```(?:json)?", "", raw_output).replace("```", "").strip()
    try:
        return json.loads(clean_output)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", clean_output, re.DOTALL)
    if match:
        return json.loads(match.group())

    raise json.JSONDecodeError("No valid JSON object found", clean_output, 0)

def main():
    parser = argparse.ArgumentParser(description="Test task decomposition")
    parser.add_argument("--task", type=str, default="pick the cup", help="Instruction to test")
    parser.add_argument(
        "--output",
        type=str,
        default="task.json",
        help="Output JSON plan path",
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading {MODEL_NAME} on {device}...")
    print("Selected configuration: SmolLM2-1.7B-Instruct with one-shot prompting")
    my_approach = MyApproach(MODEL_NAME, device=device)

    instruction = args.task
    # I combine the base rules with exactly one example because one-shot was the
    # strongest setting in my results table.
    context = SYSTEM_BASE + ONE_SHOT_EXAMPLE
    
    query = f"Instruction: {instruction}\nOutput JSON:"

    print(f"\nExecuting query for instruction: '{instruction}'")
    
    # This is the only LLM call in the text-to-robot path.
    result = my_approach.generate_response(context, query, verbose=False)
    
    print("\n--- Raw Model Output ---")
    print(result)
    
    # I save only valid JSON because the ROS executor reads this file directly.
    print("\n--- JSON Parsing Test ---")
    try:
        parsed_json = parse_json_output(result)
        print("Success! Valid JSON detected:")
        print(json.dumps(parsed_json, indent=2))
        
        with open(args.output, "w") as f:
            json.dump(parsed_json, f, indent=2)
        print(f"Output successfully saved to {args.output}")
    except json.JSONDecodeError as exc:
        print("Warning: The model's output was not perfectly formatted as JSON.")
        print(f"JSON error: {exc}")

if __name__ == "__main__":
    main()
