"""
test_llms.py
============
Evaluation harness for robot task-decomposition using small LLMs.
Built on top of MyApproach from llm_functions.py.

Research questions:
  RQ1 – Action-sequence accuracy (precision, recall, F1 - action level)
  RQ2 – Prompting strategy effect (zero-shot, one-shot, few-shot)

Usage:
  python test_llms.py --models qwen --shots zero
  python test_llms.py --models all --shots zero one few --output results/
  python test_llms.py --models smollm2 --shots zero --tasks 0 1
"""

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from llm_functions import MyApproach


# CONSTANTS
VALID_ACTIONS = ["move_pose", "open_gripper", "close_gripper"]
VALID_ACTIONS_SET = set(VALID_ACTIONS)

MODEL_IDS = {
    "smollm2": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "qwen":    "Qwen/Qwen2.5-1.5B-Instruct",
    "llama":   "meta-llama/Llama-3.2-1B-Instruct",
    "falcon":  "tiiuae/Falcon3-1B-Instruct",
}


# EVALUATION TASKS
TASKS = [
    {
        "instruction": "put the blue block inside the yellow bowl",
        "ground_truth": {
            "step_1": {"action": "move_pose",      "object_a": None,          "object_b": "blue_block",  "spatial_relation": None},
            "step_2": {"action": "close_gripper", "object_a": "blue_block", "object_b": None,           "spatial_relation": None},
            "step_3": {"action": "move_pose",       "object_a": "blue_block", "object_b": "yellow_bowl", "spatial_relation": "above"},
            "step_4": {"action": "open_gripper",  "object_a": "blue_block", "object_b": "yellow_bowl", "spatial_relation": "in"},
        },
    },
    {
        "instruction": "move the green cube to the left of the blue bowl",
        "ground_truth": {
            "step_1": {"action": "move_pose",      "object_a": None,          "object_b": "green_cube", "spatial_relation": None},
            "step_2": {"action": "close_gripper", "object_a": "green_cube", "object_b": None,          "spatial_relation": None},
            "step_3": {"action": "move_pose",       "object_a": "green_cube", "object_b": "blue_bowl",  "spatial_relation": "left_of"},
            "step_4": {"action": "open_gripper",  "object_a": "green_cube", "object_b": "blue_bowl",  "spatial_relation": "left_of"},
        },
    },
    {
        "instruction": "place the yellow block on top of the green block",
        "ground_truth": {
            "step_1": {"action": "move_pose",      "object_a": None,            "object_b": "yellow_block", "spatial_relation": None},
            "step_2": {"action": "close_gripper", "object_a": "yellow_block", "object_b": None,            "spatial_relation": None},
            "step_3": {"action": "move_pose",       "object_a": "yellow_block", "object_b": "green_block",  "spatial_relation": "above"},
            "step_4": {"action": "open_gripper",  "object_a": "yellow_block", "object_b": "green_block",  "spatial_relation": "on"},
        },
    },
    {
        "instruction": "push the blue bowl behind the green bowl",
        "ground_truth": {
            "step_1": {"action": "move_pose", "object_a": None,        "object_b": "blue_bowl",  "spatial_relation": None},
            "step_2": {"action": "move_pose", "object_a": "blue_bowl", "object_b": "green_bowl", "spatial_relation": "behind"},
        },
    },
    {
        "instruction": "put the green bowl next to the yellow block",
        "ground_truth": {
            "step_1": {"action": "move_pose",      "object_a": None,          "object_b": "green_bowl",    "spatial_relation": None},
            "step_2": {"action": "close_gripper", "object_a": "green_bowl", "object_b": None,            "spatial_relation": None},
            "step_3": {"action": "move_pose",       "object_a": "green_bowl", "object_b": "yellow_block", "spatial_relation": "next_to"},
            "step_4": {"action": "open_gripper",  "object_a": "green_bowl", "object_b": "yellow_block", "spatial_relation": "next_to"},
        },
    },
    {
        "instruction": "pick the blue box",
        "ground_truth": {
            "step_1": {"action": "move_pose",      "object_a": None,        "object_b": "blue_box", "spatial_relation": None},
            "step_2": {"action": "close_gripper", "object_a": "blue_box", "object_b": None,        "spatial_relation": None},
        },
    },
    {
        "instruction": "close the gripper",
        "ground_truth": {
            "step_1": {"action": "close_gripper", "object_a": None, "object_b": None, "spatial_relation": None},
        },
    },
]


# PROMPT TEMPLATES
SYSTEM_BASE = """You are a robot planner. Decompose the given instruction into a sequence of robot actions.

AVAILABLE ACTIONS — you MUST use ONLY these, spelled exactly as shown:
  • move_pose
  • close_gripper
  • open_gripper

OUTPUT FORMAT — output ONLY a valid JSON object, no extra text, no markdown fences:
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
1. To pick up an object: FIRST move the object (object_b = item), THEN close_gripper on it.
2. To place an object: FIRST pick it up, THEN move destination (object_b = destination), THEN open_gripper.
3. Never go to the destination before picking the object up.
4. If the task only requires a push, use move steps without gripping.
5. If a field is not applicable, set it to null (not the string "null")."""

EXAMPLE_1 = """
EXAMPLE:
Instruction: push the cube into the red area
Output JSON:
{
  "step_1": {"action": "move_pose",  "object_a": null, "object_b": "cube",     "spatial_relation": null},
  "step_2": {"action": "move_pose",  "object_a": "cube", "object_b": "red_area", "spatial_relation": "into"}
}"""

EXAMPLE_2 = """
EXAMPLE:
Instruction: put the red cube in the blue bowl
Output JSON:
{
  "step_1": {"action": "move_pose",      "object_a": null,        "object_b": "red_cube",  "spatial_relation": null},
  "step_2": {"action": "close_gripper","object_a": "red_cube", "object_b": null,          "spatial_relation": null},
  "step_3": {"action": "move_pose",      "object_a": "red_cube", "object_b": "blue_bowl",  "spatial_relation": "above"},
  "step_4": {"action": "open_gripper", "object_a": "red_cube", "object_b": "blue_bowl",  "spatial_relation": "in"}
}"""

EXAMPLE_3 = """
EXAMPLE:
Instruction: pick the green block
Output JSON:
{
  "step_1": {"action": "move_pose",      "object_a": null,          "object_b": "green_block", "spatial_relation": null},
  "step_2": {"action": "close_gripper","object_a": "green_block", "object_b": null,            "spatial_relation": null}
}"""

def build_system_prompt(shot: str) -> str:
    if shot == "zero":
        return SYSTEM_BASE
    elif shot == "one":
        return SYSTEM_BASE + EXAMPLE_2
    else:
        return SYSTEM_BASE + EXAMPLE_1 + EXAMPLE_2 + EXAMPLE_3


# JSON PARSING

def parse_json_output(raw: str) -> Optional[Dict]:
    """Parse model output as JSON, tolerating markdown fences and preamble."""
    clean = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


# EVALUATION METRICS

def evaluate_action(parsed_json: Optional[Dict], ground_truth: Dict) -> Dict:
    """
    Evaluate predicted JSON sequence against ground truth at the Action level.
    """
    gt_steps = list(ground_truth.keys())
    n_gt = len(gt_steps)

    if parsed_json is None:
        return {"tp": 0, "fp": 0, "fn": n_gt, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_steps = list(parsed_json.keys())
    n_pred = len(pred_steps)

    tp = 0
    fp = 0
    fn = 0

    for i in range(max(n_gt, n_pred)):
        if i < n_gt and i < n_pred:
            gt_action = ground_truth[gt_steps[i]].get("action")
            # We assume dict values for prediction are dicts.
            pred_step_val = parsed_json[pred_steps[i]]
            pred_action = pred_step_val.get("action") if isinstance(pred_step_val, dict) else None

            if pred_action == gt_action:
                tp += 1
            else:
                fp += 1
                fn += 1
        elif i < n_pred:
            # Model produced extra steps
            fp += 1
        elif i < n_gt:
            # Model missed steps
            fn += 1

    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


# TASK RESULT DATACLASS

@dataclass
class TaskResult:
    model:           str
    shot:            str
    instruction:     str
    raw_output:      str
    parsed_json:     Optional[Dict]
    action_metrics:  Dict


# MAIN EVALUATION LOOP

def run_evaluation(
    model_keys: List[str],
    shot_modes: List[str],
    output_dir: str = "results",
    tasks: List[Dict] = TASKS,
) -> List[TaskResult]:

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_results: List[TaskResult] = []

    import torch

    for model_key in model_keys:
        model_id = MODEL_IDS[model_key]
        device   = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"\n{'='*70}")
        print(f"  MODEL : {model_id}  ({device})")
        print(f"{'='*70}")

        try:
            approach = MyApproach(model_id, device=device)
        except Exception as e:
            print(f"  [ERROR] Could not load model: {e}")
            continue

        for shot in shot_modes:
            system_prompt = build_system_prompt(shot)
            print(f"\n  ── Shot mode: {shot.upper()} ──")

            for task in tasks:
                instruction  = task["instruction"]
                ground_truth = task["ground_truth"]
                query        = f"Instruction: {instruction}\nOutput JSON:"

                print(f"\n    Task: {instruction!r}")

                try:
                    raw = approach.generate_response(
                        context=system_prompt,
                        query=query,
                        verbose=False,
                    )
                except Exception as e:
                    print(f"      [ERROR during generation]: {e}")
                    continue

                parsed = parse_json_output(raw)
                action_metrics = evaluate_action(parsed, ground_truth)

                result = TaskResult(
                    model=model_key, shot=shot, instruction=instruction,
                    raw_output=raw, parsed_json=parsed,
                    action_metrics=action_metrics,
                )
                all_results.append(result)

                # Console summary
                preview = raw[:80].strip().replace("\n", " ")
                print(f"      Output : {preview!r}{'...' if len(raw) > 80 else ''}")
                if parsed is not None:
                    print(f"      JSON   : OK")
                else:
                    print(f"      JSON   : FAILED")
                print(f"      Action Level - TP={action_metrics['tp']} FP={action_metrics['fp']} FN={action_metrics['fn']} "
                      f"P={action_metrics['precision']:.2f} R={action_metrics['recall']:.2f} F1={action_metrics['f1']:.2f}")

        del approach
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_results


# AGGREGATE REPORT
def aggregate_and_report(results: List[TaskResult], output_dir: str = "results"):
    groups: Dict[Tuple[str, str], List[TaskResult]] = {}
    for r in results:
        groups.setdefault((r.model, r.shot), []).append(r)

    summary = {}
    for (model, shot), rlist in groups.items():

        def _mean(fn):
            vals = [fn(r) for r in rlist]
            return float(np.mean(vals)) if vals else 0.0

        summary[(model, shot)] = {
            "n_tasks":          len(rlist),
            "json_rate":        _mean(lambda r: 1.0 if r.parsed_json is not None else 0.0),
            "action_tp":        _mean(lambda r: r.action_metrics["tp"]),
            "action_fp":        _mean(lambda r: r.action_metrics["fp"]),
            "action_fn":        _mean(lambda r: r.action_metrics["fn"]),
            "action_precision": _mean(lambda r: r.action_metrics["precision"]),
            "action_recall":    _mean(lambda r: r.action_metrics["recall"]),
            "action_f1":        _mean(lambda r: r.action_metrics["f1"]),
        }

    # Print table
    print("\n" + "─" * 70)
    print("AGGREGATE RESULTS")
    print("─" * 70)
    print(f"{'Model':<12} {'Shot':<6} {'JSON%':>6} {'ActP':>6} {'ActR':>6} {'ActF1':>6}")
    print("─" * 70)
    for (model, shot), m in sorted(summary.items()):
        print(f"{model:<12} {shot:<6} {m['json_rate']*100:>5.1f}% "
              f"{m['action_precision']:>6.3f} {m['action_recall']:>6.3f} {m['action_f1']:>6.3f}")
    print("─" * 70)

    out_path = Path(output_dir) / "summary.json"
    with open(out_path, "w") as f:
        json.dump({f"{m}_{s}": v for (m, s), v in summary.items()}, f, indent=2)
    print(f"\nSummary → {out_path}")

    detail_path = Path(output_dir) / "detail.json"
    with open(detail_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)
    print(f"Detail  → {detail_path}")

    return summary


# CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate small LLMs for robot task decomposition."
    )
    parser.add_argument("--models", nargs="+", default=["qwen"],
                        choices=list(MODEL_IDS.keys()) + ["all"])
    parser.add_argument("--shots", nargs="+", default=["zero", "one", "few"],
                        choices=["zero", "one", "few"])
    parser.add_argument("--output", default="results")
    parser.add_argument("--tasks", nargs="+", type=int, default=None,
                        help="0-based task indices. Omit for all tasks.")
    return parser.parse_args()

def main():
    args = parse_args()
    model_keys  = list(MODEL_IDS.keys()) if "all" in args.models else args.models
    shot_modes  = args.shots
    task_subset = TASKS if args.tasks is None else [TASKS[i] for i in args.tasks]

    import torch

    print("Robot Task Decomposition — LLM Evaluation")
    print(f"  Models : {model_keys}")
    print(f"  Shots  : {shot_modes}")
    print(f"  Tasks  : {len(task_subset)}")
    print(f"  Device : {'cuda' if torch.cuda.is_available() else 'cpu'}")

    results = run_evaluation(model_keys, shot_modes, args.output, task_subset)
    aggregate_and_report(results, args.output)
    print("\nDone.")

if __name__ == "__main__":
    main()

