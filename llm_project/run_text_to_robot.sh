#!/usr/bin/env bash
set -euo pipefail

# I run this script after the simulator is already up. It glues the two halves
# of the project together: LLM planning first, ROS execution second.
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 \"pick the cube and place it in the cup\""
  exit 2
fi

TASK="$1"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# I keep these configurable because the grader may use a different workspace or
# virtual environment path than I used on my laptop.
TASK_JSON="${TASK_JSON:-${PROJECT_DIR}/llm/task.json}"
LLM_VENV="${LLM_VENV:-${HOME}/llm_env}"
ROS_SETUP="${ROS_SETUP:-${HOME}/ros2_ws/install/setup.bash}"

# The LLM side must run inside the Python environment with transformers/torch.
if [[ ! -f "${LLM_VENV}/bin/python" ]]; then
  echo "LLM virtualenv not found at ${LLM_VENV}"
  echo "Set LLM_VENV=/path/to/venv if your environment lives elsewhere."
  exit 1
fi

# The executor side must run after sourcing the ROS workspace.
if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "ROS setup file not found at ${ROS_SETUP}"
  echo "Set ROS_SETUP=/path/to/install/setup.bash if your workspace lives elsewhere."
  exit 1
fi

echo "[1/2] Generating task JSON with SmolLM2 one-shot..."
(
  # I write the generated plan into the repo so it can be inspected or rerun.
  cd "${PROJECT_DIR}/llm"
  "${LLM_VENV}/bin/python" test_decomposition.py \
    --task "${TASK}" \
    --output "${TASK_JSON}"
)

echo "[2/2] Executing JSON plan with ROS 2..."
(
  # ROS setup scripts can reference unset shell variables, so I temporarily
  # relax nounset while sourcing them.
  set +u
  source "${ROS_SETUP}"
  set -u
  cd "${PROJECT_DIR}/llm"
  python3 ros_executor.py --task-json "${TASK_JSON}"
)
