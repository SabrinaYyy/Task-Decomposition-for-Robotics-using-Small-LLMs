# Applied Robotics Final Project - Option 4

This repository implements the Option 4 language-to-action pipeline:

```text
natural language instruction -> small LLM JSON plan -> ROS 2 executor -> Gen3 Gazebo actions
```

The final execution uses the base cup/cube scene, not the extra-credit colored-block tabletop scene.

## What Was Implemented

- LLM evaluation for task decomposition in `llm_project/llm/test_llms.py`
- Selected model for execution: `HuggingFaceTB/SmolLM2-1.7B-Instruct` with one-shot prompting
- Task decomposition script: `llm_project/llm/test_decomposition.py`
- JSON-to-ROS executor: `llm_project/llm/ros_executor.py`
- End-to-end runner: `llm_project/run_text_to_robot.sh`
- Option 4 launch file: `llm_project/launch/option4.launch.py`

The executor supports the required actions:

```text
move_pose
close_gripper
open_gripper
```

## Build

Place this repository in the `src/` folder of a ROS 2 workspace that also contains the course Kortex packages, including `ros2_kortex_lite` / `kortex_sim` / `kortex_description`.

Example workspace layout:

```text
<ws>/
  src/
    applied-robotics-w2026/
    ros2_kortex_lite/
    ...
```

Then build from the workspace root. `--symlink-install` is optional; a normal build also works.

```bash
cd <ws>
colcon build --packages-up-to llm_project
source install/setup.bash
```

## Python LLM Environment

The LLM dependencies should be installed in a Python virtual environment, not system Python. The virtual environment can have any name/location; the examples below use `<llm_venv>`.

```bash
python3 -m venv <llm_venv>
source <llm_venv>/bin/activate
cd <ws>/src/applied-robotics-w2026/llm_project
pip install -e .
```

If the virtual environment already exists and has the packages installed, this step can be skipped. When running the end-to-end script, pass the same environment path with `LLM_VENV=<llm_venv>`.

## How To Launch And Test

1. Open a terminal and launch the simulator/controllers:

```bash
cd <ws>
source install/setup.bash
ros2 launch llm_project option4.launch.py
```

This starts the Kortex/Gazebo simulation, the 3D kinematic controller without redundancy, the potential-field pose action server, and the gripper controller.

2. Open another terminal and check that the action servers are available:

```bash
cd <ws>
source install/setup.bash
ros2 topic echo /gen3/feedback/pose --once
ros2 action list
ros2 control list_controllers
```

Expected actions:

```text
/planner/move_pose
/gripper_controller/gripper_cmd
```

3. In the second terminal, run one of the text-to-robot tasks:

```bash
cd <ws>/src/applied-robotics-w2026/llm_project
LLM_VENV=<llm_venv> ROS_SETUP=<ws>/install/setup.bash \
./run_text_to_robot.sh "pick the cube"
```

Required tasks:

```bash
LLM_VENV=<llm_venv> ROS_SETUP=<ws>/install/setup.bash \
./run_text_to_robot.sh "pick the cube"

LLM_VENV=<llm_venv> ROS_SETUP=<ws>/install/setup.bash \
./run_text_to_robot.sh "move the block to the left of the cup"

LLM_VENV=<llm_venv> ROS_SETUP=<ws>/install/setup.bash \
./run_text_to_robot.sh "pick the cube and place it in the cup"
```

The helper script defaults to:

```text
LLM_VENV=~/llm_env
ROS_SETUP=~/ros2_ws/install/setup.bash
```

Because the TA may use a different virtual environment and workspace, the commands above explicitly set `LLM_VENV=<llm_venv>` and `ROS_SETUP=<ws>/install/setup.bash`.

## Alternative: Run The Pieces Manually

To generate a JSON plan from text:

```bash
cd <ws>/src/applied-robotics-w2026/llm_project/llm
source <llm_venv>/bin/activate
python3 test_decomposition.py --task "pick the cube" --output task.json
```

The LLM writes the latest plan to:

```text
llm_project/llm/task.json
```

To execute that JSON plan:

```bash
cd <ws>/src/applied-robotics-w2026/llm_project/llm
source <ws>/install/setup.bash
python3 ros_executor.py --task-json task.json
```

There are also fixed test plans:

```bash
python3 ros_executor.py --task-json plans/pick_cube.json
python3 ros_executor.py --task-json plans/block_left_of_cup.json
python3 ros_executor.py --task-json plans/cube_in_cup.json
```

## LLM Evaluation

The LLM results were generated with:

```bash
cd <ws>/src/applied-robotics-w2026/llm_project
source <llm_venv>/bin/activate
python llm/test_llms.py --models all --shots zero one few --tasks 0 1 2 3 4 --output results_option4/
```

Best configuration from the evaluation:

```text
SmolLM2-1.7B-Instruct, one-shot prompting
JSON validity: 100%
Action precision: 0.850
Action recall: 0.900
Action F1: 0.867
```

This is why `test_decomposition.py` uses SmolLM2 with one-shot prompting.

## Execution Notes

The executor includes small validation/normalization rules because small LLMs can output valid JSON with symbolic mistakes. For example:

- `red_cube` is normalized to `cube`
- `left_of_cup` is interpreted as `object_b="cup"` and `spatial_relation="left_of"`
- duplicate pick-only moves after grasping are dropped
- `open_gripper` without a placement target is dropped for the pick-only task

Controller choice:

```text
3D position control without redundancy
```

This was used because it was more stable than the redundant 6D controller for the required pick-and-place behaviors.

Current calibrated execution parameters:

```text
cube/block pose: (0.450, -0.007, 0.492)
cup pose:        (0.345,  0.170, 0.450)
gripper open:    0.0
gripper close:   0.65
```

## ROS Action Interfaces

Pose action:

```text
/planner/move_pose
highlevel_interfaces/action/PoseCommand
```

Gripper action:

```text
/gripper_controller/gripper_cmd
control_msgs/action/ParallelGripperCommand
```

## Reflection On The Demo

During the demo, I did not explain the system as well as I should have. Near the end of the semester I was balancing multiple final projects, assignments from three courses, and a final exam on the first day of the exam period. Because of that, I focused more on reviewing the course concepts than on carefully reviewing my own implementation before the demo.

After revisiting the project, I can see that the demo result was mainly a result of my time management and preparation, not a lack of course material. The lectures, assignments, and project instructions provided the content needed to understand and complete the pipeline. I should have spent more time making sure I could explain each part of my code clearly, especially the ROS action flow, the controller setup, and the JSON-to-action executor.

This README and the comments in the code are my attempt to make the final version easier to understand and reproduce.
