#!/usr/bin/env python3
"""
I use this executor as the bridge between the LLM output and the robot.
It reads the JSON plan, cleans up small LLM mistakes, and dispatches ROS 2 actions.

Supported actions (matching the LLM closed set):
  move_pose     → /planner/move_pose  (highlevel_interfaces/action/PoseCommand)
  open_gripper  → /gripper_controller/gripper_cmd (control_msgs/action/ParallelGripperCommand)
  close_gripper → /gripper_controller/gripper_cmd

Usage:
  python3 ros_executor.py --task-json /path/to/task.json

I calibrated the object positions in the same frame used by /gen3/feedback/pose.
For this project that pose comes from the bracelet_link FK output in the
kinematic controller.
"""

import argparse
import json
import re

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from control_msgs.action import ParallelGripperCommand
from highlevel_interfaces.action import PoseCommand


# ---------------------------------------------------------------------------
# I keep the scene geometry explicit here because the final task uses a fixed
# cup/cube setup. This avoids adding camera perception when the assignment only
# needs symbolic task execution.
# ---------------------------------------------------------------------------
OBJECT_POSITIONS = {
    # I treat "cube" and "block" as the same object because the LLM sometimes
    # uses either word for the required wooden cube.
    "cube":  (0.450, -0.007, 0.492),
    "block": (0.450, -0.007, 0.492),  # same object, different LLM word
    "cup":   (0.345,  0.170, 0.450),
}

# I add a little z clearance when placing so the object does not collide with
# the cup rim or table before I open the gripper.
PLACE_Z_LIFT = 0.08
SIDE_PLACE_Z_LIFT = 0.02

# These offsets turn symbolic relations from the LLM into actual Cartesian
# targets. Positive y is the "left of cup" direction in my calibrated setup.
SPATIAL_OFFSETS = {
    "in":      ( 0.00,  0.00,  0.05),
    "inside":  ( 0.00,  0.00,  0.05),
    "into":    ( 0.00,  0.00,  0.05),
    "on":      ( 0.00,  0.00,  0.10),
    "above":   ( 0.00,  0.00,  0.15),
    "left":    ( 0.00,  0.15,  0.00),
    "left_of": ( 0.00,  0.15,  0.10),
    "right":   ( 0.00, -0.15,  0.00),
    "right_of": (0.00, -0.15,  0.10),
    "behind":  (-0.15,  0.00,  0.00),
    "front":   ( 0.15,  0.00,  0.00),
    "in_front_of": (0.15, 0.00, 0.00),
    "next_to": ( 0.00,  0.15,  0.00),
}

OBJECT_ALIASES = {
    "box": "cube",
    "wood_cube": "cube",
    "wooden_cube": "cube",
    "it": "cube",
}

# The one-shot prompt used to mention colors, so I strip color prefixes instead
# of treating red_cube/blue_cube as separate physical objects.
COLOR_PREFIXES = ("red_", "blue_", "green_", "yellow_")

# Small LLMs sometimes put the relation inside object_b, like left_of_cup.
# I split that back into object_b=cup and spatial_relation=left_of.
RELATION_PREFIXES = {
    "left_of_": "left_of",
    "right_of_": "right_of",
    "in_front_of_": "in_front_of",
    "inside_": "in",
    "into_": "in",
    "in_": "in",
    "above_": "above",
    "on_": "on",
    "behind_": "behind",
}

# These values are tuned for the simulated cube. I stop at a partial close,
# because commanding 1.0 can squeeze the cube out before the plan continues.
GRIPPER_OPEN   = 0.0
GRIPPER_CLOSED = 0.65
GRIPPER_JOINT = "right_finger_bottom_joint"
GRIPPER_VELOCITY = 0.2
GRIPPER_EFFORT = 4.0
GRIPPER_WAIT_TIMEOUT = 3.0

# I always approach from above first. Direct horizontal moves near the table can
# bump the cube or scrape the cup.
APPROACH_HEIGHT = 0.15

# I fail fast if a server is missing so I know the launch setup is wrong.
SERVER_TIMEOUT = 15.0


class Executor(Node):
    def __init__(self):
        super().__init__("ros_executor")
        # The executor only acts as a client. The actual motion is handled by my
        # potential-field pose server and the gripper controller.
        self._pose_client    = ActionClient(self, PoseCommand,    "/planner/move_pose")
        self._gripper_client = ActionClient(
            self,
            ParallelGripperCommand,
            "/gripper_controller/gripper_cmd",
        )
        self._last_pick_position = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wait_for_pose_server(self) -> bool:
        self.get_logger().info("Waiting for /planner/move_pose…")
        if not self._pose_client.wait_for_server(timeout_sec=SERVER_TIMEOUT):
            self.get_logger().error("/planner/move_pose not available after "
                                    f"{SERVER_TIMEOUT}s — is the simulation running?")
            return False
        return True

    def _wait_for_gripper_server(self) -> bool:
        self.get_logger().info("Waiting for /gripper_controller/gripper_cmd…")
        if not self._gripper_client.wait_for_server(timeout_sec=SERVER_TIMEOUT):
            self.get_logger().error("/gripper_controller/gripper_cmd not available after "
                                    f"{SERVER_TIMEOUT}s — is the gripper controller running?")
            return False
        return True

    @staticmethod
    def _normalise_symbol(value):
        """I normalize LLM text into the compact symbols used in my pose table."""
        symbol = (value or "").strip().lower().replace(" ", "_")
        symbol = re.sub(r"^(the_|a_|an_)", "", symbol)
        for prefix in COLOR_PREFIXES:
            if symbol.startswith(prefix):
                symbol = symbol[len(prefix):]
                break
        return OBJECT_ALIASES.get(symbol, symbol)

    def _split_relation_object(self, obj_b, relation):
        """I recover relation/object pairs when the LLM merges them together."""
        for prefix, parsed_relation in RELATION_PREFIXES.items():
            if obj_b.startswith(prefix):
                parsed_object = self._normalise_symbol(obj_b[len(prefix):])
                return parsed_object, relation or parsed_relation
        return obj_b, relation

    @staticmethod
    def _step_index(step_key):
        """I sort step_1, step_2, ... numerically instead of alphabetically."""
        match = re.search(r"\d+", step_key)
        return int(match.group()) if match else 0

    def _resolve_position(self, step: dict):
        """I convert one symbolic move_pose step into a Cartesian target."""
        obj_a    = self._normalise_symbol(step.get("object_a"))
        obj_b    = self._normalise_symbol(step.get("object_b"))
        relation = self._normalise_symbol(step.get("spatial_relation"))
        obj_b, relation = self._split_relation_object(obj_b, relation)

        if obj_b and relation:
            # For placement, I start from the reference object's pose and apply
            # the requested spatial relation as an offset.
            if obj_b not in OBJECT_POSITIONS:
                self.get_logger().warn(f"Unknown object_b '{obj_b}' — add it to OBJECT_POSITIONS")
                return None
            bx, by, bz = OBJECT_POSITIONS[obj_b]
            dx, dy, dz = SPATIAL_OFFSETS.get(relation, (0.0, 0.0, 0.0))
            z_lift = SIDE_PLACE_Z_LIFT if relation in {"left", "left_of", "right", "right_of", "next_to"} else PLACE_Z_LIFT
            return (bx + dx, by + dy, bz + dz + z_lift)

        pick_object = obj_a or obj_b
        if pick_object:
            # For picking, I just go to the object itself.
            if pick_object not in OBJECT_POSITIONS:
                self.get_logger().warn(f"Unknown object '{pick_object}' — add it to OBJECT_POSITIONS")
                return None
            return OBJECT_POSITIONS[pick_object]

        self.get_logger().warn("move_pose step has neither object_a nor object_b — skipping")
        return None

    # ------------------------------------------------------------------
    # Action dispatchers
    # ------------------------------------------------------------------

    def _move_pose(self, x: float, y: float, z: float) -> bool:
        """I send one Cartesian goal to /planner/move_pose and wait for the result."""
        if not self._wait_for_pose_server():
            return False

        goal = PoseCommand.Goal()
        goal.x, goal.y, goal.z = x, y, z
        goal.roll = goal.pitch = goal.yaw = 0.0

        self.get_logger().info(f"  move_pose → ({x:.3f}, {y:.3f}, {z:.3f})")
        send_future = self._pose_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self,send_future)

        handle = send_future.result()
        if not handle.accepted:
            self.get_logger().error("  move_pose goal REJECTED")
            return False

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self,result_future)
        res = result_future.result().result
        self.get_logger().info(f"  move_pose result: {res.message}")
        return res.success

    def _gripper(self, position: float) -> bool:
        """I send one gripper command and continue after a short grasp wait."""
        if not self._wait_for_gripper_server():
            return False

        label = "open_gripper" if position < 0.5 else "close_gripper"
        goal = ParallelGripperCommand.Goal()
        goal.command.name = [GRIPPER_JOINT]
        goal.command.position = [position]
        goal.command.velocity = [GRIPPER_VELOCITY]
        goal.command.effort = [GRIPPER_EFFORT]

        self.get_logger().info(f"  {label} → {GRIPPER_JOINT}={position}")
        send_future = self._gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self,send_future)

        handle = send_future.result()
        if not handle.accepted:
            self.get_logger().error(f"  {label} goal REJECTED")
            return False

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=GRIPPER_WAIT_TIMEOUT)
        if not result_future.done():
            # During a real grasp the fingers may stop on the cube before the
            # controller reports "goal reached", so I keep the plan moving.
            self.get_logger().info(
                f"  {label} still active after {GRIPPER_WAIT_TIMEOUT:.1f}s; continuing with plan"
            )
            return True

        result = result_future.result().result
        self.get_logger().info(
            f"  {label} reached_goal={result.reached_goal} stalled={result.stalled}"
        )
        return True

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def execute_plan(self, plan: dict) -> bool:
        """I execute the cleaned JSON steps in order."""
        ordered_steps = self._normalise_plan(plan)
        for index, step in enumerate(ordered_steps, start=1):
            step_key = f"step_{index}"
            action = (step.get("action") or "").strip().lower()
            self.get_logger().info(f"[{step_key}] action={action}")

            if action == "move_pose":
                pos = self._resolve_position(step)
                if pos is None:
                    continue

                obj_b    = (step.get("object_b")        or "").strip()
                relation = (step.get("spatial_relation") or "").strip()
                is_pick  = not (obj_b and relation)

                if is_pick:
                    # I approach from above, lower to the grasp pose, then let
                    # close_gripper lift the cube after contact.
                    x, y, z = pos
                    self._last_pick_position = pos
                    self.get_logger().info(f"  approach phase → z={z + APPROACH_HEIGHT:.3f}")
                    self._move_pose(x, y, z + APPROACH_HEIGHT)
                    self.get_logger().info("  lower phase")
                else:
                    # For placement, I also go above the target before lowering
                    # so the carried cube does not sweep into the cup.
                    x, y, z = pos
                    self.get_logger().info(f"  placement approach phase → z={z + APPROACH_HEIGHT:.3f}")
                    self._move_pose(x, y, z + APPROACH_HEIGHT)
                    self.get_logger().info("  placement lower phase")

                ok = self._move_pose(*pos)
                if not ok:
                    self.get_logger().warn(f"  {step_key} did not fully succeed — continuing")

            elif action == "open_gripper":
                self._gripper(GRIPPER_OPEN)

            elif action == "close_gripper":
                if self._gripper(GRIPPER_CLOSED) and self._last_pick_position is not None:
                    # After grasping, I lift before any later horizontal move.
                    x, y, z = self._last_pick_position
                    self.get_logger().info("  lift phase after grasp")
                    self._move_pose(x, y, z + APPROACH_HEIGHT)

            else:
                self.get_logger().warn(f"  Unknown action '{action}' — skipping")

        self.get_logger().info("Plan execution complete.")
        return True

    def _normalise_plan(self, plan: dict):
        """I repair common valid-JSON-but-wrong-plan mistakes before moving."""
        steps = [plan[key] for key in sorted(plan.keys(), key=self._step_index)]
        normalised = []
        deferred_moves = []
        has_closed = False
        picked_object = None

        for step in steps:
            step = dict(step)
            action = (step.get("action") or "").strip().lower()
            obj_a = self._normalise_symbol(step.get("object_a"))
            obj_b = self._normalise_symbol(step.get("object_b"))
            relation = self._normalise_symbol(step.get("spatial_relation"))

            fixed_obj_b, fixed_relation = self._split_relation_object(obj_b, relation)
            if fixed_obj_b != obj_b or fixed_relation != relation:
                self.get_logger().info(
                    f"  normalised target '{obj_b}'/'{relation}' -> '{fixed_obj_b}'/'{fixed_relation}'"
                )
                step["object_b"] = fixed_obj_b
                step["spatial_relation"] = fixed_relation
                obj_b = fixed_obj_b
                relation = fixed_relation

            if has_closed and action == "move_pose" and obj_a == picked_object and not obj_b and not relation:
                # For "pick the cube", SmolLM2 sometimes adds a second move
                # back to the cube after grasping. I drop it so the cube stays lifted.
                self.get_logger().info("  dropping duplicate pick move after close_gripper")
                continue

            if has_closed and action == "open_gripper" and not obj_b and not relation:
                # A pick-only task should end while holding the cube, so an
                # open_gripper with no placement target is an LLM over-generation.
                self.get_logger().info("  dropping open_gripper because no placement target was provided")
                continue

            is_place_move = action == "move_pose" and step.get("object_a") and step.get("object_b")
            if is_place_move and not has_closed:
                # If the LLM tries to place before closing the gripper, I delay
                # that move until right after the grasp.
                self.get_logger().info("  deferring placement move until after close_gripper")
                deferred_moves.append(step)
                continue

            normalised.append(step)

            if action == "close_gripper":
                has_closed = True
                picked_object = obj_a
                normalised.extend(deferred_moves)
                deferred_moves = []

        normalised.extend(deferred_moves)
        return normalised


def main():
    parser = argparse.ArgumentParser(description="Execute an LLM JSON plan via ROS 2 actions")
    parser.add_argument("--task-json", required=True, help="Path to the LLM-generated JSON plan")
    args = parser.parse_args()

    with open(args.task_json) as f:
        plan = json.load(f)

    rclpy.init()
    node = Executor()
    try:
        node.execute_plan(plan)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
