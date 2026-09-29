"""XGB flipover recovery environment configuration.

The robot starts upside down (pitch ≈ π) and must recover to an upright
standing posture. Observations use the same 48-dim deployment format as
the walk task (see xgb_observations.py), so the trained policy exports
directly to qiyuan_mc.

Reward design:
  * upright:        gravity z close to 1 (robot standing)        weight +5.0
  * base_lin_vel:   penalize residual velocity after recovery    weight -0.5
  * base_ang_vel:   penalize residual angular velocity           weight -0.5
  * action_rate_l2: smooth actions                                 weight -0.1
  * dof_pos_limits:  stay within joint limits                      weight -1.0
"""

import math

import torch

from mjlab.asset_zoo.robots import get_xgb_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def _base_lin_vel_l2(env, asset_cfg=None):
  """Penalize base linear velocity (L2 squared) in body frame."""
  if asset_cfg is None:
    asset_cfg = SceneEntityCfg("robot")
  asset = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_lin_vel_b), dim=1)

from .xgb_observations import (
  base_ang_vel_025,
  base_lin_vel_2x,
  joint_vel_005,
  roll_pitch_zero,
  velocity_commands_2x,
)

# Flipover recovery: robot starts upside down (pitch ≈ π).
_FLIPOVER_PITCH = (math.pi - 0.3, math.pi + 0.3)
_FLIPOVER_Z = (0.28, 0.35)


def xgb_flipover_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create XGB flipover recovery environment configuration."""
  cfg = make_velocity_env_cfg()

  # Flat terrain (no rough terrain needed for recovery).
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.num_envs = 4096

  cfg.scene.entities = {"robot": get_xgb_robot_cfg()}

  # Remove terrain-specific sensors (flat ground).
  remove_sensors = {
    "terrain_scan",
    "foot_height_scan",
    "self_collision",
    "thigh_ground_touch",
    "shank_ground_touch",
    "trunk_ground_touch",
  }
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name not in remove_sensors
  )

  # Joint position action with XGB action scale.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  from mjlab.asset_zoo.robots import XGB_ACTION_SCALE
  joint_pos_action.scale = XGB_ACTION_SCALE

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  # Episode length: 10 seconds for recovery.
  cfg.episode_length_s = 10

  # Override reset events: robot starts upside down.
  cfg.events["reset_base"] = EventTermCfg(
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": _FLIPOVER_Z,
        "roll": (-0.3, 0.3),
        "pitch": _FLIPOVER_PITCH,
        "yaw": (-3.14, 3.14),
      },
      "velocity_range": {},
    },
  )

  # Keep joint reset but allow some variation in starting pose.
  cfg.events["reset_robot_joints"] = EventTermCfg(
    func=mdp.reset_joints_by_offset,
    mode="reset",
    params={
      "position_range": (0.0, 0.0),
      "velocity_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    },
  )

  # Disable push robot during recovery training.
  cfg.events.pop("push_robot", None)

  # Zero velocity command (observation format expects it, but no tracking).
  cfg.commands["twist"] = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(10.0, 10.0),
    rel_standing_envs=1.0,  # All standing (zero command).
    rel_heading_envs=0.0,
    rel_forward_envs=0.0,
    heading_command=False,
    debug_vis=False,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(0.0, 0.0),
      lin_vel_y=(0.0, 0.0),
      ang_vel_z=(0.0, 0.0),
    ),
  )

  # Replace rewards with flipover-specific ones.
  cfg.rewards = {
    "upright": RewardTermCfg(
      func=mdp.upright,
      weight=5.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      },
    ),
    "base_lin_vel": RewardTermCfg(
      func=_base_lin_vel_l2,
      weight=-0.5,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("base_link",))},
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.5,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("base_link",))},
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
  }

  # Terminations: time_out only (no fell_over since robot starts fallen).
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
  }

  # No curriculum for flipover.
  cfg.curriculum = {}

  # Remove terrain-specific events.
  for key in ("foot_friction", "foot_friction_slide", "foot_friction_spin",
              "foot_friction_roll", "randomize_terrain"):
    cfg.events.pop(key, None)
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  # Use the same 48-dim deployment observation format as walk.
  cfg.observations["actor"].terms = {
    "base_lin_vel": ObservationTermCfg(func=base_lin_vel_2x),
    "base_ang_vel": ObservationTermCfg(func=base_ang_vel_025),
    "roll_pitch": ObservationTermCfg(func=roll_pitch_zero),
    "vel_cmd": ObservationTermCfg(func=velocity_commands_2x),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel, params={"biased": True}
    ),
    "joint_vel": ObservationTermCfg(func=joint_vel_005),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  # Critic uses the same terms as actor (no privileged sensors on flat ground).
  cfg.observations["critic"].terms = {
    "base_lin_vel": ObservationTermCfg(func=base_lin_vel_2x),
    "base_ang_vel": ObservationTermCfg(func=base_ang_vel_025),
    "roll_pitch": ObservationTermCfg(func=roll_pitch_zero),
    "vel_cmd": ObservationTermCfg(func=velocity_commands_2x),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=joint_vel_005),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False

  return cfg
