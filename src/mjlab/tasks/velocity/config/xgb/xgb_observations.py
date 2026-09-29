"""Custom observation functions for XGB, matching qiyuan_mc deployment format.

The deployment (qiyuan_mc) uses a 48-dim observation vector with specific
scaling factors. These functions produce observations that match the deployment
format exactly, so a policy trained in mjlab can be exported as ONNX and used
directly in qiyuan_mc.

Deployment obs layout (48-dim), see qiyuan_mc src/main_controller.cpp makeObs():
    [0:3]   2*base_vel       (from odom network at deployment, GT during training)
    [3:6]   gyro*0.25        (body angular velocity * 0.25)
    [6:9]   (roll, pitch, 0) (Euler angles from root quaternion)
    [9:12]  2*vel_cmd        (velocity command * 2)
    [12:24] jpos_delta       (joint_pos - default_pos, no scaling)
    [24:36] qd*0.05          (joint velocity * 0.05)
    [36:48] last_action      (previous action)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def base_lin_vel_2x(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Body-frame linear velocity multiplied by 2.

  Matches deployment obs[0:3] = 2*base_vel. During training this uses
  ground-truth velocity; at deployment the odom network provides the estimate.
  """
  asset = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_b * 2.0


def base_ang_vel_025(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Body-frame angular velocity multiplied by 0.25.

  Matches deployment obs[3:6] = gyro*0.25.
  """
  asset = env.scene[asset_cfg.name]
  return asset.data.root_link_ang_vel_b * 0.25


def roll_pitch_zero(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Euler angles (roll, pitch, 0) from root quaternion.

  Matches deployment obs[6:9] = (roll, pitch, 0).
  """
  asset = env.scene[asset_cfg.name]
  quat = asset.data.root_link_quat_w  # (N, 4) in (w, x, y, z) order.
  qw = quat[..., 0]
  qx = quat[..., 1]
  qy = quat[..., 2]
  qz = quat[..., 3]
  # roll = atan2(2(w*x + y*z), 1 - 2(x^2 + y^2))
  roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
  # pitch = asin(2(w*y - z*x))
  pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
  zeros = torch.zeros_like(roll)
  return torch.stack([roll, pitch, zeros], dim=-1)


def velocity_commands_2x(
  env: ManagerBasedRlEnv, command_name: str = "twist"
) -> torch.Tensor:
  """Velocity command multiplied by 2.

  Matches deployment obs[9:12] = 2*vel_cmd.
  """
  cmd = env.command_manager.get_command(command_name)
  assert cmd is not None
  return cmd * 2.0


def joint_vel_005(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Joint velocity multiplied by 0.05.

  Matches deployment obs[24:36] = qd*0.05.
  Note: joint_vel_rel = joint_vel - default_joint_vel, and default_joint_vel=0,
  so this is equivalent to raw joint_vel * 0.05.
  """
  asset = env.scene[asset_cfg.name]
  jnt_ids = asset_cfg.joint_ids
  return asset.data.joint_vel[:, jnt_ids] * 0.05
