"""Train an odometry network for XGB.

The odom network estimates body-frame base linear velocity (3-dim) from
proprioceptive sensors (29-dim), matching qiyuan_mc's OdomEstimator input:

  [0:2]   roll, pitch          (from root quaternion, unscaled)
  [2:14]  joint_pos_delta      (q - default_pos, unscaled)
  [14:26] joint_vel * 0.05      (scaled, matching RL_JVEL_SCALE)
  [26:29] body_ang_vel * 0.25   (scaled, matching RL_GYRO_SCALE)

Output: root_link_lin_vel_b (3-dim, raw unscaled).

The network is an MLP (256, 128) trained with MSE loss on rollouts from the
already-trained walk policy. Data is collected in-env using ground-truth
velocity as the target. Exported ONNX uses input name "input" and output name
"output" (auto-detected by mc_ctrl as an MLP model).

Usage:
  uv run python src/mjlab/tasks/velocity/config/xgb/train_odom.py \
      --task Mjlab-Velocity-Flat-Xgb \
      --checkpoint logs/rsl_rl/xgb_velocity/<run_dir>/model_3000.pt \
      --num-envs 2048 --collect-steps 2000 --epochs 50
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Train XGB odom network")
  p.add_argument("--task", default="Mjlab-Velocity-Flat-Xgb")
  p.add_argument("--checkpoint", required=True, help="Path to trained walk policy .pt")
  p.add_argument("--num-envs", type=int, default=2048)
  p.add_argument("--collect-steps", type=int, default=2000,
                help="Number of env steps to collect data from")
  p.add_argument("--epochs", type=int, default=50)
  p.add_argument("--batch-size", type=int, default=4096)
  p.add_argument("--lr", type=float, default=1e-3)
  p.add_argument("--device", default=None)
  p.add_argument("--output-dir", default="logs/odom/xgb",
                help="Directory to save odom ONNX and checkpoints")
  p.add_argument("--no-randomization", action="store_true",
                help="Disable domain randomization for clean data collection")
  return p.parse_args()


def quaternion_to_roll_pitch(qw, qx, qy, qz):
  """Compute (roll, pitch) from quaternion (w, x, y, z)."""
  roll = torch.atan2(2.0 * (qw * qx + qy * qz),
                     1.0 - 2.0 * (qx * qx + qy * qy))
  pitch = torch.asin(torch.clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0))
  return roll, pitch


def build_odom_input(asset_data, num_envs):
  """Build the 29-dim odom input from asset data.

  Matches mc_ctrl OdomEstimator::buildOdomInput:
    [0:2]   roll, pitch
    [2:14]  joint_pos_delta (q - default_pos)
    [14:26] joint_vel * 0.05
    [26:29] body_ang_vel * 0.25
  """
  quat = asset_data.root_link_quat_w  # (N, 4) w,x,y,z
  qw, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
  roll, pitch = quaternion_to_roll_pitch(qw, qx, qy, qz)

  jpos = asset_data.joint_pos  # (N, 12)
  jpos_default = asset_data.default_joint_pos  # (N, 12) or (12,)
  if jpos_default.dim() == 1:
    jpos_default = jpos_default.unsqueeze(0).expand_as(jpos)
  jpos_delta = jpos - jpos_default

  jvel = asset_data.joint_vel * 0.05  # scaled
  gyro = asset_data.root_link_ang_vel_b * 0.25  # scaled

  odom_in = torch.cat([
    roll.unsqueeze(1),       # [0]
    pitch.unsqueeze(1),      # [1]
    jpos_delta,              # [2:14]
    jvel,                    # [14:26]
    gyro,                    # [26:29]
  ], dim=1)
  return odom_in  # (N, 29)


def collect_data(args):
  """Run walk policy in env, collect (odom_input, GT_base_vel) pairs."""
  import mjlab.tasks  # noqa: F401 - register tasks
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = args.num_envs
  agent_cfg = load_rl_cfg(args.task)

  if args.no_randomization:
    # Disable push and terrain randomization for clean data.
    env_cfg.events.pop("push_robot", None)
    env_cfg.events.pop("randomize_terrain", None)
    env_cfg.observations["actor"].enable_corruption = False

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True,
              map_location=device)
  policy = runner.get_inference_policy(device=device)

  obs, _ = env.reset()
  all_inputs = []
  all_targets = []
  n_dropped = 0

  asset = env.unwrapped.scene["robot"]
  print(f"[collect] num_envs={args.num_envs} steps={args.collect_steps}")

  for step in range(args.collect_steps):
    with torch.no_grad():
      action = policy(obs)
    result = env.step(action)
    # mjlab returns (next_obs, reward, terminated, info) — unpack defensively.
    if len(result) == 5:
      obs, reward, terminated, truncated, info = result
    else:
      obs, reward, terminated, info = result

    odom_in = build_odom_input(asset.data, args.num_envs)
    gt_vel = asset.data.root_link_lin_vel_b.clone()

    # Drop NaN/Inf samples (rough terrain can produce unstable envs) and
    # clip physically implausible velocities to avoid poisoning the regressor.
    finite = torch.isfinite(odom_in).all(dim=1) & torch.isfinite(gt_vel).all(dim=1)
    plausible = (gt_vel.abs() < 10.0).all(dim=1)
    valid = finite & plausible
    n_dropped += int((~valid).sum().item())
    if bool(valid.any()):
      all_inputs.append(odom_in[valid].cpu())
      all_targets.append(gt_vel[valid].cpu())

    if (step + 1) % 200 == 0:
      print(f"  step {step+1}/{args.collect_steps} "
            f"vel_mean={gt_vel[finite].mean(0).tolist()} dropped={n_dropped}")

  env.close()

  inputs = torch.cat(all_inputs, dim=0)   # (N, 29)
  targets = torch.cat(all_targets, dim=0)  # (N, 3)
  print(f"[collect] total samples: {inputs.shape[0]} (dropped {n_dropped})")
  return inputs, targets


class OdomMLP(nn.Module):
  """MLP odom estimator: 29-dim input → 3-dim base_lin_vel_b."""

  def __init__(self, input_dim=29, hidden=(256, 128), output_dim=3):
    super().__init__()
    layers = []
    d = input_dim
    for h in hidden:
      layers += [nn.Linear(d, h), nn.ReLU()]
      d = h
    layers.append(nn.Linear(d, output_dim))
    self.net = nn.Sequential(*layers)

  def forward(self, x):
    return self.net(x)


def train_network(inputs, targets, args, device):
  """Train the MLP odom network with MSE loss."""
  model = OdomMLP(input_dim=inputs.shape[1]).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  criterion = nn.MSELoss()

  N = inputs.shape[0]
  inputs = inputs.to(device)
  targets = targets.to(device)

  best_loss = float("inf")
  best_state = {k: v.clone() for k, v in model.state_dict().items()}
  for epoch in range(args.epochs):
    perm = torch.randperm(N, device=device)
    epoch_loss = 0.0
    n_batches = 0
    for i in range(0, N, args.batch_size):
      idx = perm[i:i + args.batch_size]
      x, y = inputs[idx], targets[idx]
      pred = model(x)
      loss = criterion(pred, y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item()
      n_batches += 1

    avg_loss = epoch_loss / max(n_batches, 1)
    if avg_loss < best_loss:
      best_loss = avg_loss
      best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if (epoch + 1) % 5 == 0 or epoch == 0:
      with torch.no_grad():
        pred = model(inputs)
        rmse = torch.sqrt(((pred - targets) ** 2).mean(0))
      print(f"  epoch {epoch+1}/{args.epochs} loss={avg_loss:.6f} "
            f"rmse_xyz=({rmse[0]:.4f},{rmse[1]:.4f},{rmse[2]:.4f})")

  model.load_state_dict(best_state)
  model.eval()
  model.to("cpu")
  return model


def export_onnx(model, output_path, input_dim=29):
  """Export the odom MLP as ONNX with input name "input", output name "output"."""
  model.eval()
  dummy = torch.randn(1, input_dim)
  torch.onnx.export(
    model,
    (dummy,),
    output_path,
    export_params=True,
    opset_version=18,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={},
    dynamo=False,
  )
  print(f"[export] ONNX saved to {output_path}")


def main():
  args = parse_args()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  print(f"[odom_train] device={device}")
  print(f"[odom_train] checkpoint={args.checkpoint}")

  inputs, targets = collect_data(args)

  model = train_network(inputs, targets, args, device)

  output_dir = Path(args.output_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  output_dir.mkdir(parents=True, exist_ok=True)

  onnx_path = output_dir / "odom_mix_walk.onnx"
  export_onnx(model, str(onnx_path))

  # Also save a torch checkpoint.
  pt_path = output_dir / "odom_model.pt"
  torch.save({
    "model_state_dict": model.state_dict(),
    "input_dim": 29,
    "output_dim": 3,
    "hidden": (256, 128),
  }, str(pt_path))
  print(f"[odom_train] checkpoint saved to {pt_path}")
  print(f"[odom_train] ONNX saved to {onnx_path}")
  print(f"[odom_train] Done. Copy onnx to qiyuan_mc/models_xgbrl/odom_mix_walk.onnx")


if __name__ == "__main__":
  main()
