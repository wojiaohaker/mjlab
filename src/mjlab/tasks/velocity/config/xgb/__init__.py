from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  xgb_flat_env_cfg,
  xgb_rough_env_cfg,
)
from .rl_cfg import xgb_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Xgb",
  env_cfg=xgb_rough_env_cfg(),
  play_env_cfg=xgb_rough_env_cfg(play=True),
  rl_cfg=xgb_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Xgb",
  env_cfg=xgb_flat_env_cfg(),
  play_env_cfg=xgb_flat_env_cfg(play=True),
  rl_cfg=xgb_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
