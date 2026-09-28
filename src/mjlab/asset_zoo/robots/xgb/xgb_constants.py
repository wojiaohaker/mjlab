"""XGB quadruped constants.

The XGB is a 12-DOF quadruped robot with 3 joints per leg (ABAD, HIP, KNEE).
Joint naming: F{AR|BL|RR|RL}_{ABAD|HIP|KNEE}_JOINT where AR=FR, BL=FL, AR=RR, BL=RL.
PD gains match qiyuan_mc deployment exactly: kp=20, kd=0.7, torque limit=28 Nm.
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

XGB_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "xgb" / "xmls" / "xgb.xml"
)
assert XGB_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(XGB_XML))


##
# Actuator config.
##

# PD gains match qiyuan_mc deployment: kp=20, kd=0.7.
# Torque limit from MJCF actuatorfrcrange: 28 Nm.
# Armature 0.0 matches xgbrl IsaacLab IdealPDActuatorCfg (no rotor inertia data).
XGB_ABAD_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ABAD_JOINT",),
  stiffness=20.0,
  damping=0.7,
  effort_limit=28.0,
  armature=0.0,
)
XGB_HIP_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_HIP_JOINT",),
  stiffness=20.0,
  damping=0.7,
  effort_limit=28.0,
  armature=0.0,
)
XGB_KNEE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_KNEE_JOINT",),
  stiffness=20.0,
  damping=0.7,
  effort_limit=28.0,
  armature=0.0,
)

##
# Keyframes.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.32),  # Matrix standing body height.
  joint_pos={
    ".*_ABAD_JOINT": 0.0,   # Matrix: abad_stand_pos = 0.
    ".*_HIP_JOINT": 0.8,    # Matrix: hip_stand_pos = 0.8.
    ".*_KNEE_JOINT": -1.5,  # Matrix: knee_stand_pos = -1.5.
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = r"^[FR][LR]_foot_collision$"

# Foot collisions: condim 6 (full friction), high friction.
# Other collisions: condim 1 (frictionless), to avoid sticky legs.
_collision_regex = r".*_collision\d*"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(_collision_regex,),
  contype=1,
  conaffinity=1,
  # Harden all collision geoms.
  solref=(0.01, 1),
  # Configure feet colliders. Other colliders are frictionless (condim=1).
  condim={_foot_regex: 6, _collision_regex: 1},
  priority={_foot_regex: 1, ".*": 0},
  friction={_foot_regex: (1, 5e-3, 5e-4)},
)

##
# Final config.
##

XGB_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    XGB_ABAD_ACTUATOR_CFG,
    XGB_HIP_ACTUATOR_CFG,
    XGB_KNEE_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_xgb_robot_cfg() -> EntityCfg:
  """Get a fresh XGB robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=XGB_ARTICULATION,
  )


# Action scale matches qiyuan_mc deployment: qdes = q_default + 0.25 * action.
XGB_ACTION_SCALE: dict[str, float] = {}
for a in XGB_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  for n in a.target_names_expr:
    XGB_ACTION_SCALE[n] = 0.25


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_xgb_robot_cfg())
  viewer.launch(robot.spec.compile())
