## mjlab 全流程命令

### 1. 列出所有可用任务（确认注册）
```bash
cd /home/qiyuan/Softwares/mjlab
uv run python -c "import mjlab.tasks; from mjlab.tasks.registry import list_tasks; [print(t) for t in list_tasks()]"
```

### 2. 训练（flat 地形）
```bash
cd /home/qiyuan/Softwares/mjlab
WANDB_MODE=disabled uv run train Mjlab-Velocity-Flat-Xgb \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 3000
```
- `WANDB_MODE=disabled` 关闭 wandb 日志（要联网记录就删掉这行）
- `--env.scene.num-envs 4096` 并行环境数（GPU 显存不够就调小：1024/2048）
- `--agent.max-iterations 3000` 训练轮数
- 训练完成自动导出 ONNX 到 `logs/rsl_rl/xgb_velocity/<时间戳>/<时间戳>.onnx`

### 3. rough 地形训练（进阶，走楼梯/崎岖地面）
```bash
cd /home/qiyuan/Softwares/mjlab
WANDB_MODE=disabled uv run train Mjlab-Velocity-Rough-Xgb \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 5000
```

### 4. 查看训练曲线（TensorBoard）
```bash
cd /home/qiyuan/Softwares/mjlab
uv run --with 'setuptools<80' tensorboard --logdir logs/rsl_rl/xgb_velocity --port 6006
# 浏览器打开 http://localhost:6006
```

### 5. 可视化机器人行走（play，加载 checkpoint）
```bash
cd /home/qiyuan/Softwares/mjlab
uv run play Mjlab-Velocity-Flat-Xgb \
  --checkpoint_file logs/rsl_rl/xgb_velocity/2026-09-28_17-58-53/model_2999.pt
```
- 弹出 MuJoCo viewer 窗口，机器人按随机速度命令行走
- 可换任意 checkpoint：`model_500.pt`、`model_1500.pt` 等对比学习过程

### 6. 单独可视化选项
```bash
# 随机动作（不加载模型，看机器人被动倒下）
uv run play Mjlab-Velocity-Flat-Xgb --agent random
# 零动作
uv run play Mjlab-Velocity-Flat-Xgb --agent zero
# 录制视频
uv run play Mjlab-Velocity-Flat-Xgb \
  --checkpoint_file logs/rsl_rl/xgb_velocity/2026-09-28_17-58-53/model_2999.pt \
  --video --video_length 400
```

### 7. 关键产物路径
```
logs/rsl_rl/xgb_velocity/2026-09-28_17-58-53/
├── 2026-09-28_17-58-53.onnx   ← 部署用 ONNX 模型
├── model_2999.pt               ← 最终 checkpoint（play 用）
├── model_0.pt ~ model_2999.pt  ← 每 50 iter 存一个
├── events.out.tfevents.*       ← TensorBoard 日志
├── params/{agent.yaml,env.yaml}← 训练配置快照
└── git/mjlab.diff              ← 代码改动记录
```

### 8. 检查 ONNX 模型输入输出维度
```bash
cd /home/qiyuan/Softwares/mjlab
uv run python -c "
import onnx
m = onnx.load('logs/rsl_rl/xgb_velocity/2026-09-28_17-58-53/2026-09-28_17-58-53.onnx')
for i in m.graph.input:
    print('IN:', i.name, [d.dim_value for d in i.type.tensor_type.shape.dim])
for o in m.graph.output:
    print('OUT:', o.name, [d.dim_value for d in o.type.tensor_type.shape.dim])
"
```

---

**⚠️ 部署提醒**：直接把这个 ONNX 部署到 qiyuan_mc 会有问题——mjlab 默认观测格式（base_lin_vel, projected_gravity, joint_pos, joint_vel, command, ...）和 qiyuan_mc 期望的 48 维格式（`2×vel, gyro×0.25, roll_pitch, 2×vel_cmd, jpos_delta, qd×0.05, last_action`）不一样。部署前需要像 xgbrl 那样在 [env_cfgs.py](file:///home/qiyuan/Softwares/mjlab/src/mjlab/tasks/velocity/config/xgb/env_cfgs.py) 里自定义 actor 观测组来匹配 qiyuan_mc 的部署格式。需要我帮你改观测格式吗？