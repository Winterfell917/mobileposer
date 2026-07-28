# Week1 — Device Position Classification v1

> 分支建议：`test` / `exp/week1-pos-cls`  
> 目标：忽略朝向干扰，预测手表左右腕 + 手机左右袋。

## 实验决策（写死，避免后面改来改去）

| 项 | 取值 |
|----|------|
| 设备 | 1 watch + 1 phone |
| 手表位置 | 0=左腕(LW), 1=右腕(RW) |
| 手机位置 | 0=左袋(LP), 1=右袋(RP) |
| 输入 | `[watch_acc(3), watch_gyro(3), phone_acc(3), phone_gyro(3)]` → 12 维 |
| 角速度 | 由相邻帧 `ori` 旋转差分估计（仓库本身不存 gyro） |
| 窗长 / 训练步长 | 90 帧 / 15 帧（约 3s @30FPS） |
| 朝向 | **忽略**：不做 `simulations.py` 的 drift/offset 随机扰动 |
| 训练 | AMASS 合成（每条序列枚举 4 种佩戴组合） |
| 测试 | IMUPoser（必须提供真实佩戴标签） |
| 主指标 | **Joint Accuracy**（手表+手机都对） |

IMU 槽位与 MobilePoser 一致：`0=LW, 1=RW, 2=LP, 3=RP`（腕朝向实际来自肘关节 18/19 代理）。

## 目录

```
experiments/week1_pos_cls/
  README.md
  configs/default.yaml
  dataset/
    build_amass_pos_cls.py
    build_imuposer_pos_cls.py
    pos_dataset.py
  models/pos_classifier.py
  train.py
  eval.py
  visualize.py
  outputs/{checkpoints,logs,figures,data}
```

## 使用步骤（在仓库根目录执行）

### 0. 环境与路径
```bash
conda activate mobileposer
cd /home/caolindong/projects/mobileposer
```
编辑 `configs/default.yaml`：
- `data.processed_amass_dir`（默认 `data/processed`）
- `data.processed_imuposer_file`（默认 `data/processed/eval/imuposer_full.pt`）

原始数据应放在仓库根目录：
- `data/raw/AMASS/`
- `data/raw/IMUPoser/`

主配置见 `mobileposer/config.py` 的 `paths`（已指向上述本地 `data/`）。

先跑通 MobilePoser 预处理（若还没有 `.pt`）：
```bash
cd mobileposer
python data_process_mocap.py --dataset amass
python data_process_mocap.py --dataset imuposer
cd ..
```

### 1. 造训练/验证集（AMASS）
```bash
python experiments/week1_pos_cls/dataset/build_amass_pos_cls.py \
  --config experiments/week1_pos_cls/configs/default.yaml
```
输出：`outputs/data/amass_train.pt`、`amass_val.pt`、`norm_stats.pt`

### 2. 造测试集（IMUPoser）
**不要猜标签。** 按真实佩戴填写，例如左腕+右袋：
```bash
python experiments/week1_pos_cls/dataset/build_imuposer_pos_cls.py \
  --config experiments/week1_pos_cls/configs/default.yaml \
  --watch-side 0 --phone-side 1
```

### 3. 训练
```bash
python experiments/week1_pos_cls/train.py \
  --config experiments/week1_pos_cls/configs/default.yaml
```
最优权重：`outputs/checkpoints/best_joint_acc.pt`  
日志：`outputs/logs/train_log.csv`

### 4. 评估
```bash
python experiments/week1_pos_cls/eval.py --split val
python experiments/week1_pos_cls/eval.py --split test
```

### 5. 可视化
```bash
python experiments/week1_pos_cls/visualize.py --split test
```
图在 `outputs/figures/`。

## 结果记录

| Split | Watch Acc | Phone Acc | Joint Acc | Seq Joint Acc | 日期 | 备注 |
|-------|-----------|-----------|-----------|---------------|------|------|
| AMASS Val | | | | | | |
| IMUPoser Test | | | | | | |

## 已知注意点
- IMUPoser 标签必须人工确认；脚本拒绝无标签运行。
- 腕 IMU 特征来自肘关节朝向代理，报告里写清楚。
- 静止段左右难分：可后续按动态/准静态分开统计。
