# Week1 — Device Position Classification v1

> 分支建议：`test` / `exp/week1-pos-cls`  
> 目标：忽略朝向干扰，预测手表左右腕 + 手机左右袋。

## 实验决策

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
官方IMUPoser录制时，每人同时戴了全部五个设备（0——左手腕手表，1——右手腕手表，2——左口袋手机，3——右口袋手机，4——头上的耳机）。本实验测试从0、1、2、3四路中挑两路作为一只手表、一只手机的输入。

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
# 单组合（默认读 imuposer_test_{lw|rw}_{lp|rp}.pt）
python experiments/week1_pos_cls/eval.py --split both \
  --watch-side 0 --phone-side 1

# 多组合一次跑完
python experiments/week1_pos_cls/eval.py --split test \
  --combos lw_lp,lw_rp,rw_lp,rw_rp
```
产物写在 `outputs/logs/`（**不入 git**，数字记到下方结果表）：
- `metrics_val.json`
- `metrics_test_{lw_lp,lw_rp,rw_lp,rw_rp}.json`
- `metrics_summary.json`

### 5. 可视化
```bash
python experiments/week1_pos_cls/visualize.py --split test
```
图在 `outputs/figures/`。

## 结果记录

配置：`default.yaml`；训练窗长 W=90；checkpoint=`best_joint_acc.pt`。  
AMASS Val / IMUPoser LW+RP：2026-07-29；其余三组合：2026-08-02。  
每个 IMUPoser 组合均为 n=6003 窗 / 167 序列。

### 主表（W=90）

| Split | Combo | Watch Acc | Phone Acc | Joint Acc | Seq Joint Acc |
|-------|-------|-----------|-----------|-----------|---------------|
| AMASS Val | 4 组合混合 | 0.977 | 0.948 | **0.928** | **0.976** |
| IMUPoser | LW+LP | 0.885 | 0.617 | **0.557** | **0.467** |
| IMUPoser | LW+RP | 0.891 | 0.790 | **0.719** | **0.749** |
| IMUPoser | RW+LP | 0.937 | 0.594 | **0.546** | **0.461** |
| IMUPoser | RW+RP | 0.944 | 0.793 | **0.755** | **0.778** |

- **Joint Acc**：单窗上手表+手机都对。
- **Seq Joint Acc**：同一序列内对窗预测做多数投票后再算 Joint（更接近「初始化」判定）。

### 窗长消融（Joint Acc）

| W | AMASS Val | LW+LP | LW+RP | RW+LP | RW+RP |
|---|-----------|-------|-------|-------|-------|
| 30 | 0.828 | 0.471 | 0.626 | 0.477 | 0.656 |
| 60 | 0.899 | 0.533 | 0.691 | 0.519 | 0.722 |
| 90 | 0.928 | 0.557 | 0.719 | 0.546 | 0.755 |
| 150 | 0.952 | 0.601 | 0.762 | 0.590 | 0.784 |

### init_done（K=30 帧 ≈1s 预测稳定）

| Split / Combo | 触发率 | 稳定预测正确率 | 中位耗时 |
|---------------|--------|----------------|----------|
| AMASS Val | 0.924 | 0.960 | 1.0 s |
| IMUPoser LW+LP | 1.000 | 0.467 | 1.0 s |
| IMUPoser LW+RP | 1.000 | 0.677 | 1.0 s |
| IMUPoser RW+LP | 1.000 | 0.527 | 1.0 s |
| IMUPoser RW+RP | 1.000 | 0.790 | 1.0 s |

### 简要结论
- 合成域（AMASS）很强（Joint 0.93）；真实域四组合 Joint 约 **0.55–0.76**。
- **右袋（RP）明显好于左袋（LP）**：LW+RP 0.72 / RW+RP 0.76，而 LW+LP / RW+LP 仅 ~0.55。
- 手表侧整体不难（Watch Acc 0.88–0.94）；瓶颈主要在手机侧（Phone Acc 在 LP 组合掉到 ~0.60）。
- 更长窗抬高窗级 Acc；序列级 Acc 对窗长相对不敏感。
- 动态动作通常更好；静止/上肢复杂动作更难（详见各 `metrics_test_*.json` 的 `by_motion`）。

## 已知注意点
- IMUPoser 官方数据是 5 路全开；测试标签 = watch/phone 槽位，不是 zip 里另附的佩戴元数据。
- 腕 IMU 特征来自肘关节朝向代理，报告里写清楚。
- 静止段左右难分：可后续按动态/准静态分开统计。
- `outputs/` 下 `.pt` / checkpoint / metrics JSON 视为可复现产物，默认不提交；以本 README 结果表为对外记录。
