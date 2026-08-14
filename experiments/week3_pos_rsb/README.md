# Week3 — Unknown-mount cascade (position → \(R_{SB}\) → pose)

> 相对 Week1 / Week2 的关系  
> **Week1**：忽略朝向干扰，只做位置分类（acc+gyro）  
> **Week2**：位置 **已知**，估 \(R_{SB}\)  
> **Week3**：输入只有 \(a_M, R_{MS}\)，\(R_{SB}\) **未知**；先位置，再外参，再姿态

## 第 1 步（本目录当前内容）：未知朝向推测设备位置

从 \(a_M, R_{MS}\) 推断设备位置。遵循附图协议：

\[
R_{MS} = R_{MB}\, R_{BS},\quad a_M \text{ 不变}
\]

- 表 / 机各自独立随机 \(R_{BS}\)（欧拉角均匀 \(\pm 45^\circ\)）
- 整条序列（因而同一窗内每一帧）\(R_{BS}\) 恒定
- 网络 **看不到** \(R_{BS}\)，只输出手表左右腕、手机左右袋

## 实验决策

| 项 | 取值 |
|----|------|
| 设备 | 1 watch + 1 phone |
| 手表位置 | 0=左腕(LW), 1=右腕(RW) |
| 手机位置 | 0=左袋(LP), 1=右袋(RP) |
| 输入 | 每设备 `acc(3)+ori_flat(9)` ×2 → **24 维** |
| 窗长 / 训练步长 | 90 帧 / 15 帧（约 3s @30FPS） |
| 朝向 | **未知**：\(R_{MS}=R_{MB}R_{BS}\)，\(a_M\) 不拧；\(R_{BS}\) 序列级恒定 |
| 训练 | AMASS 合成（每条序列枚举 4 种佩戴组合，在线注入 \(R_{BS}\)） |
| 测试 | IMUPoser 录制流当作 \(R_{MB}\) 再 inject（与 Week2 D5 一致） |
| 主指标 | **Joint Accuracy**（手表+手机都对） |
| 最优权重 | epoch **13** / 共 40；`best_joint_acc.pt` |

IMU 槽位与 MobilePoser 一致：`0=LW, 1=RW, 2=LP, 3=RP`。  
模型：与 Week1 相同的双头 BiLSTM，仅输入维从 12 改为 24。  
数据：在线注入，不把展开窗写到磁盘。

## 目录

```
experiments/week3_pos_rsb/
  README.md
  configs/default.yaml
  dataset/
    build_amass.py
    pos_dataset.py
  models/pos_classifier.py
  train.py
  eval.py
  visualize.py
  outputs/{checkpoints,logs,figures,data}
```

## 使用步骤（在仓库根目录执行）

### 0. 环境
```bash
conda activate mobileposer
cd /home/caolindong/projects/mobileposer
```

### 1. 造 AMASS 索引 + 归一化统计（不写展开窗）
```bash
python experiments/week3_pos_rsb/dataset/build_amass.py \
  --config experiments/week3_pos_rsb/configs/default.yaml
```
输出：`outputs/data/amass_index.pt`、`norm_stats.pt`

### 2. 训练
```bash
python experiments/week3_pos_rsb/train.py \
  --config experiments/week3_pos_rsb/configs/default.yaml
```
最优权重：`outputs/checkpoints/best_joint_acc.pt`  
日志：`outputs/logs/train_log.csv`  
已有 `last.pt` 时会从下一 epoch 续训。

### 3. 评估
```bash
python experiments/week3_pos_rsb/eval.py \
  --config experiments/week3_pos_rsb/configs/default.yaml
```
产物写在 `outputs/logs/`（**不入 git**，数字记到下方结果表）：
- `metrics_val.json`
- `metrics_test_{lw_lp,lw_rp,rw_lp,rw_rp}.json`
- `metrics_summary.json` / `metrics_step1.json`

### 4. 可视化（序列时间线 + SMPL mesh 标注）
```bash
# 默认：自动挑好/坏序列，输出 timeline + mesh 拼图 + 混淆矩阵
python experiments/week3_pos_rsb/visualize.py --split test --combo lw_rp

# 指定序列（可与 Week1 的 110/12 对照）
python experiments/week3_pos_rsb/visualize.py --split test --combo lw_rp --seq-ids 13,8

# 按 seq-id 顺序导出全部序列的 GT/Pred timeline
python experiments/week3_pos_rsb/visualize.py --split test --combo lw_rp --all-sequences

# 指定序列并导出 GIF
python experiments/week3_pos_rsb/visualize.py --split test --combo lw_rp \
  --seq-ids 13 --gif --max-mesh-frames 30
```
图在 `outputs/figures/`（`--all-sequences` 时在子目录 `timelines_all_test_{combo}/`）：
- `timeline_seq{ID:03d}_{good|mid|bad}_test_{combo}.png`：横轴 Frame，纵轴位置 0/1/2/3，GT 实线 / Pred 虚线
- `mesh_seq{ID}_test_{combo}.png` / `.gif`：GT 姿态 mesh + 位置标注
- `cm_*.png`：混淆矩阵

说明：mesh 姿态来自 IMUPoser GT pose；颜色表示本步位置分类结果（不是姿态预测）。

## 结果记录（第 1 步）

配置：`default.yaml`；训练窗长 W=90；checkpoint=`best_joint_acc.pt`（epoch 13）。  
评测日期：2026-08-14。  
每个 IMUPoser 组合均为 n=6003 窗 / 167 序列（与 Week1 相同切窗）。

### 主表（W=90）

| Split | Combo | Watch Acc | Phone Acc | Joint Acc | Seq Joint Acc |
|-------|-------|-----------|-----------|-----------|---------------|
| AMASS Val | 4 组合混合 | 0.971 | 0.851 | **0.832** | **0.918** |
| IMUPoser | LW+LP | 0.964 | 0.672 | **0.646** | **0.593** |
| IMUPoser | LW+RP | 0.940 | 0.581 | **0.545** | **0.521** |
| IMUPoser | RW+LP | 0.924 | 0.715 | **0.666** | **0.605** |
| IMUPoser | RW+RP | 0.925 | 0.676 | **0.636** | **0.605** |

- **Joint Acc**：单窗上手表+手机都对。
- **Seq Joint Acc**：同一序列内对窗预测做多数投票后再算 Joint（更接近「初始化」判定）。

### 窗长消融（Joint Acc）

| W | AMASS Val | LW+LP | LW+RP | RW+LP | RW+RP |
|---|-----------|-------|-------|-------|-------|
| 30 | 0.755 | 0.575 | 0.520 | 0.604 | 0.582 |
| 60 | 0.809 | 0.616 | 0.541 | 0.646 | 0.623 |
| 90 | 0.832 | 0.646 | 0.545 | 0.666 | 0.636 |
| 150 | 0.853 | 0.677 | 0.550 | 0.682 | 0.655 |

### init_done（K=30 帧 ≈1s 预测稳定）

| Split / Combo | 触发率 | 稳定预测正确率 | 中位耗时 |
|---------------|--------|----------------|----------|
| AMASS Val | 0.926 | 0.875 | 1.0 s |
| IMUPoser LW+LP | 1.000 | 0.617 | 1.0 s |
| IMUPoser LW+RP | 1.000 | 0.533 | 1.0 s |
| IMUPoser RW+LP | 1.000 | 0.659 | 1.0 s |
| IMUPoser RW+RP | 1.000 | 0.611 | 1.0 s |

### 廉价验证：训练组合是否均衡？

检查产物：`outputs/logs/metrics_val.json` 的 `cheap_checks`。

| 检查 | 结果 |
|------|------|
| `amass_train` 四组合计数 | 各 **25.00%**（完全均衡，每组合 84103 窗） |
| `amass_val` 四组合计数 | 各 **25.00%**（每组合 19909 窗） |

真机 LP/RP 差异 **不是** 训练集组合偏斜造成的。

### 按运动类型（by_motion，四组合窗级加权平均，W=90）

数据来自 `outputs/logs/metrics_test_{lw_lp,lw_rp,rw_lp,rw_rp}.json` 的 `by_motion`。  
IMUPoser 原始命名有拼写/大小写变体，下表已合并同义名（如 `Joogging→Jogging`、`Lower Body→LowerBody`、`Armcrossing/ArmsCrossing→ArmCrossing`、`startClap*→StartClapping` 等）。`n` 为四组合窗数之和。

**准确率较高（全身/下肢动态为主）**

| 运动 | n | Joint | Watch | Phone |
|------|--:|------:|------:|------:|
| Walking | 6128 | **0.807** | 0.988 | 0.818 |
| Kicking | 436 | **0.750** | 0.995 | 0.755 |
| Jogging | 1704 | **0.645** | 0.917 | 0.688 |
| LowerBody | 2696 | **0.634** | 0.897 | 0.709 |
| Hopping | 656 | **0.611** | 0.963 | 0.637 |

**准确率较低（上肢精细/对称动作，Phone 侧常崩）**

| 运动 | n | Joint | Watch | Phone |
|------|--:|------:|------:|------:|
| ArmCrossing | 784 | **0.522** | 0.980 | 0.531 |
| Waving | 836 | **0.505** | 0.917 | 0.555 |
| HeadMovements | 2236 | **0.496** | 0.833 | 0.591 |
| ArmSwings | 1236 | **0.464** | 0.959 | 0.488 |
| Punching | 76 | **0.382** | 0.947 | 0.382 |
| ClappingFull | 368 | **0.345** | 0.821 | 0.421 |

中游常见：`TennisSwings` / `Basketball` / `ArmRaises` / `Pushups` / `Boxing` / `Sitting` / `JumpingJacks`（Joint 约 0.53–0.61）。

观察：
- **高分运动**仍是 Walking、Kicking、Jogging、LowerBody、Hopping：身体整体加速度大，左右袋相对可分。
- **低分运动**仍是上肢动作：Watch 往往仍高（0.82–0.98），**Phone Acc 掉到 ~0.38–0.55**，拖垮 Joint。
- JumpingJacks 在未知朝向下掉得特别明显（见与 Week1 对照）。
- 单个「好序列」≠该运动整体容易：如 ClappingFull 整体 Joint 仅 0.345，但 RW+LP 下 seq38 可到 1.0。

### 定性可视化

推荐示例（combo=`lw_rp`，与 Week1 同一可视化协议）：

| 类型 | seq-id | 运动 | 窗级 joint-acc |
|------|--------|------|----------------|
| 好序列 | **13** | P1/Kicking | 1.000 |
| 坏序列 | **8** | P1/ArmSwing | 0.000 |
| Week1 对照好 | 110 | P6/Walking | 0.713（Week1 为 0.993） |
| Week1 对照坏 | 12 | P1/Boxing | 0.480（Week1 为 0.000） |

图：
- `outputs/figures/timeline_seq013_good_test_lw_rp.png`
- `outputs/figures/timeline_seq008_bad_test_lw_rp.png`
- `outputs/figures/mesh_seq013_test_lw_rp.png` / `.gif`
- `outputs/figures/mesh_seq008_test_lw_rp.png` / `.gif`
- `outputs/figures/cm_{watch,phone,joint4}_test_{combo}.png`
- 全序列时间线：`outputs/figures/timelines_all_test_lw_rp/`（167 条）

其它组合自动挑选的好/坏序列：LW+LP seq39 Hopping / seq100 HeadMovements；RW+LP seq38 ClappingFull / seq83 HeadMovements；RW+RP seq13 Kicking / seq100 HeadMovements。

## 与 Week1（无朝向干扰）对照

对照设定：同一批 AMASS 子集、同一 IMUPoser 切窗、同一双头 BiLSTM、同一 W=90。  
差别只有输入与朝向协议：

| | Week1 | Week3 第 1 步 |
|--|-------|----------------|
| 输入 | acc+gyro **12 维** | acc+\(R_{MS}\) **24 维** |
| 朝向 | **忽略**（不做 mount offset） | **未知** \(R_{BS}\sim\pm 45^\circ\)，序列恒定 |
| 加速度 | 原始 \(a_M\) | 原始 \(a_M\)（不拧） |

### 主指标对照（Joint Acc）

| Split | Week1 | Week3 | Δ |
|-------|------:|------:|--:|
| AMASS Val | **0.928** | 0.832 | −0.096 |
| IMUPoser LW+LP | 0.557 | **0.646** | +0.089 |
| IMUPoser LW+RP | **0.719** | 0.545 | −0.174 |
| IMUPoser RW+LP | 0.546 | **0.666** | +0.120 |
| IMUPoser RW+RP | **0.755** | 0.636 | −0.119 |

Watch / Phone 拆开看：

| Split | Week1 W / P | Week3 W / P |
|-------|-------------|-------------|
| AMASS Val | 0.977 / **0.948** | 0.971 / **0.851** |
| LW+LP | 0.885 / 0.617 | **0.964** / **0.672** |
| LW+RP | 0.891 / **0.790** | **0.940** / 0.581 |
| RW+LP | 0.937 / 0.594 | 0.924 / **0.715** |
| RW+RP | **0.944** / **0.793** | 0.925 / 0.676 |

Seq Joint 对照：Week1 四组合 0.467 / 0.749 / 0.461 / 0.778；Week3 为 0.593 / 0.521 / 0.605 / 0.605。未知朝向后序列级判定被「抹平」，不再出现 Week1 那种 RP 组合明显更高的两极分化。

### 差异解读

1. **合成域（AMASS）是更干净的对照。** Val Joint 从 0.928 降到 0.832，几乎全部来自 **Phone**（0.948→0.851）；Watch 几乎不变（0.977→0.971）。未知 \(R_{BS}\) 主要伤害口袋左右可分性：拧朝向后 \(R_{MS}\) 不再直接等于骨骼朝向，口袋两侧的重力/朝向线索被扰乱，而腕部左右仍可主要靠加速度摆臂分辨。
2. **真机 LP≪RP 在未知朝向下反转/消失。** Week1 是 RP 明显更好（Joint 0.72/0.76 vs LP 0.55/0.55）；Week3 变成 **LP 更好**（0.65/0.67 vs RP 0.55/0.64），最差组合从 LP 换成了 **LW+RP（0.545）**。init_done 正确率同样从「RP 高」变成「LP 略高」。说明 Week1 的 RP 优势高度依赖「朝向未被扰乱」的口袋信号；一旦 \(R_{MS}\) 带未知 \(R_{BS}\)，这条捷径失效。
3. **Watch 在真机上不降反升。** 四组合 Watch Acc 从 0.88–0.94 到 0.92–0.96。Week3 输入是 \(a_M+R_{MS}\) 而不是 acc+gyro：即使 \(R_{BS}\) 未知，旋转矩阵仍提供比有限差分 gyro 更稳的腕部左右线索。瓶颈仍然在 Phone。
4. **运动层面：下肢动态仍最稳，上肢动作更差，JumpingJacks 掉档。** Walking 仍最高（0.870→0.807）；Jogging 0.814→0.645；JumpingJacks 从 Week1 的 0.744 掉到 0.532（对称开合 + 未知朝向让口袋更难分）。ClappingFull / Punching 两边都差。
5. **窗长趋势一致。** 两边都是更长窗抬高窗级 Acc；未知朝向下 AMASS 的 W=30→150 为 0.755→0.853（Week1 为 0.828→0.952），绝对值更低但斜率类似。序列级对窗长仍相对不敏感。
6. **init_done。** 触发率两边都接近 1s 中位；AMASS 稳定正确率 0.960→0.875，与 Joint 下降同量级。真机正确率不再跟 RP 绑定。

### 简要结论
- 未知朝向下，位置分类 **仍然可做**，但比 Week1 更难：合成域 Joint 约 0.83（−10 个点），真机四组合 Joint 约 **0.55–0.67**（Week1 为 0.55–0.76）。
- 瓶颈仍是 **手机左右袋**；手表左右腕在未知 \(R_{BS}\) 下依然很容易。
- Week1 的「右袋明显好于左袋」**不能**推广到未知朝向设定；本步 LP/RP 差距缩小甚至反转。
- 更长窗仍然有用；Walking / Kicking 等全身动态最好，上肢精细动作最差。
- 本步输出的 Pred 槽位将作为第 2 步估 \(R_{SB}\) 的条件。

## 后续步骤（未实现）

2. 用 Pred 位置条件化，估 \(R_{SB}\)（接 Week2）  
3. \(R_{MB}=R_{MS}R_{SB}^{\top}\) → MobilePoser 姿态，并做 \(R_{MS}\to\) pose baseline 可视化

## 已知注意点
- IMUPoser 官方数据是 5 路全开；测试时把录制 ori 当作 \(R_{MB}\) 再乘随机 \(R_{BS}\)。
- 与 Week1 对照时，输入模态也不相同（gyro vs \(R_{MS}\)），AMASS Val 是更干净的「只改朝向协议」对照。
- `outputs/` 下 checkpoint / metrics JSON / 图视为可复现产物，默认不提交；以本 README 结果表为对外记录。
