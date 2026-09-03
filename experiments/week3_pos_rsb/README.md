# Week3 — Unknown-mount cascade (position → \(R_{SB}\) → pose)

> 相对 Week1 / Week2 的关系  
> **Week1**：忽略朝向干扰，只做位置分类（acc+gyro）  
> **Week2**：位置 **已知**，估 \(R_{SB}\)  
> **Week3**：输入只有 \(a_M, R_{MS}\)，\(R_{SB}\) **未知**；先位置，再外参，再姿态
>
> **对外口径（2026-09-03 起）：** \(R_{BS}\) **窗内恒定、序列内可变**；XYZ 欧拉**每个轴均匀 \([0^\circ, 180^\circ]\)**（不是旧的 \(\pm 45^\circ\)，也不是轴角幅度 180°）。  
> 下方第 1–3 步结果表若未另行标明，仍是 **\(\pm 45^\circ\)** 旧采样，不能与新范围混用。
> 不要与 08-14/08-15「整段序列一个 \(R_{BS}\)」旧表混用。旧权重备份：`outputs/checkpoints/seq_constant/`。

## 第 1 步：未知朝向推测设备位置

从 \(a_M, R_{MS}\) 推断设备位置。遵循附图协议：

\[
R_{MS} = R_{MB}\, R_{BS},\quad a_M \text{ 不变}
\]

- 表 / 机各自独立随机 \(R_{BS}\)（XYZ 欧拉**每个轴**均匀 \([0^\circ, 180^\circ]\)）
- **一个窗（90 帧）内 \(R_{BS}\) 恒定**；同一条序列的不同窗可以不同
- 网络 **看不到** \(R_{BS}\)，只输出手表左右腕、手机左右袋

## 实验决策

| 项 | 取值 |
|----|------|
| 设备 | 1 watch + 1 phone |
| 手表位置 | 0=左腕(LW), 1=右腕(RW) |
| 手机位置 | 0=左袋(LP), 1=右袋(RP) |
| 输入 | 每设备 `acc(3)+ori_flat(9)` ×2 → **24 维** |
| 窗长 / 训练步长 | 90 帧 / 15 帧（约 3s @30FPS） |
| 朝向 | **未知**：\(R_{MS}=R_{MB}R_{BS}\)，\(a_M\) 不拧；\(R_{BS}\) **窗内恒定**；欧拉每轴 \([0^\circ,180^\circ]\) |
| 训练 | AMASS 合成（每条序列枚举 4 种佩戴组合，在线注入 \(R_{BS}\)） |
| 测试 | IMUPoser 录制流当作 \(R_{MB}\) 再 inject（与 Week2 D5 一致） |
| 主指标 | **Joint Accuracy**（手表+手机都对） |
| 最优权重 | epoch **28** / 共 40；`best_joint_acc.pt`（窗内恒定协议） |

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
  models/rot_extrinsic_dual.py   # Week2 dual 结构拷贝（不重训）
  train.py
  eval.py / visualize.py         # 第 1 步
  eval_step2.py / visualize_step2.py
  eval_step3.py / visualize_step3.py / cascade_pose.py
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

### 5. 第 2 步：Pred 槽位 → \(R_{SB}\)（冻结 Week2 dual）
```bash
python experiments/week3_pos_rsb/eval_step2.py \
  --config experiments/week3_pos_rsb/configs/default.yaml

python experiments/week3_pos_rsb/visualize_step2.py
```
产物：`outputs/logs/metrics_step2.json`；图 `outputs/figures/step2_*.png`。  
**不重训**外参网络：加载 `week2_rot_ext/.../best_rot_err_dual.pt`（epoch 38）与 `norm_stats_dual.pt`。位置网仍用本目录 `norm_stats.pt`。

### 6. 第 3 步：校准 \(R_{MB}\) 接 MobilePoser
```bash
python experiments/week3_pos_rsb/eval_step3.py \
  --config experiments/week3_pos_rsb/configs/default.yaml

python experiments/week3_pos_rsb/visualize_step3.py --combo lw_rp --seq-ids 13,8
```
产物：`outputs/logs/metrics_step3.json`；图 `outputs/figures/step3_*.png`。  
需官方 `checkpoints/weights.pth`。默认 12 序列 × 四组合（与 Week2 姿态下游相同），打包 `[watch, phone, Head]`，Head 不注入。

## 结果记录（第 1 步）

配置：`default.yaml`；训练窗长 W=90；checkpoint=`best_joint_acc.pt`（epoch 28）。  
评测日期：2026-08-16（窗内恒定协议）。  
每个 IMUPoser 组合均为 n=6003 窗 / 167 序列（与 Week1 相同切窗）。

### 主表（W=90）

| Split | Combo | Watch Acc | Phone Acc | Joint Acc | Seq Joint Acc |
|-------|-------|-----------|-----------|-----------|---------------|
| AMASS Val | 4 组合混合 | 0.984 | 0.915 | **0.903** | **0.974** |
| IMUPoser | LW+LP | 0.974 | 0.747 | **0.727** | **0.778** |
| IMUPoser | LW+RP | 0.975 | 0.650 | **0.642** | **0.725** |
| IMUPoser | RW+LP | 0.967 | 0.789 | **0.769** | **0.874** |
| IMUPoser | RW+RP | 0.953 | 0.588 | **0.563** | **0.527** |

- **Joint Acc**：单窗上手表+手机都对。
- **Seq Joint Acc**：同一序列内对窗预测做多数投票后再算 Joint（佩戴槽位整段仍恒定；变的是每窗的 \(R_{BS}\)）。

### 窗长消融（Joint Acc）

| W | AMASS Val | LW+LP | LW+RP | RW+LP | RW+RP |
|---|-----------|-------|-------|-------|-------|
| 30 | 0.802 | 0.663 | 0.584 | 0.689 | 0.514 |
| 60 | 0.871 | 0.706 | 0.626 | 0.749 | 0.552 |
| 90 | 0.903 | 0.727 | 0.642 | 0.769 | 0.563 |
| 150 | 0.929 | 0.756 | 0.655 | 0.805 | 0.583 |

### init_done（K=30 帧 ≈1s 预测稳定）

| Split / Combo | 触发率 | 稳定预测正确率 | 中位耗时 |
|---------------|--------|----------------|----------|
| AMASS Val | 0.919 | 0.945 | 1.0 s |
| IMUPoser LW+LP | 0.994 | 0.783 | 1.0 s |
| IMUPoser LW+RP | 1.000 | 0.754 | 1.0 s |
| IMUPoser RW+LP | 0.994 | 0.759 | 1.0 s |
| IMUPoser RW+RP | 0.988 | 0.564 | 1.0 s |

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
| Walking | 6128 | **0.823** | 0.990 | 0.831 |
| Kicking | 436 | **0.821** | 0.986 | 0.835 |
| TennisSwings | 896 | **0.776** | 0.984 | 0.785 |
| Boxing | 556 | **0.712** | 0.935 | 0.754 |
| LowerBody | 2696 | **0.662** | 0.970 | 0.675 |
| Jogging | 1704 | **0.660** | 0.923 | 0.711 |

**准确率较低（上肢精细/对称动作，Phone 侧常崩）**

| 运动 | n | Joint | Watch | Phone |
|------|--:|------:|------:|------:|
| ArmCrossing | 784 | **0.529** | 0.997 | 0.529 |
| JumpingJacks | 776 | **0.513** | 0.905 | 0.564 |
| Pushups | 392 | **0.485** | 0.916 | 0.528 |
| ClappingFull | 368 | **0.432** | 0.823 | 0.505 |

中游常见：`Basketball` / `ArmSwing` / `Waving` / `Sitting` / `ArmRaises` / `Hopping` / `HeadMovements`（Joint 约 0.57–0.66）。

观察：
- **高分运动**仍是 Walking、Kicking、TennisSwings、Boxing：身体整体加速度大，左右袋相对可分。
- **低分运动**仍是上肢/对称动作：Watch 往往仍高（0.82–1.00），**Phone Acc 掉到 ~0.50–0.56**，拖垮 Joint。
- JumpingJacks 在未知朝向下仍明显低于 Week1。
- 单个「好序列」≠该运动整体容易。

### 定性可视化

推荐示例（combo=`lw_rp`，与 Week1 同一可视化协议；主片用于第 3 步三栏 mesh）：

| 类型 | seq-id | 运动 |
|------|--------|------|
| 好序列 | **13** | P1/Kicking |
| 坏序列 | **8** | P1/ArmSwing |
| Week1 对照好 | 110 | P6/Walking |
| Week1 对照坏 | 12 | P1/Boxing |

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
| 朝向 | **忽略**（不做 mount offset） | **未知** \(R_{BS}\)（下表数字是欧拉 \(\pm 45^\circ\)；现行配置已改为每轴 \([0^\circ,180^\circ]\)） |
| 加速度 | 原始 \(a_M\) | 原始 \(a_M\)（不拧） |

### 主指标对照（Joint Acc）

| Split | Week1 | Week3 | Δ |
|-------|------:|------:|--:|
| AMASS Val | **0.928** | 0.903 | −0.025 |
| IMUPoser LW+LP | 0.557 | **0.727** | +0.170 |
| IMUPoser LW+RP | **0.719** | 0.642 | −0.077 |
| IMUPoser RW+LP | 0.546 | **0.769** | +0.223 |
| IMUPoser RW+RP | **0.755** | 0.563 | −0.192 |

Watch / Phone 拆开看：

| Split | Week1 W / P | Week3 W / P |
|-------|-------------|-------------|
| AMASS Val | 0.977 / **0.948** | 0.984 / **0.915** |
| LW+LP | 0.885 / 0.617 | **0.974** / **0.747** |
| LW+RP | 0.891 / **0.790** | **0.975** / 0.650 |
| RW+LP | 0.937 / 0.594 | 0.967 / **0.789** |
| RW+RP | **0.944** / **0.793** | 0.953 / 0.588 |

Seq Joint 对照：Week1 四组合 0.467 / 0.749 / 0.461 / 0.778；Week3 为 0.778 / 0.725 / 0.874 / 0.527。未知朝向后不再是 Week1 那种「RP 序列级明显更高」。

### 差异解读

1. **合成域（AMASS）掉得不多。** Val Joint 从 0.928 降到 **0.903**（约 −2.5 点），几乎全部来自 **Phone**（0.948→0.915）；Watch 持平或略升（0.977→0.984）。未知 \(R_{BS}\) 主要伤害口袋左右可分性：\(R_{MS}\) 不再直接等于骨骼朝向。整段恒定旧协议曾掉到 0.832；窗内恒定后合成域更接近 Week1。
2. **真机不再是「右袋明显更好」。** Week1 是 RP 明显更好（Joint 0.72/0.76 vs LP 0.55/0.55）；本协议下 **LP 组合更好**（0.73/0.77 vs RP 0.64/0.56），最差换成 **RW+RP（0.563）**。Week1 的 RP 优势依赖「朝向未被扰乱」的口袋信号。
3. **Watch 在真机上不降反升。** 四组合 Watch Acc 从 0.88–0.94 到 0.95–0.97。瓶颈仍然在 Phone。
4. **运动层面：** Walking / Kicking 仍最高（Joint ≈0.82）；JumpingJacks / ClappingFull / ArmCrossing 最差。Watch 高、Phone 拖垮 Joint。
5. **窗长趋势一致。** 更长窗抬高窗级 Acc；AMASS W=30→150 为 0.802→0.929。
6. **init_done。** 中位仍约 1 s；AMASS 稳定正确率 0.945。真机正确率 0.56–0.78，RW+RP 最低。

### 简要结论
- 未知朝向下位置分类 **仍然可做**：合成域 Joint **0.90**（相对 Week1 仅 −2.5 点）；真机四组合 Joint 约 **0.56–0.77**。
- 瓶颈仍是 **手机左右袋**；手表左右腕在未知 \(R_{BS}\) 下依然很容易。
- Week1 的「右袋明显好于左袋」**不能**推广到未知朝向设定。
- 更长窗仍然有用；Walking / Kicking 等全身动态最好，上肢精细动作最差。
- 本步输出的 Pred 槽位将作为第 2 步估 \(R_{SB}\) 的条件。

## 第 2 步：由 Pred 位置推断 \(R_{SB}\)

不重训 \(R_{SB}\) 网络。冻结 Week2 双设备 `RotExtrinsicDualNet`（`best_rot_err_dual.pt`，epoch 38），把第 1 步槽位转成 Week2 绝对槽后作为 one-hot 条件。

槽位映射：第 1 步 \(y\in\{0,1\}\times\{0,1\}\) → `slot_watch∈{0,1}`、`slot_phone∈{2,3}`。  
特征都是 24 维 acc+\(R_{MS}\)；**位置网用本目录 `norm_stats.pt`，外参网用 Week2 `norm_stats_dual.pt`**。

协议与第 1 步相同：\(R_{MS}=R_{MB}R_{BS}\)，\(a_M\) 不拧，\(R_{BS}\) 窗内恒定、序列内可变。

对照（单位 °，越低越好）：

| 条件 | 含义 |
|------|------|
| **None** | \(\hat{R}_{SB}=I\) |
| **Pred-win** | 每窗用第 1 步 Pred 槽位 |
| **Pred-seq** | 同一 (序列, 组合) 上多数投票槽位再喂 Week2（槽位整段恒定；\(R_{BS}\) 仍按窗变） |
| **GT-slot** | 真值位置喂 Week2（本协议上界，应≈ Week2 dual） |
| **Oracle** | 完美 \(R_{SB}\)，0° |

评测日期：2026-08-16。W=90；AMASS val n=79636；IMUPoser 每组合 n=6003 / 167 序列。

### 主表（\(R_{SB}\) 平均测地线角误差 °）

| Split | None | Pred-win | Pred-seq | GT-slot |
|-------|-----:|---------:|---------:|--------:|
| AMASS Val | 42.82 | 15.75 | **15.15** | 15.12 |
| IMUPoser LW+LP | 42.69 | 26.39 | **25.36** | 25.22 |
| IMUPoser LW+RP | 42.94 | 27.41 | **26.63** | 26.82 |
| IMUPoser RW+LP | 42.74 | 27.85 | **26.90** | 26.57 |
| IMUPoser RW+RP | 42.77 | 28.86 | **27.51** | 27.40 |

Watch / Phone 拆开（Pred-seq / GT-slot）：

| Split | Pred-seq W / P | GT-slot W / P |
|-------|----------------|---------------|
| AMASS Val | 18.58 / 11.71 | 18.57 / 11.67 |
| LW+LP | 29.36 / 21.37 | 29.38 / 21.05 |
| LW+RP | 29.74 / 23.52 | 29.74 / 23.90 |
| RW+LP | 32.24 / 21.56 | 31.87 / 21.26 |
| RW+RP | 32.33 / 22.68 | 32.17 / 22.63 |

### 分层：位置对不对，外参差多少

第 1 步 Joint 正确时，Pred-win 与 GT-slot **逐窗相同**（槽位一致，外参网冻结）。错误时才出现级联惩罚。

| Split | Joint OK n | Pred = GT-slot | Joint BAD n | Pred-win | GT-slot |
|-------|-----------:|---------------:|------------:|---------:|--------:|
| AMASS Val | 71894 | **14.38** | 7742 | 28.47 | 22.05 |
| LW+LP | 4365 | **24.12** | 1638 | 32.44 | 28.15 |
| LW+RP | 3853 | **25.27** | 2150 | 31.24 | 29.59 |
| RW+LP | 4618 | **25.45** | 1385 | 35.87 | 30.30 |
| RW+RP | 3382 | **26.46** | 2621 | 31.95 | 28.61 |

### 与 Week2 dual（位置已知）对照

Week2 dual 在 **已知槽位、只拧朝向** 下：AMASS val **15.12°**（watch 18.55 / phone 11.68）；IMUPoser D5 为 40 条序列子集 **26.03°**。

| | Week2 dual（GT 槽） | Week3 GT-slot | Week3 Pred-seq |
|--|--------------------:|--------------:|---------------:|
| AMASS Val mean ° | **15.12** | 15.12 | 15.15 |
| AMASS watch / phone | 18.55 / 11.68 | 18.57 / 11.67 | 18.58 / 11.71 |

AMASS 上 GT-slot 与 Week2 **对齐到 0.00°**（同一 79636 窗、同一冻结权重），说明本步协议与 Week2 dual 一致。Pred-seq 相对 GT-slot 只多 **0.02°**。

真机不可与 Week2 D5 的 26.03° 直接比绝对值：D5 是 40 序列子集，本步是 167 序列全量。公平上界是本表 GT-slot（25.2–27.4°）；Pred-seq 相对该上界约 **−0.2–0.3°**（lw_rp 上 Pred 略优于 GT-slot，来自序列投票）。

图：
- `outputs/figures/step2_rsb_main.png`：None / Pred-win / Pred-seq / GT-slot 分组柱
- `outputs/figures/step2_amass_watch_phone.png`：AMASS 表/机拆开
- `outputs/figures/step2_amass_stratified.png`：Joint OK vs BAD

### 第 2 步解读

1. **级联几乎吃满 Week2 上界。** AMASS Pred-seq 15.15° vs GT-slot 15.12° vs Week2 15.12°。第 1 步 Joint 0.90 已经够用：90% 窗上外参误差与「位置已知」完全相同（14.38°）；剩下 10% 窗 Pred 28.47 vs GT 22.05，槽位错了大约再罚 **6.4°**。
2. **序列投票略优于逐窗。** AMASS 15.75→15.15°；真机每组合再降约 0.8–1.4°。佩戴**槽位**整段恒定，多数投票抹掉第 1 步的窗级抖动；这与「\(R_{BS}\) 窗内才恒定」不矛盾。
3. **相对 None（≈43°）收益很大。** 合成从 43° 降到 ~15°；真机降到 25–28°。不估外参等于把 \(\pm 45^\circ\) 的 \(R_{BS}\) 吃进后续姿态。
4. **真机瓶颈仍是外参网本身，不是位置错。** Joint BAD 时 Pred 相对 GT-slot 只再差约 2–6°；而 GT-slot 已经在 25–30°（域差）。RW+LP 的级联惩罚最大（~5.6°）。
5. **腕难袋易沿用 Week2。** AMASS Pred-seq watch 18.58° / phone 11.71°，与 Week2 dual 的 18.55 / 11.68 同形态。

### 简要结论
- 未知朝向下：**位置 Pred → 冻结 Week2 dual** 可以把 \(R_{SB}\) 从 ~43° 拉到合成 **15.15°**、真机 **25–28°**，与位置已知上界对齐。
- 不必为第 2 步重训外参网；第 1 步 Joint ~0.90 时，槽位误差对外参的边际伤害很小（AMASS +0.02°）。
- 本步 \(\hat{R}_{SB}\) 供第 3 步 \(R_{MB}=R_{MS}\hat{R}_{SB}^{\top}\) 接 MobilePoser。

## 第 3 步：校准后接 MobilePoser 姿态

公式与 Week2 姿态下游相同：

\[
R_{MB} = R_{MS}\,\hat{R}_{SB}^{\top},\quad a_M = a_{\mathrm{obs}}
\]

冻结第 1 步位置网 + 第 2 步 Week2 dual，把校准后的 \(R_{MB},a_M\) 送入官方 MobilePoser（`checkpoints/weights.pth`）。注入/校准仍在解剖真值通道；**Pred-seq 按 90 帧段把真实表/机流散射进预测槽**（Head=4 不注入）。None / GT-slot / Oracle **仍走解剖 GT 通道**。默认 **12 序列 × 四组合**（与 Week2 姿态下游同一批序列，便于对照）。

对照（越低越好）：

| 条件 | 含义 | 进姿态网的槽 |
|------|------|-------------|
| **None** | 不校准，\(R_{MS}\to\) pose | 解剖 GT |
| **Pred-seq** | 第 1 步 **90 帧段内**多数投票槽位 → Week2 dual \(\hat{R}_{SB}\)（段内平均）→ 校准后把**真实表/机 IMU 填进预测槽** | **预测槽** |
| **GT-slot** | 真值位置喂 Week2（本协议上界 ≈ Week2 dual Learned） | 解剖 GT |
| **Oracle** | 完美 \(R_{SB}\) | 解剖 GT |

评测日期：2026-09-02（窗内恒定 / 分段注入 / **Pred-pack**）。W=90；AMASS / IMUPoser 各 12 序列、48 combo-run。  
序列级 Joint（段内多数投票槽位都对，再按 combo-run 平均）：AMASS **0.979**，IMUPoser **0.833**。

### 主表（四组合平均，Pred-pack）

| 数据 | None pos | **Pred-seq pos** | GT-slot pos | Oracle pos | None ang | **Pred-seq ang** | GT-slot ang | Oracle ang |
|------|--------:|-----------------:|-----------:|-----------:|---------:|-----------------:|-----------:|-----------:|
| AMASS | 15.38 | **13.92** | 13.80 | 13.12 | 30.78 | **27.60** | 27.36 | 25.78 |
| IMUPoser | 11.33 | **7.47** | 6.95 | 5.46 | 21.92 | **16.50** | 15.27 | 12.20 |

单位：位置 cm / 角度 °。SIP / mesh 同序：AMASS None 34.5° / 18.8 cm → Pred 31.0° / 17.1 cm；IMUPoser None 23.8° / 13.7 cm → Pred 17.3° / 9.3 cm。

None / GT-slot / Oracle 与 2026-08-16 GT 打包评测逐格相同（打包通道没变）。**只有 Pred-seq 列是新口径。**

### IMUPoser 分组合位置 cm（Pred-pack）

| combo | None | Pred-seq | GT-slot | Oracle |
|-------|-----:|---------:|--------:|-------:|
| lw_lp_h | 11.68 | **7.09** | 6.89 | 5.42 |
| lw_rp_h（官方训配） | 10.15 | **7.07** | 6.92 | 5.33 |
| rw_lp_h | 12.46 | **8.43** | 7.44 | 5.55 |
| rw_rp_h | 11.03 | **7.28** | 6.53 | 5.54 |

AMASS 四分位 Pred 挤在 13.8–14.1 cm（None 15.2–15.7）。官方组合 `lw_rp_h` 上 Pred 与 GT-slot 只差 **0.15 cm**；RW 组合差距更大（`rw_lp_h` 8.43 vs 7.44，`rw_rp_h` 7.28 vs 6.53），和官方权只训过左腕+右袋一致。

### Pred-seq：序列 Joint OK / BAD

打包按 90 帧段，分层按**整段** Joint（窗口多数投票）。不是逐段误差。

| 数据 | Pred Joint-OK | Pred Joint-BAD |
|------|-------------:|---------------:|
| AMASS | 13.96 cm（n=47） | 11.92 cm（n=1，不可读） |
| IMUPoser | 7.52 cm（n=40） | 7.18 cm（n=8） |

真机 BAD 没有崩掉，也不能据此说「填反也不痛」：n=8 很小；序列级 Joint 不等于段级打包错误率。看平均：Pred-pack 相对 GT-slot 多 **0.52 cm**，相对下面作废的 GT 打包 Pred（7.21 cm）多 **0.26 cm**。

### 作废对照：2026-08-16 GT 打包 Pred-seq

当时 Pred-seq 校准写在真值通道、姿态网按解剖槽读取，**从未看到填反**。真机 Pred **7.21 cm**、`lw_rp_h` 6.95 cm 属于该口径，**不要写进 Pred-pack 主表或和新 7.47 cm 平均**。

| 数据 | None | Pred-seq（GT 打包，作废） | GT-slot | Oracle |
|------|-----:|-------------------------:|--------:|-------:|
| AMASS | 15.38 | 13.89 | 13.80 | 13.12 |
| IMUPoser | 11.33 | **7.21** | 6.95 | 5.46 |

### 与 Week2 dual 姿态下游对照

同一 12 序列、同一官方权重。Week2 位置已知；本步位置由第 1 步 Pred。Oracle 应对齐（完美 \(R_{SB}\) 与注入无关）。Pred-seq 列为 **Pred-pack**。

| | Week2 dual Learned | Week3 GT-slot | Week3 Pred-seq | Week2/3 Oracle |
|--|-------------------:|--------------:|---------------:|---------------:|
| AMASS pos cm | **13.53** | 13.80 | 13.92 | 13.12 |
| AMASS ang ° | 26.80 | 27.36 | 27.60 | 25.78 |
| IMUPoser pos cm | **6.87** | 6.95 | 7.47 | 5.46 |
| IMUPoser ang ° | 15.02 | 15.27 | 16.50 | 12.20 |

None 与 Learned 因 \(R_{BS}\) 采样（整段 vs 按窗）不同，不能逐窗对齐；Oracle 两边都是 13.12 / 5.46 cm，说明姿态主网与序列集合一致。

图：
- `outputs/figures/step3_pose_pos.png` / `step3_pose_ang.png`：None / Pred-seq / GT-slot / Oracle
- `outputs/figures/step3_imuposer_pos_combo.png`：真机分组合
- `outputs/figures/step3_mesh_seq013_lw_rp.png` / `step3_mesh_seq008_lw_rp.png`：GT vs \(R_{MS}\to\) pose vs 级联（与第 1 步同一对好/坏序列）
- 三栏视频：`outputs/figures/step3_pyrender/step3_pyrender_seq013_lw_rp_side.mp4`

### 第 3 步解读

1. **未知朝向闭环成立。** None > Pred-seq ≥ GT-slot ≥ Oracle。合成 15.4→13.9 cm（Oracle 13.1）；真机 **11.3→7.5 cm**（Oracle 5.5），角度 21.9°→16.5°。
2. **填进预测槽之后，级联仍接近「位置已知」。** AMASS Pred 与 GT-slot 差 **0.12 cm**（序列 Joint 0.98）。真机差 **0.52 cm**（序列 Joint 0.83）；官方组合 `lw_rp_h` 上 Pred 7.07 vs GT-slot 6.92。旧 GT 打包把这个差压成 0.27 cm / 0.3 mm，那是评测漏洞，不是鲁棒性。
3. **真机收益远大于合成。** acc 未拧，None 在合成上本就不差；真机未校准的 \(R_{MS}\) 会把 \(\pm 45^\circ\) 外参送进姿态网。
4. **四组合都能用，RW 更吃亏。** 官方训配 `lw_rp_h` Pred 7.07 cm；`rw_lp_h` 8.43 cm。解读仍以 `lw_rp_h` 为主。
5. **瓶颈在姿态网与真机域，不在第 1 步。** 真机 GT-slot 已在 6.95 cm，离 Oracle 5.5 cm 是外参网域差。相对 Week2 Learned（6.87 cm），未知位置 + Pred-pack 大约多 **0.6 cm**。

### 简要结论
- 位置 Pred → \(R_{SB}\) → \(R_{MB}\) → 把真实表/机填进预测槽 → MobilePoser **可以接上**：真机从 **11.3 cm 降到 7.5 cm**，仍接近位置已知的 Week2 Learned（6.9 cm）与 Oracle（5.5 cm）。
- 旧真机 Pred **7.21 cm 是 GT 打包，作废**；不能和新 7.47 cm 混用。
- 不必为姿态这一步重训任何网络。未知朝向的主要伤害已经被第 2 步的 \(\hat{R}_{SB}\) 吃掉；槽位填反的额外代价目前是亚厘米。
- \(R_{MS}\to\) pose 基线（None）在真机上明显更差，可视化见 `step3_mesh_seq*` / pyrender 视频。

## 已知注意点
- **口径：** 现行是 \(R_{BS}\) 窗内恒定，XYZ 欧拉**每个轴均匀 \([0^\circ, 180^\circ]\)**。旧 \(\pm 45^\circ\) 表（Joint、7.47 cm 等）不要和新范围混用。旧整段协议数字（AMASS Joint 0.832、真机姿态 11.6→6.6 cm 等）也不要写进本周汇报。
- IMUPoser 官方数据是 5 路全开；测试时把录制 ori 当作 \(R_{MB}\) 再乘随机 \(R_{BS}\)。
- 与 Week1 对照时，输入模态也不相同（gyro vs \(R_{MS}\)），AMASS Val 是更干净的「只改朝向协议」对照。
- 第 2 步外参网与位置网 **各自归一化**：勿把 Week3 `norm_stats.pt` 喂给 Week2 dual。
- Week2 D5 外参 26.03° 是 40 序列子集；第 1/2 步 IMUPoser 为 167 序列；**第 3 步姿态默认 12 序列**（与 Week2 姿态下游对齐）。
- 第 2 步 Pred-seq 按 (序列, 组合) 投票；第 3 步 Pred-seq 按非重叠 90 帧段投票并段内平均 \(\hat{R}_{SB}\)，因为 \(R_{BS}\) 只在窗/段内恒定。
- **Pred-seq 进姿态网用预测槽位**（真值 IMU 按 90 帧段散射到 Pred 通道）。旧表（真机 7.21 cm 等）是按解剖 GT 通道打包的，不能与新 Pred-seq 混用；None / GT-slot / Oracle 仍走 GT 通道。
- 第 3 步需要官方 `checkpoints/weights.pth`。官方权只训过 `lw_rp_h`，主结论用该组合。
- `outputs/` 下 checkpoint / metrics JSON / 图视为可复现产物，默认不提交；以本 README 结果表为对外记录。
