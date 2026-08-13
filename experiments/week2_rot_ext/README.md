# Week2 — Device Mounting Extrinsic \(R_{SB}\) Estimation

> 目标：在**佩戴位置已知**时，估计设备相对对应骨骼的静态安装旋转 \(R_{SB}\)。

## 已锁定决策

| 项 | 选择 | 含义 |
|----|------|------|
| D1 | **A + 双设备联训** | 主线仍保留单设备；另提供 **表+机联合训练**（`*_dual`） |
| D2 | **A** | 网络显式回归 \(R_{SB}\)（6D → SO(3)） |
| D3 | **A** | 输入 `acc(3)+ori_flat(9)=12` |
| D4 | **A** | BiLSTM 编码器（对齐 Week1 风格） |
| D5 | **A+B** | 真机：注入已知 \(R_{SB}\) 报角度误差 **且** 报校准后 ori proxy |
| D6 | **仅 A** | 下游只做：校准后接 Week1 Joint Acc |
| D7 | **B** | 本周不做显式 T/N-pose 基线（表内 None/Oracle/Learned） |

## 与 Week1 / 全系统

```text
Week1：外参已知（或忽略） → 判左右位置
Week2：位置已知（slot GT） → 估 R_SB
之后 ：位置 + R_SB → 干净 IMU → 姿态主网
```

物理直觉：人体动作在变，安装外参近似不变；模型从变化的运动信号里分离出恒定的安装旋转。

**和 TIC 的区别：** TIC 常同时回归 drift+offset。本周主任务只做**静态** \(R_{SB}\)（合成关闭 drift）。

## 旋转约定

\[
R_{\mathrm{obs}} = R_{\mathrm{bone}}\, R_{SB},\quad
a_{\mathrm{obs}} = a_{\mathrm{bone}}
\]

注入外参时**只右乘朝向，加速度保持不变**。  
校准：\(\hat{R}_{\mathrm{cal}} = R_{\mathrm{obs}}\, \hat{R}_{SB}^{\top}\)，\(a_{\mathrm{cal}} = a_{\mathrm{obs}}\)。

## 目录

```text
experiments/week2_rot_ext/
  README.md
  configs/default.yaml
  dataset/
    build_amass_rot_ext.py
    build_imuposer_rot_ext.py   # 可选导出；D5 主评在 eval_imuposer_d5.py
    rot_dataset.py
  models/rot_extrinsic.py       # 单设备
  models/rot_extrinsic_dual.py  # 双设备联训（共享编码器 + 双头）
  train.py / eval.py            # 单设备
  train_dual.py / eval_dual.py  # 双设备
  eval_imuposer_d5.py           # D5 A+B（单设备）
  eval_imuposer_d5_dual.py      # D5 A+B（双设备联训）
  eval_downstream_week1.py      # D6-A（默认单设备×2；`--dual` 用联训模型）
  dataset/build_amass_rot_ext_dual.py
  outputs/{data,checkpoints,logs}
```

## 使用步骤（仓库根目录）

```bash
conda activate mobileposer
cd /home/caolindong/projects/mobileposer
```

### 1. 造 AMASS 训练/验证集

```bash
python experiments/week2_rot_ext/dataset/build_amass_rot_ext.py \
  --config experiments/week2_rot_ext/configs/default.yaml
```

### 2. 训练

```bash
python experiments/week2_rot_ext/train.py \
  --config experiments/week2_rot_ext/configs/default.yaml
```

最优：`outputs/checkpoints/best_rot_err.pt`。

**稳定性说明：** 训练损失用平滑的 \(1-\cos\theta\)（chordal），**不用** `acos` 反传（`acos` 在 \(\theta\to 0\) 时梯度爆炸，曾导致约第 10 epoch 起 NaN）。验证指标仍报测地线角度 °；并启用 `grad_clip_norm=1.0`、`lr=5e-4`。

### 3. 合成评估（主表）

```bash
python experiments/week2_rot_ext/eval.py --split val
```

### 4. D5：IMUPoser 上 A+B（需已训练权重）

`imuposer_full.pt` **没有**原生 \(R_{SB}\) 标签。协议：

1. 把录制流当作干净参考；
2. **注入已知**随机 \(R_{SB}\)（与训练同约定）；
3. 网络恢复外参。

- **D5-A**：\(\mathrm{geo}(\hat{R}_{SB}, R_{SB}^{\mathrm{gt}})\)（°）  
- **D5-B**：校准后 ori 相对干净参考的平均角度误差；对照 None / Learned / Oracle  

```bash
python experiments/week2_rot_ext/eval_imuposer_d5.py \
  --config experiments/week2_rot_ext/configs/default.yaml
```

日志：`outputs/logs/metrics_imuposer_d5.json`。

### 5. D6-A：校准后接 Week1（需 Week1 + Week2 权重）

对表+机分别注入 \(R_{SB}\)，None / Learned / Oracle 校准后喂 Week1 分类器，看 Joint Acc 是否回升。  
**同一命令会跑两套数据：**

| 数据 | 含义 |
|------|------|
| **AMASS** | 合成骨对齐 IMU（主表，与此前一致） |
| **IMUPoser** | 真机录制流当作干净参考，再 inject \(R_{SB}\)（与 D5 同协议），再接 Week1 |

```bash
# 单设备 Week2 ×2
python experiments/week2_rot_ext/eval_downstream_week1.py \
  --config experiments/week2_rot_ext/configs/default.yaml

# 仅 IMUPoser / 仅 AMASS（可选）
# python .../eval_downstream_week1.py --skip-amass
# python .../eval_downstream_week1.py --skip-imuposer
```

日志：`outputs/logs/metrics_downstream_week1.json`（内含 `datasets.amass` / `datasets.imuposer`）。  
期望：两边均为 `none ≪ learned ≤ oracle`。

### 6. 双设备联训（表 + 机一起估 \(R_{SB}\)）

与单设备并行的一条线：一次前向同时输出手表、手机两套外参。

| 项 | 设定 |
|----|------|
| 输入 | `[T,24]` = watch(acc+ori 12) + phone(12)；条件为 watch/phone 的 slot one-hot |
| 输出 | \(\hat{R}_{SB}^{\mathrm{watch}}\)、\(\hat{R}_{SB}^{\mathrm{phone}}\)（各 6D→SO(3)） |
| 结构 | 共享 BiLSTM 编码器 + `head_watch` / `head_phone` |
| 损失 | chordal(watch) + chordal(phone)（+ 两侧 6D MSE） |
| 数据 | 每窗枚举 4 种佩戴组合；两侧 **独立** 随机 \(R_{SB}\) |

```bash
# 造双设备数据（不覆盖单设备 amass_train.pt）
python experiments/week2_rot_ext/dataset/build_amass_rot_ext_dual.py \
  --config experiments/week2_rot_ext/configs/default.yaml

# 训练 → best_rot_err_dual.pt
python experiments/week2_rot_ext/train_dual.py \
  --config experiments/week2_rot_ext/configs/default.yaml
# 中断后续训：--resume experiments/week2_rot_ext/outputs/checkpoints/last_dual.pt

# 合成评估
python experiments/week2_rot_ext/eval_dual.py --split val

# D5：IMUPoser 真机 inject（表+机同时注入，联训双头恢复）
python experiments/week2_rot_ext/eval_imuposer_d5_dual.py \
  --config experiments/week2_rot_ext/configs/default.yaml

# D6：用联训模型接 Week1（AMASS + IMUPoser 真机 inject）
python experiments/week2_rot_ext/eval_downstream_week1.py --dual
```

产物：`amass_dual_{train,val}.pt`、`norm_stats_dual.pt`、`best_rot_err_dual.pt`、`metrics_dual_val.json`、`metrics_imuposer_d5_dual.json`、`metrics_downstream_week1_dual.json`（含 amass / imuposer）。

### 7. 接 MobilePoser 姿态（\(R_{MB}, a_M\)）

校准公式（与本周协议一致）：

\[
R_{MB} = R_{\mathrm{obs}}\,\hat{R}_{SB}^{\top},\quad a_M = a_{\mathrm{obs}}
\]

默认评测 **四组合** `(LW|RW)×(LP|RP)`，每组再加干净 Head，打包为 `[watch, phone, Head]`。Week2 估表/机外参；Head 不注入。对照 None / Learned / Oracle。

注意：官方 `weights.pth` 训练时 combo mask 是 `lw_rp_h`；另外三组是域偏移（通道仍合法，未用槽填零）。

需先把官方预训练权重放到 `checkpoints/weights.pth`。

```bash
python experiments/week2_rot_ext/eval_pose_downstream.py \
  --config experiments/week2_rot_ext/configs/default.yaml

python experiments/week2_rot_ext/eval_pose_downstream.py --dual
```

日志：`outputs/logs/metrics_pose_downstream.json`（`--dual` 则为 `*_dual.json`）。

## D5 说明（为何 A、B 能一起做）

| | 内容 |
|--|------|
| **A** | 有注入的 \(R_{SB}\) GT → 直接报外参角度误差 |
| **B** | 同一批窗上，再报「校准后朝向是否回到干净参考」 |

二者共用同一套 inject 协议，**不冲突**。  
若将来有真实标定 \(R_{SB}\) 文件，可再加一条「无注入、直接比标定」的评测；当前数据条件下 inject-GT 是可复现的 D5-A。

## D7

显式 T/N-pose 基线仍不做。姿态下游改为：Week2 校准后的 \(R_{MB}, a_M\) 接 MobilePoser（见上文第 7 步）。

---

## 实验结果

> 记录日期：2026-08-13（协议改为**只拧朝向、加速度不变**后全量重训重评；当晚补姿态下游）  
> 设定：`offset_range=45°`，窗长 90，AMASS 子集 CMU / ACCAD / BioMotionLab_NTroje  
> 权重：单设备 `best_rot_err.pt`（epoch 37）；双设备 `best_rot_err_dual.pt`（epoch 38）；姿态 `checkpoints/weights.pth`  
> 日志：`train_log.csv` / `train_log_dual.csv`、`metrics_val.json` / `metrics_dual_val.json`、`metrics_imuposer_d5.json` / `metrics_imuposer_d5_dual.json`、`metrics_downstream_week1.json` / `metrics_downstream_week1_dual.json`、`metrics_pose_downstream.json` / `metrics_pose_downstream_dual.json`  
> **综合汇总：** `python experiments/week2_rot_ext/summarize_metrics.py` → `outputs/logs/metrics_summary.json`

相对上一轮（acc 随 \(R_{SB}\) 一起旋转）：合成外参约 13° → **15°**，真机约 21–25° → **26–28°**。D6 的 None 也升高（acc 不再被拧乱，未校准时分类更容易）。

### 综合主表（None / Learned / Oracle）

把分散的 A1–A3 / B1–B3 收成一张总表。外参行单位为 °（越低越好）；D6 行为 Week1 Joint Acc（越高越好）。

| Track | 评测 | None | **Learned** | Oracle |
|-------|------|-----:|------------:|-------:|
| 单设备 | AMASS 外参 ° | 42.95 | **15.01** | 0 |
| 双设备 | AMASS 外参 ° | 42.90 | **15.12** | 0 |
| 单设备 | D5 真机外参 ° | 43.01 | **27.88** | ~0.03 |
| 双设备 | D5 真机外参 ° | 43.08 | **26.03** | ~0.03 |
| 单设备 | D6 AMASS Joint | 0.637 | **0.890** | 0.938 |
| 双设备 | D6 AMASS Joint | 0.637 | **0.875** | 0.938 |
| 单设备 | D6 IMUPoser Joint | 0.548 | **0.682** | 0.682 |
| 双设备 | D6 IMUPoser Joint | 0.548 | **0.694** | 0.682 |
| 单设备 | 姿态 AMASS Pos. cm | 15.44 | **13.55** | 13.12 |
| 双设备 | 姿态 AMASS Pos. cm | 15.44 | **13.53** | 13.12 |
| 单设备 | 姿态 IMUPoser Pos. cm | 11.97 | **6.90** | 5.46 |
| 双设备 | 姿态 IMUPoser Pos. cm | 11.97 | **6.87** | 5.46 |

**一眼结论：** 外参任务两条线仍成立（None ≫ Learned ≫ Oracle）；合成外参单设备略优，真机外参与真机 D6 联训更好；合成 D6 Joint 仍略偏向单设备×2。姿态下游（四组合平均）Learned 夹在 None 与 Oracle 之间，**真机位置误差约 12→7 cm**，收益明显大于合成。下文 A/B 为分项明细。

---

### A. 单设备（`RotExtrinsicNet`，一次估一台）

#### A0. 训练稳定性

| 项 | 结果 |
|----|------|
| 训练窗 / 验证窗 | 336,412 / 79,636（四槽均衡） |
| 损失 | chordal \(1-\cos\theta\) + 0.1×6D MSE；`lr=5e-4`，`grad_clip=1.0` |
| 全程 | **40 epoch 均有限，无 NaN** |
| 最佳 | epoch 37，val **15.01°**；epoch 40 ≈ 15.10° |

训练从 ~22.2°（epoch 1）平滑降到 ~15°，后期在 15.0–15.3° 小幅波动。

#### A1. AMASS 合成评估（`eval.py --split val`）

全量 79,636 窗；主指标 \(\mathrm{geo}(\hat{R}_{SB}, R_{SB})\)（°）。

| | 均值 ° | 中位 ° |
|--|--------|--------|
| **Learned** | **15.01** | **11.05** |

| Slot | 均值 ° | 中位 ° | n |
|------|--------|--------|---|
| RP | **10.49** | 7.43 | 19909 |
| LP | **10.59** | 7.64 | 19909 |
| LW | 19.36 | 15.62 | 19909 |
| RW | 19.60 | 16.17 | 19909 |

Contrast（1280 窗子集；**全量以 15.01° 为准**）：None 42.95° / Learned（子集）22.62° / Oracle 0°。

**要点：** 相对乱戴 ~43° 压到约 1/3；**口袋 (~10.5°) 明显好于手腕 (~19.5°)**。腕袋差距比「acc 也拧」时更大。

#### A2. D5：IMUPoser 真机 inject（`eval_imuposer_d5.py`）

40 序列、5920 窗；录制流当干净参考，注入已知 \(R_{SB}\) 再恢复。

**D5-A**（外参误差 °）：整体 **27.88**（中位 23.25）。

| Slot | 均值 ° | 中位 ° |
|------|--------|--------|
| LP | 26.01 | 17.92 |
| RP | 26.32 | 20.85 |
| LW | 27.43 | 22.62 |
| RW | 31.77 | 29.17 |

**D5-B**（校准后 ori °）：None **43.01** → Learned **27.88** → Oracle ~0.03。

**要点：** None ≫ Learned ≫ Oracle；合成→真机域差约 **+12.9°**（15.01→27.88）；真机上 RW 明显最差。

#### A3. D6：校准后接 Week1（`eval_downstream_week1.py`）

同一命令含 AMASS + IMUPoser；单设备对表/机各跑一次 Week2（`single_device_x2`）。

**AMASS**（30 序列、3668 窗）：

| 设定 | Watch Acc | Phone Acc | **Joint Acc** |
|------|-----------|-----------|---------------|
| None | 0.888 | 0.706 | **0.637** |
| **Learned** | 0.959 | 0.920 | **0.890** |
| Oracle | 0.979 | 0.953 | **0.938** |

**IMUPoser**（40 序列、5920 窗；inject 协议同 D5）：

| 设定 | Watch Acc | Phone Acc | **Joint Acc** |
|------|-----------|-----------|---------------|
| None | 0.826 | 0.661 | **0.548** |
| **Learned** | 0.941 | 0.723 | **0.682** |
| Oracle | 0.939 | 0.725 | **0.682** |

**要点：** 合成域 Joint 0.64→0.89→0.94，闭环清晰。真机域 Learned≈Oracle（0.682）——即使用完美外参，Week1 在真机上仍有明显域差；手机侧是主要短板。None 高于旧协议，是因为加速度不再被安装旋转拧乱。

---

### B. 双设备联训（`RotExtrinsicDualNet`，一次估表+机）

#### B0. 训练稳定性

| 项 | 结果 |
|----|------|
| 损失 | chordal(watch)+chordal(phone)+0.1×6D；同 lr / clip |
| 全程 | **40 epoch 无 NaN，`skipped=0`**（中断后 `--resume last_dual.pt` 从 epoch 30 续完） |
| 最佳 | epoch 38，val mean **15.12°**（watch 18.55° / phone 11.68°） |

#### B1. AMASS 合成评估（`eval_dual.py --split val`）

全量 79,636 窗（四组合各 19,909）。

| | 均值 ° | 中位 ° |
|--|--------|--------|
| watch | **18.55** | 15.27 |
| phone | **11.68** | 8.68 |
| **mean** | **15.12** | — |

None ≈ 42.9°（表/机）；Oracle 0°。四组合 mean 约 14.93–15.39°，带 RW 略差。

**要点：** mean 略逊于单设备 15.01°；腕侧相对单设备腕 ~19.5°→18.55° 略好，口袋略逊（~10.5°→11.68°）；「腕难袋易」仍在。

#### B2. D5：IMUPoser 真机 inject（`eval_imuposer_d5_dual.py`）

40 序列 × 四组合；表+机同时注入、双头一次恢复。

**D5-A**（°）：

| | 均值 ° | 中位 ° |
|--|--------|--------|
| watch | 29.57 | 27.26 |
| phone | 22.48 | 17.63 |
| **mean** | **26.03** | **23.59** |

分槽：LP 21.59 / RP 23.38 / LW 26.61 / RW 32.52。  
组合 mean：LW+RP 24.42（最好）… RW+RP 28.36（最差）。

**D5-B** pooled：None **43.08** → Learned **26.03** → Oracle ~0.03。

**要点：** 真机 mean **优于单设备 D5 约 1.9°**（26.03 vs 27.88）；相对合成 15.12° 域差约 **+10.9°**（小于单设备的 +12.9°）。

#### B3. D6：校准后接 Week1（`eval_downstream_week1.py --dual`）

**AMASS：**

| 设定 | Watch Acc | Phone Acc | **Joint Acc** |
|------|-----------|-----------|---------------|
| None | 0.888 | 0.706 | **0.637** |
| **Learned** | 0.966 | 0.895 | **0.875** |
| Oracle | 0.979 | 0.953 | **0.938** |

**IMUPoser：**

| 设定 | Watch Acc | Phone Acc | **Joint Acc** |
|------|-----------|-----------|---------------|
| None | 0.826 | 0.661 | **0.548** |
| **Learned** | 0.942 | 0.731 | **0.694** |
| Oracle | 0.939 | 0.725 | **0.682** |

**要点：** 合成 Joint 0.64→0.87→0.94（相对单设备 Learned 低 1.5pt，主因 phone）。真机上 Learned **略高于** Oracle（0.694 vs 0.682），联训 Joint 也好于单设备 0.682。

---

### C. 单设备 vs 双设备对照

| 评测 | 单设备 | 双设备联训 | 谁更好 |
|------|--------|------------|--------|
| AMASS 外参 mean ° | **15.01** | 15.12 | 单设备 |
| D5 真机外参 mean ° | 27.88 | **26.03** | 联训（−1.9°） |
| 合成→真机域差 ° | +12.9 | **+10.9** | 联训更小 |
| D6 AMASS Joint（Learned） | **0.890** | 0.875 | 单设备×2（+1.5pt） |
| D6 IMUPoser Joint（Learned） | 0.682 | **0.694** | 联训（+1.2pt） |
| D6 IMUPoser Oracle 上限 | 0.682 | 0.682 | 同（Week1 真机上限） |
| 姿态 AMASS Pos. cm（Learned） | 13.55 | **13.53** | 几乎打平 |
| 姿态 IMUPoser Pos. cm（Learned） | 6.90 | **6.87** | 几乎打平 |

---

### F. 姿态下游（`eval_pose_downstream.py`）

12 序列 × 四组合 `(LW|RW)×(LP|RP)` + 干净 Head；官方 `weights.pth`。单位：位置误差 cm / 角度误差 °（越低越好）。

**四组合平均（pooled）：**

| Track | 数据 | None pos | **Learned pos** | Oracle pos | None ang | **Learned ang** | Oracle ang |
|-------|------|--------:|----------------:|-----------:|---------:|----------------:|-----------:|
| 单设备 | AMASS | 15.44 | **13.55** | 13.12 | 31.15 | **26.69** | 25.78 |
| 双设备 | AMASS | 15.44 | **13.53** | 13.12 | 31.15 | **26.80** | 25.78 |
| 单设备 | IMUPoser | 11.97 | **6.90** | 5.46 | 22.58 | **15.32** | 12.20 |
| 双设备 | IMUPoser | 11.97 | **6.87** | 5.46 | 22.58 | **15.02** | 12.20 |

**单设备分组合位置 cm：**

| combo | AMASS None / Learned / Oracle | IMUPoser None / Learned / Oracle |
|-------|-------------------------------|----------------------------------|
| lw_lp_h | 16.69 / **13.56** / 13.05 | 11.34 / **6.94** / 5.42 |
| lw_rp_h（官方训配） | 14.54 / **13.64** / 13.25 | 11.62 / **6.90** / 5.33 |
| rw_lp_h | 15.21 / **13.44** / 13.09 | 13.75 / **6.40** / 5.55 |
| rw_rp_h | 15.31 / **13.56** / 13.10 | 11.16 / **7.37** / 5.54 |

**要点：** 校准后接 MobilePoser 闭环成立（None > Learned ≥ Oracle）。合成上 acc 未拧，None 本来就不差，只降约 2 cm；真机从 ~12 cm 降到 ~7 cm，接近 Oracle 5.5 cm。四组合接近；单/双几乎打平。`lw_rp_h` 与另外三组没有崩，但官方权重按该 combo 训练，解读时仍以 `lw_rp_h` 为主。

---

### D. 综合分析

1. **外参主任务成立（两条线都成立）**  
   合成 Learned ≪ None（~15.0° / ~15.1° vs ~43°）；真机 inject 仍有增益且协议自洽（A≈B-Learned，Oracle≈0）。只拧朝向后角度误差比拧 acc 时大约 2°，任务仍可解。

2. **联训在真机外参上仍更强，合成上不再领先**  
   合成 mean 单设备略优（15.01 vs 15.12）；真机联训更好、域差更小。腕难袋易在「acc 不变」后更明显。

3. **下游 D6 并不单调跟随外参 mean**  
   合成域单设备×2 Joint 更高：联训优化双侧 chordal，不是 Week1 Joint；联训 phone 外参更差，而 Week1 对手机更敏感。  
   真机域则联训 Joint 反超，并略高于 Oracle——此时瓶颈已从 Week2 外参转到 **Week1 真机分类上限**。

4. **真机下游的天花板不在外参**  
   IMUPoser 上即使 Oracle 校准，Joint 仅 ~0.68（AMASS Oracle 0.94）。继续抠 Week2 角度对真机 Joint 的边际收益有限，除非同时做 Week1 真机适配。

5. **槽位规律**  
   合成：袋易腕难（差距拉大）。真机 D5：RW 仍最差。D6：phone_acc 始终低于 watch_acc。None Joint 因 acc 未拧而高于旧协议。

6. **接 MobilePoser 的姿态闭环成立**  
   Learned 夹在 None 与 Oracle 之间。合成位置只降约 2 cm（acc 不变，None 本就不差）；真机 **12→7 cm**（Oracle 5.5 cm），角度 23°→15°。单/双几乎打平；四组合均可用，主表以官方 `lw_rp_h` 为准。

---

### E. 结论与检查清单

| 检查项 | 单设备 | 双设备联训 |
|--------|--------|------------|
| 训练无 NaN | **是** | **是** |
| AMASS：Learned ≪ None | **是**（15.0° vs 43°） | **是**（15.1° vs 43°） |
| D5：真机 inject 有增益 | **是**（27.9° vs 43°） | **是**（26.0° vs 43°） |
| D6 AMASS：Joint 回升 | **是**（0.64→0.89→0.94） | **是**（0.64→0.87→0.94） |
| D6 IMUPoser：Joint 回升 | **是**（0.55→0.68→0.68） | **是**（0.55→0.69→0.68） |
| 姿态 AMASS：Pos. 回升 | **是**（15.4→13.6→13.1） | **是**（15.4→13.5→13.1） |
| 姿态 IMUPoser：Pos. 回升 | **是**（12.0→6.9→5.5） | **是**（12.0→6.9→5.5） |

**可写进汇报的结论：**

1. 位置已知时，BiLSTM 可从短窗朝向估计静态 \(R_{SB}\)（加速度保持不变）：合成约 **15.0°**（单）/ **15.1°**（联训）；真机 inject 约 **27.9°** / **26.0°**。  
2. **双设备联训**在真机 D5 与真机 D6 上优于单设备，并缩小域差；合成外参与合成 D6 Joint 略逊于单设备×2。  
3. 外参校准对 Week1 有明确收益；真机位置分类上限受 Week1 域差约束（Oracle Joint ~0.68），不全是 Week2 问题。  
4. 校准后的 \(R_{MB}, a_M\) 可直接接 MobilePoser：真机位置误差 **12→7 cm**（Oracle 5.5 cm）；单/双姿态几乎打平。  
5. 部署取舍：要一次前向、偏真机外参 → 联训；要抠合成 Joint → 单设备×2 仍可作对照主表。

**可选下一步：** 联训 phone 损失加权以抬合成 D6；真机微调 / 增广缩 D5 域差；Week1 真机适配抬分类上限；无 GT 级联（Pred 位 → 外参 → 姿态）。
