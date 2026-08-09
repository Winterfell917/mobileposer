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
a_{\mathrm{obs}} = R_{SB}^{\top} a_{\mathrm{bone}}
\]

校准：\(\hat{R}_{\mathrm{cal}} = R_{\mathrm{obs}}\, \hat{R}_{SB}^{\top}\)，\(a_{\mathrm{cal}} = R_{SB}\, a_{\mathrm{obs}}\)。

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

# 合成评估
python experiments/week2_rot_ext/eval_dual.py --split val

# D5：IMUPoser 真机 inject（表+机同时注入，联训双头恢复）
python experiments/week2_rot_ext/eval_imuposer_d5_dual.py \
  --config experiments/week2_rot_ext/configs/default.yaml

# D6：用联训模型接 Week1（AMASS + IMUPoser 真机 inject）
python experiments/week2_rot_ext/eval_downstream_week1.py --dual
```

产物：`amass_dual_{train,val}.pt`、`norm_stats_dual.pt`、`best_rot_err_dual.pt`、`metrics_dual_val.json`、`metrics_imuposer_d5_dual.json`、`metrics_downstream_week1_dual.json`（含 amass / imuposer）。

## D5 说明（为何 A、B 能一起做）

| | 内容 |
|--|------|
| **A** | 有注入的 \(R_{SB}\) GT → 直接报外参角度误差 |
| **B** | 同一批窗上，再报「校准后朝向是否回到干净参考」 |

二者共用同一套 inject 协议，**不冲突**。  
若将来有真实标定 \(R_{SB}\) 文件，可再加一条「无注入、直接比标定」的评测；当前数据条件下 inject-GT 是可复现的 D5-A。

## D7

本周**不做**显式 T/N-pose 基线；对照为 None / Oracle / Learned。

---

## 实验结果

> 记录日期：2026-08-09（单设备主结果 08-07；双设备联训与 D6-IMUPoser 补齐后汇总）  
> 设定：`offset_range=45°`，窗长 90，AMASS 子集 CMU / ACCAD / BioMotionLab_NTroje  
> 权重：单设备 `best_rot_err.pt`（epoch 37）；双设备 `best_rot_err_dual.pt`（epoch 37）  
> 日志：`train_log.csv` / `train_log_dual.csv`、`metrics_val.json` / `metrics_dual_val.json`、`metrics_imuposer_d5.json` / `metrics_imuposer_d5_dual.json`、`metrics_downstream_week1.json` / `metrics_downstream_week1_dual.json`  
> **综合汇总：** `python experiments/week2_rot_ext/summarize_metrics.py` → `outputs/logs/metrics_summary.json`

### 综合主表（None / Learned / Oracle）

把分散的 A1–A3 / B1–B3 收成一张总表。外参行单位为 °（越低越好）；D6 行为 Week1 Joint Acc（越高越好）。

| Track | 评测 | None | **Learned** | Oracle |
|-------|------|-----:|------------:|-------:|
| 单设备 | AMASS 外参 ° | 42.95 | **13.00** | 0 |
| 双设备 | AMASS 外参 ° | 42.90 | **12.65** | 0 |
| 单设备 | D5 真机外参 ° | 43.01 | **25.09** | ~0.03 |
| 双设备 | D5 真机外参 ° | 43.08 | **21.46** | ~0.03 |
| 单设备 | D6 AMASS Joint | 0.596 | **0.883** | 0.938 |
| 双设备 | D6 AMASS Joint | 0.596 | **0.869** | 0.938 |
| 单设备 | D6 IMUPoser Joint | 0.505 | **0.664** | 0.682 |
| 双设备 | D6 IMUPoser Joint | 0.505 | **0.682** | 0.682 |

**一眼结论：** 外参任务两条线都成立（None ≫ Learned ≫ Oracle）；联训在合成/真机外参与真机 D6 上更好；合成 D6 Joint 仍略偏向单设备×2。下文 A/B 为分项明细。

---

### A. 单设备（`RotExtrinsicNet`，一次估一台）

#### A0. 训练稳定性

| 项 | 结果 |
|----|------|
| 训练窗 / 验证窗 | 336,412 / 79,636（四槽均衡） |
| 损失 | chordal \(1-\cos\theta\) + 0.1×6D MSE；`lr=5e-4`，`grad_clip=1.0` |
| 全程 | **40 epoch 均有限，无 NaN**（此前 `acos` 反传约第 10 epoch 起崩溃） |
| 最佳 | epoch 37，val **13.00°**；epoch 40 ≈ 13.03° |

训练从 ~21.5°（epoch 1）平滑降到 ~13°，后期在 13.0–13.3° 小幅波动。

#### A1. AMASS 合成评估（`eval.py --split val`）

全量 79,636 窗；主指标 \(\mathrm{geo}(\hat{R}_{SB}, R_{SB})\)（°）。

| | 均值 ° | 中位 ° |
|--|--------|--------|
| **Learned** | **13.00** | **9.43** |

| Slot | 均值 ° | 中位 ° | n |
|------|--------|--------|---|
| RP | **10.00** | 7.22 | 19909 |
| LP | **10.05** | 7.25 | 19909 |
| LW | 15.68 | 11.99 | 19909 |
| RW | 16.28 | 12.50 | 19909 |

Contrast（1280 窗子集；**全量以 13.00° 为准**）：None 42.95° / Learned（子集）19.92° / Oracle 0°。

**要点：** 相对乱戴 ~43° 压到约 1/3；**口袋 (~10°) 明显好于手腕 (~16°)**。

#### A2. D5：IMUPoser 真机 inject（`eval_imuposer_d5.py`）

40 序列、5920 窗；录制流当干净参考，注入已知 \(R_{SB}\) 再恢复。

**D5-A**（外参误差 °）：整体 **25.09**（中位 19.83）。

| Slot | 均值 ° | 中位 ° |
|------|--------|--------|
| LW | 23.39 | 18.51 |
| RP | 24.25 | 18.65 |
| LP | 25.80 | 17.99 |
| RW | 26.91 | 23.57 |

**D5-B**（校准后 ori °）：None **43.01** → Learned **25.09** → Oracle ~0.03。

**要点：** None ≫ Learned ≫ Oracle；合成→真机域差约 **+12°**（13→25）；真机上四槽差距缩小。

#### A3. D6：校准后接 Week1（`eval_downstream_week1.py`）

同一命令含 AMASS + IMUPoser；单设备对表/机各跑一次 Week2（`single_device_x2`）。

**AMASS**（30 序列、3668 窗）：

| 设定 | Watch Acc | Phone Acc | **Joint Acc** |
|------|-----------|-----------|---------------|
| None | 0.851 | 0.694 | **0.596** |
| **Learned** | 0.956 | 0.914 | **0.883** |
| Oracle | 0.979 | 0.953 | **0.938** |

**IMUPoser**（40 序列、5920 窗；inject 协议同 D5）：

| 设定 | Watch Acc | Phone Acc | **Joint Acc** |
|------|-----------|-----------|---------------|
| None | 0.790 | 0.635 | **0.505** |
| **Learned** | 0.920 | 0.719 | **0.664** |
| Oracle | 0.939 | 0.725 | **0.682** |

**要点：** 合成域 Joint 0.60→0.88→0.94，闭环清晰。真机域仍满足 none ≪ learned ≤ oracle，但 Oracle 上限仅 ~0.68——即使用完美外参，Week1 在真机上仍有明显域差；手机侧是主要短板。

---

### B. 双设备联训（`RotExtrinsicDualNet`，一次估表+机）

#### B0. 训练稳定性

| 项 | 结果 |
|----|------|
| 损失 | chordal(watch)+chordal(phone)+0.1×6D；同 lr / clip |
| 全程 | **40 epoch 无 NaN，`skipped=0`** |
| 最佳 | epoch 37，val mean **12.65°**（watch 14.45° / phone 10.86°） |

#### B1. AMASS 合成评估（`eval_dual.py --split val`）

全量 79,636 窗（四组合各 19,909）。

| | 均值 ° | 中位 ° |
|--|--------|--------|
| watch | **14.45** | 11.45 |
| phone | **10.86** | 8.09 |
| **mean** | **12.65** | — |

None ≈ 42.9°（表/机）；Oracle 0°。四组合 mean 约 12.45–12.88°，带 RW 略差。

**要点：** mean 略优于单设备 13.00°；腕侧改善（相对单设备腕 ~16°→14.45°），口袋略逊（~10°→10.86°）；「腕难袋易」仍在，差距收窄。

#### B2. D5：IMUPoser 真机 inject（`eval_imuposer_d5_dual.py`）

40 序列 × 四组合；表+机同时注入、双头一次恢复。

**D5-A**（°）：

| | 均值 ° | 中位 ° |
|--|--------|--------|
| watch | 21.55 | 19.01 |
| phone | 21.36 | 16.82 |
| **mean** | **21.46** | **18.57** |

分槽：LW 19.66 / LP 20.57 / RP 22.15 / RW 23.44。  
组合 mean：LW+LP 20.02（最好）… RW+RP 22.88（最差）。

**D5-B** pooled：None **43.08** → Learned **21.46** → Oracle ~0.03。

**要点：** 真机 mean **优于单设备 D5 约 3.6°**（21.46 vs 25.09）；相对合成 12.65° 域差约 **+8.8°**（小于单设备的 +12°）。

#### B3. D6：校准后接 Week1（`eval_downstream_week1.py --dual`）

**AMASS：**

| 设定 | Watch Acc | Phone Acc | **Joint Acc** |
|------|-----------|-----------|---------------|
| None | 0.851 | 0.694 | **0.596** |
| **Learned** | 0.957 | 0.897 | **0.869** |
| Oracle | 0.979 | 0.953 | **0.938** |

**IMUPoser：**

| 设定 | Watch Acc | Phone Acc | **Joint Acc** |
|------|-----------|-----------|---------------|
| None | 0.790 | 0.635 | **0.505** |
| **Learned** | 0.937 | 0.729 | **0.682** |
| Oracle | 0.939 | 0.725 | **0.682** |

**要点：** 合成 Joint 0.60→0.87→0.94（相对单设备 Learned 低 1.4pt，主因 phone）。真机上 Learned≈Oracle（0.682），联训 Joint **略好于**单设备 0.664。

---

### C. 单设备 vs 双设备对照

| 评测 | 单设备 | 双设备联训 | 谁更好 |
|------|--------|------------|--------|
| AMASS 外参 mean ° | 13.00 | **12.65** | 联训 |
| D5 真机外参 mean ° | 25.09 | **21.46** | 联训（−3.6°） |
| 合成→真机域差 ° | +12.1 | **+8.8** | 联训更小 |
| D6 AMASS Joint（Learned） | **0.883** | 0.869 | 单设备×2（+1.4pt） |
| D6 IMUPoser Joint（Learned） | 0.664 | **0.682** | 联训（+1.8pt） |
| D6 IMUPoser Oracle 上限 | 0.682 | 0.682 | 同（Week1 真机上限） |

---

### D. 综合分析

1. **外参主任务成立（两条线都成立）**  
   合成 Learned ≪ None（~13° / ~12.7° vs ~43°）；真机 inject 仍有增益且协议自洽（A≈B-Learned，Oracle≈0）。

2. **联训在「估外参」上整体更强，尤其真机**  
   合成略优、真机明显优；共享编码器可能帮助两侧互相提供运动上下文，缩小 sim-to-real 间隙。腕侧是合成上的主要受益者。

3. **下游 D6 并不单调跟随外参 mean**  
   合成域单设备×2 Joint 更高：联训损失优化的是双侧 chordal，不是 Week1 Joint；且联训 phone 外参略差，而 Week1 对手机更敏感。  
   真机域则联训 Joint 反超，并顶到 Oracle——此时瓶颈已从 Week2 外参转到 **Week1 真机分类上限**。

4. **真机下游的天花板不在外参**  
   IMUPoser 上即使 Oracle 校准，Joint 仅 ~0.68（AMASS Oracle 0.94）。说明录制分布 / 特征与 Week1 训练域仍有差距；继续抠 Week2 角度对真机 Joint 的边际收益有限，除非同时做 Week1 真机适配。

5. **槽位规律**  
   合成：袋易腕难。真机 D5：四槽接近，RW 仍偏难。D6：phone_acc 始终低于 watch_acc。

---

### E. 结论与检查清单

| 检查项 | 单设备 | 双设备联训 |
|--------|--------|------------|
| 训练无 NaN | **是** | **是** |
| AMASS：Learned ≪ None | **是**（13° vs 43°） | **是**（12.7° vs 43°） |
| D5：真机 inject 有增益 | **是**（25° vs 43°） | **是**（21.5° vs 43°） |
| D6 AMASS：Joint 回升 | **是**（0.60→0.88→0.94） | **是**（0.60→0.87→0.94） |
| D6 IMUPoser：Joint 回升 | **是**（0.50→0.66→0.68） | **是**（0.50→0.68→0.68） |

**可写进汇报的结论：**

1. 位置已知时，BiLSTM 可从短窗 IMU 估计静态 \(R_{SB}\)：合成约 **13°**（单）/ **12.7°**（联训）；真机 inject 约 **25°** / **21.5°**。  
2. **双设备联训**在外参精度与真机 D5 上优于单设备，并缩小域差；合成 D6 Joint 略逊 1.4pt，真机 D6 则略优并接近 Oracle。  
3. 外参校准对 Week1 有明确收益；真机下游上限受 Week1 域差约束（Oracle Joint ~0.68），不全是 Week2 问题。  
4. 部署取舍：要一次前向、偏真机外参 → 联训；要抠合成 Joint → 单设备×2 仍可作对照主表。

**可选下一步（非本周必做）：** 联训 phone 损失加权以抬合成 D6；真机微调 / 增广缩 D5 域差；Week1 真机适配抬 Oracle 上限；腕部分头或加长窗；集成接姿态主网（原 D6-B）。
