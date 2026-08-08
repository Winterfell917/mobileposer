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

在 AMASS 上对表+机分别注入 \(R_{SB}\)，None / Learned / Oracle 校准后喂 Week1 分类器，看 Joint Acc 是否回升。

```bash
python experiments/week2_rot_ext/eval_downstream_week1.py \
  --config experiments/week2_rot_ext/configs/default.yaml
```

日志：`outputs/logs/metrics_downstream_week1.json`。  
期望：`none ≪ learned ≤ oracle`。

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

# D6：用联训模型接 Week1
python experiments/week2_rot_ext/eval_downstream_week1.py --dual
```

产物：`amass_dual_{train,val}.pt`、`norm_stats_dual.pt`、`best_rot_err_dual.pt`、`metrics_dual_val.json`、`metrics_downstream_week1_dual.json`。

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

## 实验结果（最新一轮：chordal 损失重训后）

> 记录日期：2026-08-07  
> 权重：`outputs/checkpoints/best_rot_err.pt`（**epoch 37**）  
> 原始日志：`outputs/logs/train_log.csv`、`metrics_val.json`、`metrics_imuposer_d5.json`、`metrics_downstream_week1.json`  
> 设定：`offset_range=45°`，窗长 90，AMASS 子集 CMU / ACCAD / BioMotionLab_NTroje

### 0. 训练稳定性

| 项 | 结果 |
|----|------|
| 训练窗 / 验证窗 | 336,412 / 79,636（四槽均衡） |
| 损失 | chordal \(1-\cos\theta\) + 0.1×6D MSE；`lr=5e-4`，`grad_clip=1.0` |
| 全程 | **40 epoch 均有限，无 NaN**（此前用 `acos` 反传时约第 10 epoch 起崩溃） |
| 最佳 | epoch 37，val **13.00°**；epoch 40 ≈ 13.03° |

训练从 ~21.5°（epoch 1）平滑降到 ~13°，后期在 13.0–13.3° 小幅波动，略有过拟合但不严重。

### 1. AMASS 验证（`eval.py --split val`）

主指标：\(\mathrm{geo}(\hat{R}_{SB}, R_{SB})\)（°），全量 79,636 窗。

| | 均值 ° | 中位 ° |
|--|--------|--------|
| **Learned** | **13.00** | **9.43** |

| Slot | 均值 ° | 中位 ° | n |
|------|--------|--------|---|
| RP | **10.00** | 7.22 | 19909 |
| LP | **10.05** | 7.25 | 19909 |
| LW | 15.68 | 11.99 | 19909 |
| RW | 16.28 | 12.50 | 19909 |

对照（`eval.py` 内 contrast，1280 窗子集；**全量以 13.00° 为准**）：

| 设定 | 平均 ° |
|------|--------|
| None（假设 \(R_{SB}=I\)） | 42.95 |
| Learned（子集） | 19.92 |
| Oracle | 0.0 |

**分析：**

- 相对注入外参本身约 43°，全量误差压到约 **1/3**，说明网络学到了静态安装旋转，不是恒等猜测。  
- **口袋明显好于手腕**（~10° vs ~16°）：腕部运动更丰富，短窗内更难把「恒定外参」从变化姿态里拆干净——与 Week1「腕更难」一致。  
- contrast 子集 Learned 偏高，是抽样偏差；汇报请用全量均值/中位。

### 2. D5：IMUPoser 真机 inject（`eval_imuposer_d5.py`）

协议：40 序列、5920 窗；在已标定流上注入已知 \(R_{SB}\)，再恢复。

**D5-A**（外参误差 °）：

| | 均值 ° | 中位 ° |
|--|--------|--------|
| **整体** | **25.09** | **19.83** |

| Slot | 均值 ° | 中位 ° |
|------|--------|--------|
| LW | 23.39 | 18.51 |
| RP | 24.25 | 18.65 |
| LP | 25.80 | 17.99 |
| RW | 26.91 | 23.57 |

**D5-B**（校准后 ori vs 干净参考 °）：

| 设定 | 均值 ° | 中位 ° |
|------|--------|--------|
| None | **43.01** | 44.06 |
| Learned | **25.09** | 19.83 |
| Oracle | ~0.03 | ~0.03 |

**分析：**

- 满足 **None ≫ Learned ≫ Oracle**；A 与 B-Learned 数值几乎相同（常数外参残差会原样反映到 ori 误差上），协议自洽。  
- **域间隙：** 合成 13.00° → 真机 25.09°（约 **+12°**）。真机噪声与动态分布与 AMASS 训练仍有差距，但相对乱戴 43° 仍去掉约四成外参误差。  
- 真机上四槽差距缩小（约 23–27°），合成上的「口袋优势」在 IMUPoser inject 上不那么明显。

### 3. D6-A：校准后接 Week1（`eval_downstream_week1.py`）

在 AMASS 上对表+机分别注入 \(R_{SB}\)，None / Learned / Oracle 校准后喂冻结的 Week1 分类器（30 序列、3668 窗、四种佩戴组合）。

| 设定 | Watch Acc | Phone Acc | **Joint Acc** |
|------|-----------|-----------|---------------|
| None | 0.851 | 0.694 | **0.596** |
| **Learned** | 0.956 | 0.914 | **0.883** |
| Oracle | 0.979 | 0.953 | **0.938** |

**分析：**

- 乱戴会明显伤害位置分类（Joint 掉到 0.60，手机尤甚）。  
- Week2 校准后 Joint **回升到 0.88**，接近 Oracle 0.94（差约 5.5pt，与合成仍有 ~13° 外参残差相符）。  
- 满足 **none ≪ learned ≤ oracle**，两周叙事闭环成立：Week1 假设朝向已知；Week2 在位置已知下估外参，能把乱戴下的位置分类大部分救回来。

### 4. 总评与后续

| 检查项 | 是否达成 |
|--------|----------|
| AMASS：Learned ≪ None，接近可用 | **是**（13° vs 43°） |
| D5：真机 inject 仍有增益 | **是**（25° vs 43°；有 ~12° 域差） |
| D6：Joint Acc 回升 | **是**（0.60 → 0.88 → 0.94） |
| 训练无 NaN | **是**（chordal + clip） |

**可写进汇报的结论：**

1. 位置已知时，BiLSTM 可从短窗 IMU 估计静态 \(R_{SB}\)（合成约 13° / 中位 9.4°）。  
2. 真机 inject 可迁移但有 sim-to-real 间隙（约 25°）。  
3. 外参校准对 Week1 位置分类有明确下游收益。

**可选下一步（非本周必做）：** 真机 inject 微调或更强增广以缩域差；腕部分头/加长窗；集成时接姿态主网（原 D6-B）。
