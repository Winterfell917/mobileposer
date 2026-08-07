# Week2 — Device Mounting Extrinsic \(R_{SB}\) Estimation

> 目标：在**佩戴位置已知**时，估计设备相对对应骨骼的静态安装旋转 \(R_{SB}\)。

## 已锁定决策

| 项 | 选择 | 含义 |
|----|------|------|
| D1 | **A** | MVP：单设备（每次一个 slot：LW/RW/LP/RP） |
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
  models/rot_extrinsic.py
  train.py
  eval.py                       # AMASS val 角度误差 + None/Oracle/Learned
  eval_imuposer_d5.py           # D5 A+B
  eval_downstream_week1.py      # D6-A
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

## D5 说明（为何 A、B 能一起做）

| | 内容 |
|--|------|
| **A** | 有注入的 \(R_{SB}\) GT → 直接报外参角度误差 |
| **B** | 同一批窗上，再报「校准后朝向是否回到干净参考」 |

二者共用同一套 inject 协议，**不冲突**。  
若将来有真实标定 \(R_{SB}\) 文件，可再加一条「无注入、直接比标定」的评测；当前数据条件下 inject-GT 是可复现的 D5-A。

## D7

本周**不做**显式 T/N-pose 基线；对照为 None / Oracle / Learned。

## 成功标准

1. AMASS val：Learned 外参误差明显低于 None（假设 \(I\)），靠近 Oracle。  
2. D5：A 的 \(R_{SB}\) 误差合理；B 上 `none ≫ learned ≈ oracle`（ori °）。  
3. D6：Week1 Joint Acc 上 `none ≪ learned ≤ oracle`。
