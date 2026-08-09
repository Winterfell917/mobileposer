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

**两两组合，共四种，本实验都测试了** 
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

### 5. 可视化（序列时间线 + SMPL mesh 标注）
```bash
# 默认：自动挑好/坏序列，输出 timeline + mesh 拼图
python experiments/week1_pos_cls/visualize.py --split test --combo lw_rp

# 推荐示例：Walking 好序列 / Boxing 坏序列
python experiments/week1_pos_cls/visualize.py --split test --combo lw_rp --seq-ids 110,12

# 按 seq-id 顺序导出全部序列的 GT/Pred timeline（默认不渲染 mesh）
python experiments/week1_pos_cls/visualize.py --split test --combo lw_rp --all-sequences

# 指定序列并导出 GIF
python experiments/week1_pos_cls/visualize.py --split test --combo lw_rp \
  --seq-ids 110 --gif --max-mesh-frames 30
```
图在 `outputs/figures/`（`--all-sequences` 时在子目录 `timelines_all_test_{combo}/`）：
- `timeline_seq{ID:03d}_{good|mid|bad}_test_{combo}.png`：按序号排序；横轴 Frame，纵轴位置 0/1/2/3，GT 实线 / Pred 虚线
- `timeline_index_test_{combo}.json`：seq → 被试/动作名 / joint-acc 索引
- `mesh_seq{ID}_test_{combo}.png`：GT 姿态 mesh + 位置标注（需 `--mesh`；全量导出默认关闭）
- `cm_*.png`：混淆矩阵

说明：mesh 姿态来自 IMUPoser GT pose；颜色表示本周位置分类结果（不是姿态预测）。

### 6. Unity 实时推流（学长 Motion Viewer）

脚本：`stream_unity.py`。Python 做 TCP **服务端**，Unity 场景 **MotionViewerExample** 里 Hierarchy 的 **Online** 做客户端连上来收 SMPL `pose`/`tran`，并叠加手表/手机位置球。

| Unity（选中 Online） | 建议值 |
|----------------------|--------|
| Client → Server Ip | `127.0.0.1` |
| Client → Port | `8989`（须与脚本 `--port` 一致） |
| Client → Connect On Load | **必须勾选** |
| Client → Timeout Seconds | `60` |
| Set Motion Online → Body Prefab | `Character` |
| Hierarchy | 启用 **Online**，关掉 **Offline**；保留 **Online → Points** |

**推荐序列（combo=`lw_rp`）**

| 类型 | seq-id | 运动 | 窗级 joint-acc（约） |
|------|--------|------|----------------------|
| 好序列 | **110** | P6/Walking | 0.993 |
| 坏序列 | **12** | P1/Boxing | 0.000 |
| 备选好 | 102 | P6/ClappingFull | 1.000（单序列好，该类整体并不容易） |

**启动顺序（先 Python，再 Unity Play）**

1. 仓库根目录启动推流：
   ```bash
   # 单序列（默认循环）
   python experiments/week1_pos_cls/stream_unity.py --combo lw_rp --seq-id 110 \
     --marker-offset 0.15 --marker-radius 0.09

   # 一次连接后自动从第一个 seq 播到最后一个（每条播一遍后退出）
   python experiments/week1_pos_cls/stream_unity.py --combo lw_rp --all-seqs --auto-start \
     --marker-offset 0.15 --marker-radius 0.09

   # 只播一段区间，并循环整个播放列表
   python experiments/week1_pos_cls/stream_unity.py --combo lw_rp --all-seqs \
     --seq-start 0 --seq-end 20 --loop-playlist --auto-start
   ```
2. 终端出现 Waiting 后，到 Unity 点 **Play**。
3. 若未加 `--auto-start`：出现 `Unity connected` 后按 **Enter** 开始。
4. 若人体被裁切：Game 视图切到 **Front / Side Camera**。
5. 结束：终端 `Ctrl+C`，或先停 Unity Play。

**标注说明（默认开启）**

- 球画在**身体外侧**（关节中心 + `--marker-offset`），避免埋进 mesh。
- **绿大球**：Pred 正确（手表→手部外侧；手机→髋/口袋外侧）
- **红大球**：Pred 错误
- **小蓝 / 青球**：仅 Pred 错时标出 GT 位置
- 仍被挡住时加大：`--marker-offset 0.18 --marker-radius 0.10`；只要姿态可加 `--no-markers`

**常用参数**

| 参数 | 含义 |
|------|------|
| `--port 8989` | 监听端口（对齐 Unity Client） |
| `--fps 30` | 播放帧率（默认用 config 的 data.fps） |
| `--start` / `--end` | 裁剪帧区间 |
| `--once` | 单序列：只播一遍后退出（默认循环） |
| `--all-seqs` | 一次连接后按序播完所有（或 `--seq-start/--seq-end`）序列 |
| `--loop-playlist` | 与 `--all-seqs` 联用：播完列表后从头再来 |
| `--gap` | 序列之间停顿秒数（默认 0.5） |
| `--auto-start` | 连上立刻播，不等 Enter |
| `--no-markers` | 关闭位置球 |
| `--marker-offset` / `--marker-radius` | 球体外偏距离 / 半径（米） |
| `--no-tran` | 根平移置零（原地播） |

**排障**

- `Address already in use`：8989 仍被旧进程占用 → 终端 `Ctrl+C` 或杀掉旧 `stream_unity` 再开。
- 一直 Waiting、Unity 无角色：检查 **Connect On Load** 是否勾选、Online/Offline 是否弄反、端口是否同为 8989。
- 只有姿态没有球：确认未加 `--no-markers`，且 Hierarchy 里 **Points** 启用。
- 停 Play 后终端报 Broken pipe：一般是 Unity 主动断开；当前 `MotionViewer.disconnect` 已兼容，可忽略或直接重开脚本。

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

### 廉价验证：LP≪RP 是否来自 AMASS 训练不平衡？

检查产物：`outputs/logs/cheap_checks_lp_vs_rp.json`（2026-08-03）。

| 检查 | 结果 |
|------|------|
| `amass_train` 四组合计数 | 各 **25.00%**（完全均衡） |
| `amass_val` 四组合计数 | 各 **25.00%** |
| AMASS Val Phone Acc（按组合） | LW+LP 0.949 / LW+RP 0.942 / RW+LP 0.957 / RW+RP 0.943 |
| AMASS Val 汇总 | LP 均值 Phone **0.953**，RP 均值 **0.943**（仿真上 LP 还略好） |

**结论：排除「AMASS 标签数量不平衡导致真机 LP 差」**。真机 LP≪RP 更可能来自 IMUPoser 域（左右袋信号可分性、设备/衣物/步态等），而非训练集组合偏斜。惯用手摆臂假说也与「差距在 Phone 不在 Watch」不太吻合，可降级。

### 按运动类型（by_motion，四组合窗级加权平均，W=90）

数据来自 `outputs/logs/metrics_test_{lw_lp,lw_rp,rw_lp,rw_rp}.json` 的 `by_motion`。  
IMUPoser 原始命名有拼写/大小写变体，下表已合并同义名（如 `Joogging→Jogging`、`Lower Body→LowerBody`、`Armcrossing/ArmsCrossing→ArmCrossing`、`startClap*→StartClapping` 等）。`n` 为四组合窗数之和。

**准确率较高（全身/下肢动态为主）**

| 运动 | n | Joint | Watch | Phone |
|------|--:|------:|------:|------:|
| Walking | 6128 | **0.870** | 0.947 | 0.918 |
| Jogging | 1704 | **0.814** | 0.910 | 0.896 |
| JumpingJacks | 776 | **0.744** | 0.974 | 0.753 |
| LowerBody | 2696 | **0.698** | 0.932 | 0.753 |
| Hopping | 656 | **0.681** | 0.817 | 0.838 |

**准确率较低（上肢精细/对称动作，Phone 侧常崩）**

| 运动 | n | Joint | Watch | Phone |
|------|--:|------:|------:|------:|
| ArmRaises | 1772 | **0.448** | 0.936 | 0.467 |
| ArmCrossing | 784 | **0.425** | 0.897 | 0.480 |
| StartClapping | 200 | **0.345** | 0.745 | 0.475 |
| Waving | 836 | **0.329** | 0.895 | 0.362 |
| ClappingFull | 368 | **0.307** | 0.582 | 0.497 |
| Punching | 76 | **0.000** | 0.447 | 0.000 |

中游常见：`Pushups` / `Kicking` / `TennisSwings` / `Basketball` / `HeadMovements` / `Sitting` / `Boxing` / `ArmSwings`（Joint 约 0.45–0.61）。

观察：
- **高分运动**多为 Walking、Jogging、JumpingJacks、LowerBody、Hopping 等，身体整体加速度大，左右袋更易分。
- **低分运动**多为 ArmRaises、ArmCrossing、Waving、ClappingFull、Punching 等上肢动作：Watch 往往仍高，**Phone Acc 掉到 ~0.36–0.50**，拖垮 Joint。
- 同名运动在 **LP vs RP** 上差距仍大（例如 ArmCrossing 在 LP 组合常接近 0，在 RP 可到 0.7+）；上表是四组合平均。
- 单个「好序列」≠该运动整体容易：如 ClappingFull 整体 Joint 仅 0.307，但 LW+RP 下个别序列（如 seq102）可到 1.0。

### 简要结论
- 合成域（AMASS）很强（Joint 0.93）；真实域四组合 Joint 约 **0.55–0.76**。
- **右袋（RP）明显好于左袋（LP）**：LW+RP 0.72 / RW+RP 0.76，而 LW+LP / RW+LP 仅 ~0.55（且该差距**不见于** AMASS val）。
- 手表侧整体不难（Watch Acc 0.88–0.94）；瓶颈主要在手机侧（Phone Acc 在 LP 组合掉到 ~0.60）。
- 更长窗抬高窗级 Acc；序列级 Acc 对窗长相对不敏感。
- 运动层面：**Walking / Jogging / JumpingJacks / LowerBody / Hopping** 明显更好；**ArmRaises / ArmCrossing / Waving / ClappingFull / Punching** 明显更差（详见上表）。

## 已知注意点
- IMUPoser 官方数据是 5 路全开；测试标签 = watch/phone 槽位，不是 zip 里另附的佩戴元数据。
- 腕 IMU 特征来自肘关节朝向代理，报告里写清楚。
- 静止段左右难分：可后续按动态/准静态分开统计。
- `outputs/` 下 `.pt` / checkpoint / metrics JSON 视为可复现产物，默认不提交；以本 README 结果表为对外记录。
