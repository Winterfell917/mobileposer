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

配置：`default.yaml`；训练窗长 W=90；checkpoint=`best_joint_acc.pt`（**未重训**，仍为 AMASS 上训得的权重）。  
每个 IMUPoser 组合均为 n=6003 窗 / 167 序列。

### 修复：IMUPoser 真机 IMU 与 DIP 全局系对齐（2026-08-09）

**问题：** `data_process_mocap.process_imuposer` 原先只对 `pose`/`tran` 做 DIP 对齐，写入 `imuposer_full.pt` 的录制 `acc`/`ori` 未乘同一旋转，与 AMASS（DIP 对齐后合成）全局约定可能不一致。

**代码修改：** `mobileposer/data_process_mocap.py` 中，对齐矩阵  
\(R_{\mathrm{align}}=\begin{bmatrix}-1&0&0\\0&0&1\\0&1&0\end{bmatrix}\)  
同时作用于：

- `pose[:,0]`、`tran`（原有）
- **`ori ← R_align @ ori`**
- **`acc ← R_align @ a`**（按全局自由向量）

**复现步骤（本次评测已执行）：**

```bash
cd mobileposer && python data_process_mocap.py --dataset imuposer   # 重写 imuposer_full.pt
cd ..
# 重建 Week1 四组合测试集
for ws in 0 1; do for ps in 0 1; do
  python experiments/week1_pos_cls/dataset/build_imuposer_pos_cls.py \
    --config experiments/week1_pos_cls/configs/default.yaml \
    --watch-side $ws --phone-side $ps
done; done
# 四组合评估
python experiments/week1_pos_cls/eval.py --split test \
  --combos lw_lp,lw_rp,rw_lp,rw_rp
```

日志：`outputs/logs/metrics_summary.json`（2026-08-09 13:35）、`metrics_test_{lw_lp,lw_rp,rw_lp,rw_rp}.json`。

### 主表（W=90）— 对齐后最新

| Split | Combo | Watch Acc | Phone Acc | Joint Acc | Seq Joint Acc |
|-------|-------|-----------|-----------|-----------|---------------|
| AMASS Val | 4 组合混合 | 0.977 | 0.948 | **0.928** | **0.976** |
| IMUPoser | LW+LP | 0.771 | 0.577 | **0.430** | **0.413** |
| IMUPoser | LW+RP | 0.795 | 0.555 | **0.456** | **0.383** |
| IMUPoser | RW+LP | 0.845 | 0.592 | **0.506** | **0.431** |
| IMUPoser | RW+RP | 0.829 | 0.520 | **0.414** | **0.365** |

- **Joint Acc**：单窗上手表+手机都对。
- **Seq Joint Acc**：同一序列内对窗预测做多数投票后再算 Joint。
- AMASS Val 与对齐前相同（测试未动合成集、未重训）。

### 对齐前 → 对齐后（Joint Acc，便于对照）

| Combo | 对齐前 (≤08-02) | **对齐后 (08-09)** | Δ |
|-------|-----------------|--------------------|---|
| LW+LP | 0.557 | **0.430** | −0.127 |
| LW+RP | 0.719 | **0.456** | −0.263 |
| RW+LP | 0.546 | **0.506** | −0.040 |
| RW+RP | 0.755 | **0.414** | −0.341 |

**分析（重要）：**

1. **真机 Joint 全面下降**（约 0.41–0.51，此前 RP 组合曾到 0.72–0.76）。  
2. 原先突出的 **RP ≫ LP** 在对齐后**基本消失**（四组合 Joint 挤在 ~0.41–0.51）。  
3. **Watch Acc 也明显下降**（约 0.77–0.85，此前 0.88–0.94）。  
4. 数学上：Week1 特征里 gyro 由相邻 `ori` 差分得到，对**常数左乘** \(R_{\mathrm{align}}\) **不变**；变化主要来自 **acc 被当作全局向量旋转**。若录制 `acc` 并非与 AMASS `vacc` 同约定的全局自由向量，强行按全局旋转会**加重**域差——与本次数字一致。  
5. **结论：** DIP 对齐修复了「pose 对齐、IMU 未对齐」的预处理不一致；但在「不重训、仅改测试 IMU」设定下，**不能**说对齐一定提升位置分类。下一步应验证 IMUPoser `acc` 是否全局、或在对齐后的真机分布上 **微调/重训** Week1，再比公平对照。

### 窗长消融（Joint Acc）— 对齐后

| W | AMASS Val | LW+LP | LW+RP | RW+LP | RW+RP |
|---|-----------|-------|-------|-------|-------|
| 30 | 0.828 | 0.377 | 0.411 | 0.451 | 0.387 |
| 60 | 0.899 | 0.408 | 0.437 | 0.491 | 0.400 |
| 90 | 0.928 | 0.430 | 0.456 | 0.506 | 0.414 |
| 150 | 0.952 | 0.462 | 0.492 | 0.537 | 0.430 |

（对齐前 IMUPoser 窗长表见 git 历史；趋势仍是更长窗略升，但绝对水平低于对齐前。）

### init_done（K=30 帧 ≈1s）— 对齐后

| Split / Combo | 触发率 | 稳定预测正确率 | 中位耗时 |
|---------------|--------|----------------|----------|
| AMASS Val | 0.924 | 0.960 | 1.0 s |
| IMUPoser LW+LP | 1.000 | 0.443 | 1.0 s |
| IMUPoser LW+RP | 1.000 | 0.419 | 1.0 s |
| IMUPoser RW+LP | 1.000 | 0.443 | 1.0 s |
| IMUPoser RW+RP | 1.000 | 0.407 | 1.0 s |

### 廉价验证：LP≪RP 是否来自 AMASS 训练不平衡？

检查产物：`outputs/logs/cheap_checks_lp_vs_rp.json`（2026-08-03）。

| 检查 | 结果 |
|------|------|
| `amass_train` 四组合计数 | 各 **25.00%**（完全均衡） |
| `amass_val` 四组合计数 | 各 **25.00%** |
| AMASS Val Phone Acc（按组合） | LW+LP 0.949 / LW+RP 0.942 / RW+LP 0.957 / RW+RP 0.943 |
| AMASS Val 汇总 | LP 均值 Phone **0.953**，RP 均值 **0.943**（仿真上 LP 还略好） |

**结论（针对对齐前现象）：** 可排除「AMASS 标签数量不平衡导致真机 LP 差」。对齐前真机 LP≪RP 更可能来自旧 IMUPoser 域（左右袋可分性、设备/衣物等），而非训练集组合偏斜。  
**对齐后补充：** 见上主表，RP≫LP 已基本消失；该项历史假说不再是当前瓶颈叙述。

### 按运动类型（by_motion，对齐后四组合窗级加权，W=90）

数据来自 2026-08-09 的 `metrics_test_*.json`（原始运动名未做同义合并；`Lower Body` / `LowerBody` 分列）。

**相对较高**

| 运动 | n | Joint | Watch | Phone |
|------|--:|------:|------:|------:|
| Hopping | 656 | **0.677** | 0.895 | 0.742 |
| Lower Body | 428 | **0.668** | 0.958 | 0.706 |
| Walking | 6128 | **0.659** | 0.789 | 0.841 |
| Kicking | 436 | **0.626** | 0.982 | 0.635 |
| LowerBody | 2268 | **0.624** | 0.947 | 0.662 |

**明显偏低**

| 运动 | n | Joint | Watch | Phone |
|------|--:|------:|------:|------:|
| ArmRaisesRedo | 220 | **0.164** | 0.877 | 0.218 |
| ArmsCrossing | 80 | **0.138** | 0.775 | 0.175 |
| Boxing | 556 | **0.115** | 0.304 | 0.275 |
| ClappingFull | 368 | **0.106** | 0.321 | 0.280 |
| Punching | 76 | **0.000** | 0.000 | 0.000 |

观察：下肢/行走类仍相对更好，但 Walking Joint 自对齐前 ~0.87 降到 ~0.66；上肢精细动作依然最差，且 Watch 也出现崩盘（Boxing / Clapping / Punching）。

### 简要结论（对齐后）

- 合成域（AMASS）仍强（Joint **0.928**，未重训）。  
- 真机四组合 Joint 约 **0.41–0.51**，**低于**对齐前的 0.55–0.76。  
- 对齐前显著的 **RP ≫ LP** 在本次设定下不再成立。  
- 预处理「IMU 与 pose 同 DIP」在工程上更自洽，但对当前位置分类器并非免费增益；gyro 不变、acc 旋转是主要变化通道。  
- 后续：核对 IMUPoser `acc` 坐标系假设；或在对齐后真机分布上微调 Week1 再比。

### 附录：对齐前主表（历史，≤2026-08-02）

| Split | Combo | Watch | Phone | Joint | Seq Joint |
|-------|-------|------:|------:|------:|----------:|
| IMUPoser | LW+LP | 0.885 | 0.617 | 0.557 | 0.467 |
| IMUPoser | LW+RP | 0.891 | 0.790 | 0.719 | 0.749 |
| IMUPoser | RW+LP | 0.937 | 0.594 | 0.546 | 0.461 |
| IMUPoser | RW+RP | 0.944 | 0.793 | 0.755 | 0.778 |

## 已知注意点
- IMUPoser 官方数据是 5 路全开；测试标签 = watch/phone 槽位，不是 zip 里另附的佩戴元数据。
- 腕 IMU 朝向来自肘关节 18/19 代理，报告里写清楚。
- **2026-08-09 起** `imuposer_full.pt` 的录制 `acc/ori` 与 pose 一并 DIP 对齐；旧测试 `.pt` 需按上文步骤重建。
- 静止段左右难分：可后续按动态/准静态分开统计。
- `outputs/` 下 `.pt` / checkpoint / metrics JSON 视为可复现产物，默认不提交；以本 README 结果表为对外记录。
