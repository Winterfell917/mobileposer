# Week3 — Unknown-mount cascade (position → \(R_{SB}\) → pose)

> 相对 Week1 / Week2 的关系  
> **Week1**：忽略朝向干扰，只做位置分类（acc+gyro）  
> **Week2**：位置 **已知**，估 \(R_{SB}\)  
> **Week3**：输入只有 \(a_M, R_{MS}\)，\(R_{SB}\) **未知**；先位置，再外参，再姿态

## 第 1 步（本目录当前内容）

从 \(a_M, R_{MS}\) 推断设备位置。遵循附图协议：

\[
R_{MS} = R_{MB}\, R_{BS},\quad a_M \text{ 不变}
\]

- 表 / 机各自独立随机 \(R_{BS}\)
- 整条序列（因而同一窗内每一帧）\(R_{BS}\) 恒定
- 网络 **看不到** \(R_{BS}\)，只输出手表左右腕、手机左右袋

模型：与 Week1 相同的双头 BiLSTM，输入改为 24 维（每设备 acc3 + ori9）。  
数据：在线注入，不把展开窗写到磁盘。

```bash
conda activate mobileposer
cd /home/caolindong/projects/mobileposer

python experiments/week3_pos_rsb/dataset/build_amass.py \
  --config experiments/week3_pos_rsb/configs/default.yaml

python experiments/week3_pos_rsb/train.py \
  --config experiments/week3_pos_rsb/configs/default.yaml

python experiments/week3_pos_rsb/eval.py \
  --config experiments/week3_pos_rsb/configs/default.yaml
```

权重：`outputs/checkpoints/best_joint_acc.pt`  
日志：`outputs/logs/train_log.csv`、`metrics_step1.json`

## 后续步骤（未实现）

2. 用 Pred 位置条件化，估 \(R_{SB}\)（接 Week2）  
3. \(R_{MB}=R_{MS}R_{SB}^{\top}\) → MobilePoser 姿态，并做 \(R_{MS}\to\) pose baseline 可视化
