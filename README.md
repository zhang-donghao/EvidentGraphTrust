# Evident Graph Trust (EGT)

Evident Graph Trust (EGT) 是一个围绕证据图神经网络（Evidential Graph Neural Network, EGNN）思想，为物联网与车联网场景设计可信建模实验方案的研究仓库。该方案以论文 ["Evidential Graph Neural Networks for Uncertainty-aware Node Classification"](https://arxiv.org/html/2506.13083v1#S6.F10) 为核心参考，目标是在公开数据集上复现并扩展其可信推理能力，并针对物联网/车联网上的信任评估任务构建可重复的实验流程。

## 研究目标

1. **可信节点分类**：在物联网与车联网的通信图中识别不可信节点（如恶意设备、虚假消息发送者），输出具有不确定性量化的信任评分。
2. **模型对比**：系统地与经典图神经网络和传统机器学习方法比较，验证证据推断的优势。
3. **消融分析**：拆解证据损失、图结构增强等组件，明确各模块对性能与校准性的贡献。
4. **复现与扩展**：提供脚本化的数据处理、训练、评估流程，便于科研人员直接运行与扩展。

## 公共数据集选择

| 场景 | 数据集 | 简介 | 用途 |
| ---- | ------ | ---- | ---- |
| 车联网 | [VeReMi](https://veremi-dataset.github.io/) | 包含 V2X 通信中多种伪装与欺骗攻击的仿真数据，具有时序位置、速度等特征。 | 构建车辆节点图，预测消息可信度。 |
| 物联网 | [TON_IoT](https://research.unsw.edu.au/projects/toniot-datasets) | 包含多源 IoT 设备遥测及攻击标签，可构建设备交互图。 | 识别受攻设备及异常流量节点。 |

我们将针对两个场景分别构建图结构与特征工程流程，以验证模型在不同网络拓扑中的鲁棒性。

## 方法概述

EGT 基于 Evidential GNN 框架，结合可信度与不确定性建模，输出 Dirichlet 分布参数，实现以下特性：

- **证据层**：通过图卷积（GCN/GAT）提取节点表示，输出各类别的证据值。
- **Dirichlet 推断**：将证据转换为 Dirichlet 分布参数，估计后验概率与不确定性。
- **损失函数**：组合交叉熵、正则化项、退化惩罚与 KL 项，对齐真实标签分布并抑制过度自信。
- **先验融合**：可将域知识（如车辆信誉、设备类型）融入证据初始化，提升可信度。

## 对比实验设计

| 模型 | 类型 | 说明 |
| ---- | ---- | ---- |
| **EGT (Ours)** | Evidential GNN | 基于 Dirichlet 证据输出的图信任模型。 |
| GCN | GNN | 经典图卷积网络节点分类。 |
| GAT | GNN | 图注意力网络，衡量邻居重要性。 |
| GraphSAGE | GNN | 采样式邻居聚合，适配大规模图。 |
| MLP + Trust Features | 神经网络 | 仅使用节点特征，无图结构。 |
| XGBoost | 传统方法 | 梯度提升树，作为非深度学习基线。 |

所有模型将共享相同的训练/验证/测试划分，评估其在信任判别任务上的性能与不确定性。

## 消融实验

1. **无证据损失**：去掉证据正则与 KL 项，观察校准性下降。
2. **无结构增强**：仅使用特征，不利用图边，评估图结构贡献。
3. **无先验约束**：移除域知识先验，验证先验对可信预测的影响。
4. **简化证据层**：将证据层替换为线性层，考察多层证据聚合的必要性。

## 评价指标

- **分类性能**：Accuracy, Macro-F1, ROC-AUC, PR-AUC。
- **校准指标**：Expected Calibration Error (ECE), Brier Score, Negative Log-Likelihood (NLL)。
- **不确定性质量**：Proper Scoring Rule、可视化（置信度-准确率曲线）。
- **任务特定指标**：FAR/FRR（误报率/漏报率），尤其适用于安全场景。

## 实验流程

1. **数据预处理**
   - 下载公开数据集，解析原始记录，构建时序窗口。
   - 构图策略：根据通信关系、空间邻近性或时间共现建立边。
   - 特征工程：标准化数值特征，编码类别属性，构建历史信誉特征。

> **快速开始**：预处理脚本已经将上述流程脚本化。在运行之前，请先从 [TON_IoT 官网](https://research.unsw.edu.au/projects/toniot-datasets) 下载 `Train_Test_IoT_Telemetry.zip` 与 `Train_Test_Network.zip`（如需日志特征可追加 `Train_Test_Windows.zip`/`Train_Test_Linux.zip`），并解压到同一目录，供预处理脚本读取。
> ```bash
> # 1. 处理 VeReMi 车联网数据
> python scripts/preprocess_veremi.py --raw-root /path/to/VeReMi_csv --output-root data
>
> # 2. 处理 TON_IoT 物联网数据
> python scripts/preprocess_toniot.py --raw-root /path/to/TON_IoT --output-root data
>
> # 3. 训练证据图网络（示例）
> python src/train.py --dataset-root data --dataset-name veremi --epochs 100
> python src/train.py --dataset-root data --dataset-name toni_iot --epochs 100
> ```
> 每个脚本会在 `data/<dataset>/processed/` 目录下生成 `train/val/test` 图张量以及 `summary.json`，供后续训练与评估直接使用。

2. **建模与训练**
   - 使用 PyTorch Geometric 构建模型，支持多 GPU/多进程训练。
   - 超参数搜索：学习率、证据正则强度、dropout、图层数。
   - 采用早停与模型检查点，确保可复现。

3. **评估与分析**
   - 汇总主指标，对比各模型表现。
   - 绘制置信度-准确率曲线、可靠性图。
   - 进行消融实验，分析性能变化与统计显著性（t 检验/置信区间）。

4. **报告撰写**
   - 按经典论文结构撰写实验章节：数据集、设置、结果、讨论。
   - 在附录提供超参数、额外图表与复现实验说明。

## 仓库结构规划

```
EvidentGraphTrust/
├─ docs/
│  ├─ experiment_design.md       # 详细实验与消融计划
│  ├─ dataset_preprocessing.md   # 数据处理与构图细节
│  └─ reporting_checklist.md     # 论文撰写规范与检查表
├─ src/
│  ├─ data/
│  │  ├─ __init__.py
│  │  └─ datamodules.py          # 数据加载 & PyG DataModule
│  ├─ models/
│  │  ├─ __init__.py
│  │  └─ egtn.py                 # 证据图神经网络实现
│  ├─ utils/
│  │  ├─ __init__.py
│  │  └─ metrics.py              # 校准与分类指标
│  └─ train.py                   # 训练与评估脚本入口
├─ configs/
│  └─ default.yaml               # Hydra/Argparse 配置
├─ requirements.txt              # 依赖列表
└─ README.md                     # 项目总览
```

> **离线演示**：若当前环境无法安装 PyTorch/PyG，可先运行 `python scripts/run_toy_experiment.py` 查看仅依赖 Python 标准库的玩具信任分类实验，了解指标计算流程。详见 [`docs/toy_experiment_results.md`](docs/toy_experiment_results.md)。待网络条件允许后，再按照上文流程运行完整的图神经网络实验。

后续提交将逐步补充代码实现与实验脚本，以支持完整的端到端实验流程。

