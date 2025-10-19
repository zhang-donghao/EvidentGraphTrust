# 数据处理与图构建细节

本指南描述如何从 VeReMi 与 TON_IoT 数据集中构建证据图，生成节点与边特征，并准备训练/验证/测试划分。

## 1. VeReMi 数据集

1. **下载与解析**
   - 从 [VeReMi 官网](https://veremi-dataset.github.io/) 下载 `scenarioX` 数据包。
   - 使用官方提供的 `logs_to_csv.py` 将 OMNeT++ 日志转换为 CSV。

2. **预处理脚本**
   - 运行 `python scripts/preprocess_veremi.py --raw-root <转换后CSV目录> --output-root data`。
   - 默认会使用 1s 时间窗口、0.5s 滑动步长，将图数据保存到 `data/veremi/processed/train_graph.pt` 等文件。
   - 可通过参数 `--window-size`、`--stride`、`--distance-threshold` 等调整窗口与边构建策略。

3. **时间窗口划分**
   - 选择窗口长度 `Δt = 1s`，滑动步长 `0.5s`。
   - 对每个窗口统计参与通信的车辆集合，形成子图。

4. **节点特征**
   - 连续特征：速度、加速度、航向角、位置 (x, y)、历史误差（预测与广播位置差）。
   - 推理特征：上一窗口的信任评分、消息接收次数。
   - 标准化：对连续特征使用 `StandardScaler`，对缺失值采用前向填充。

5. **边特征**
   - 相对距离、相对速度、是否有直接通信。
   - 采用 RBF kernel 将距离映射为权重，用于 GAT/GraphSAGE。

6. **标签定义**
   - 正类：存在欺骗/伪装攻击的车辆。
   - 负类：正常车辆。
   - 保持攻击类型在各划分中分布一致。

7. **图打包**
   - 使用 PyTorch Geometric 的 `Data`/`HeteroData` 存储节点特征、边索引、边特征、标签。
   - 将多个窗口堆叠为 `InMemoryDataset`，保存为 `.pt` 文件。
   - 预处理脚本会同时生成 `summary.json` 统计各划分图数量、节点规模与攻击比率。

## 2. TON_IoT 数据集

1. **下载与解析**
   - 从 [UNSW 官网](https://research.unsw.edu.au/projects/toniot-datasets) 下载以下官方 CSV 压缩包，并解压到同一目录：
     - `Train_Test_IoT_Telemetry.zip`（IIoT/Telemetry 数据集，内含 `Train_Test_IoT_Telemetry.csv` 等传感器读数）。
     - `Train_Test_Network.zip`（Network Traffic 数据集，内含 `Train_Test_Network.csv` 等网络流量统计）。
     - 可选：若需结合系统日志，可额外下载 `Train_Test_Windows.zip`、`Train_Test_Linux.zip` 等日志 CSV，脚本会自动忽略无法识别的字段。
   - 统一时间戳并按设备 ID 分组。

2. **节点与边**
   - 节点：设备 ID 或服务实例。
   - 边：同一网段通信、历史交互次数 > 阈值、或共享协议栈。
   - 可选：构建双层图（设备层 + 服务层），使用 `HeteroData`。

3. **特征工程**
   - 统计窗口 (`Δt = 5min`) 内的均值、方差、最大值等；
   - 网络流量特征：包数、字节、流持续时间、端口分布；
   - 类别编码：协议类型（One-Hot 或嵌入）、设备类型。

4. **标签与划分**
   - 标签来自提供的攻击标记；
  - 训练/验证：选取 70% 设备；测试：剩余 30% 未见设备；
   - 确保每类攻击在测试集中出现。

5. **图存储**
   - 采用 `torch.save` 存储预处理后的图；
   - 提供索引文件，记录每个时间窗口、设备映射。
   - 运行 `python scripts/preprocess_toniot.py --raw-root <TON_IoT目录> --output-root data` 自动完成上述流程。

## 3. 数据增强与平衡

- 使用随机边 Dropout (EdgeDrop) 以模拟通信不稳定；
- SMOTE 或 Focal Loss 处理类别不平衡；
- 对时间窗口抽样，以确保攻击事件不会集中在单个批次。

## 4. 复现脚本约定

- `src/data/datamodules.py`：实现 `prepare_data()` 与 `setup(stage)` 方法；
- `scripts/download_data.sh`：提供自动下载与解压脚本；
- `configs/data/veremi.yaml`、`configs/data/toni.yaml`：存储特征选择、窗口大小等参数。

实际运行流程示例：

```bash
python scripts/preprocess_veremi.py --raw-root /data/VeReMi_csv --output-root data
python scripts/preprocess_toniot.py --raw-root /data/TON_IoT --output-root data

# 训练证据图神经网络
python src/train.py --dataset-root data --dataset-name veremi --epochs 100
python src/train.py --dataset-root data --dataset-name toni_iot --epochs 100
```

## 5. 数据质量检查

- 统计缺失值比例、节点度分布、边权分布；
- 绘制攻击节点的度数与信任分数对比；
- 使用 `pytest` + `great_expectations` 编写数据质量测试（可选）。

