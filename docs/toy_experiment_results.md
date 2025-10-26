# Toy Experiment Results

由于当前执行环境无法通过企业代理下载 PyTorch 与 PyTorch Geometric 依赖，我们临时提供了一个仅基于 Python 标准库的玩具信任分类实验脚本 `scripts/run_toy_experiment.py`。该脚本模拟三类节点的证据式分类任务，用以验证仓库中的不确定性评价指标流程，并给出可复现的数值结果。待能在具备网络访问能力的环境中安装完整依赖后，可将此脚本视作 sanity check，再运行 `src/train.py` 获取真实图神经网络实验结果。

## 运行方式

```bash
python scripts/run_toy_experiment.py
```

脚本会输出验证集与测试集上的分类、校准与不确定性指标。

## 参考输出

```text
Toy validation metrics:
    accuracy: 0.9000
    macro_f1: 0.9022
         nll: 0.2773
       brier: 0.1689
         ece: 0.0793
  uncertainty: 0.2000
     entropy: 0.3121

Toy test metrics:
    accuracy: 0.9833
    macro_f1: 0.9842
         nll: 0.1240
       brier: 0.0548
         ece: 0.0713
  uncertainty: 0.2000
     entropy: 0.2660
```

> **说明**：上述指标来源于脚本在默认随机种子下的实际运行输出，仅作为验证指标计算逻辑的例子。实际在 VeReMi 或 TON_IoT 数据集上的结果需在安装 PyTorch 依赖后运行完整训练流程获得。
