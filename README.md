# EvidentGraphTrust

EvidentGraphTrust is a lightweight research scaffold for analysing untrusted
behaviour in IoT communication graphs using evidential graph neural networks.
The implementation is inspired by the TrustGuard project while following the
ideas presented in *Evidential Graph Neural Networks for Uncertainty-aware Node
Classification*.  The code base focuses on three aspects:

1. **Trust prediction with quantified uncertainty.** An evidential GNN produces
   Dirichlet trust scores for each device, allowing us to rank likely malicious
   actors together with an uncertainty estimate.
2. **Systematic comparison.** Classic GCN/GraphSAGE baselines and a logistic
   regression classifier are trained on the same synthetic IoT benchmark for a
   head-to-head performance and calibration study.
3. **Ablation analysis.** Evidence-aware loss functions and graph structure
   enhancement can be toggled individually to assess their contribution to
   accuracy and calibration quality.

The repository is intentionally self-contained: a synthetic IoT scenario is
procedurally generated so the experiments can be executed without downloading
external datasets.

## Installation

Create a virtual environment and install the dependencies listed in
`requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project relies on PyTorch; if you have access to GPU acceleration, install
an appropriate CUDA build instead of the CPU wheel indicated in the
requirements file.

## Running the experiments

The full evaluation (baselines, evidential model, and ablations) can be launched
with:

```bash
PYTHONPATH=src python -m evident_graph_trust.experiments.run_experiment
```

Key command line switches:

- `--epochs`: number of training epochs per model (default: 120).
- `--disable-evidence-loss`: turn off the evidential loss component.
- `--disable-graph-enhancement`: remove the two-hop structural prior when
  message passing.
- `--top-k`: number of suspicious devices to display from the test partition.

The script prints two Markdown tables: the main comparison against baselines and
an ablation study isolating the evidential loss and graph enhancement modules.
It also lists the top suspicious nodes with their malicious probability and
uncertainty estimates.

## Repository structure

```
src/
  evident_graph_trust/
    data/            # Synthetic IoT graph generation utilities
    models/          # GNN architectures and traditional baselines
    training/        # Training loop and evidential losses
    evaluation/      # Reporting helpers
    experiments/     # Experiment entry point
```

## Citing

If you build upon this scaffold, please cite the original evidential GNN paper
and the TrustGuard project that inspired the structure of this repository.
