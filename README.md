# MAMBA Barebones

A lightweight PyTorch implementation of the Mamba State Space Model (SSM) architecture, including both sequential and parallel scan implementations for efficient sequence modeling.

## Overview

This repository contains:

* A minimal implementation of the Mamba SSM architecture.
* Sequential and parallel scan (Blleloch-scan) implementations of the Mamba recurrence.
* Experiment logging and evaluation notebooks.

## Repository Structure

```text
.
├── README.md
├── checkpoints
├── configs
│   └── config.py
├── data
│   └── wikitext-2
│       ├── README.md
│       ├── test.txt
│       ├── train.txt
│       └── valid.txt
├── datasets.py
├── exp.py
├── experiments.csv
├── experiments.ipynb
├── logs
├── model
│   ├── blleloch_scan_concept.py
│   ├── block_skeleton.py
│   ├── ffn_mlp.py
│   ├── language_model.py
│   ├── mamba_layer.py
│   ├── mamba_model.py
│   └── parallel_scan.py
├── requirements.txt
├── saved_models
├── test.py
├── train.py
└── utils
    ├── checkpoints.py
    ├── test_utils.py
    └── train_utils.py

```
## Installation

Clone the repository:

```bash
git clone https://github.com/<username>/MAMBA.git
cd MAMBA
```
Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Run training with:

```bash
python train.py
```

Example:

```bash
python train.py \
    --d_model 128 \
    --n_layer 4 \
    --d_state 16 \
    --dropout 0.2 \
    --weight_decay 0.01
```

## Hyperparameter Experiments

Experiments can be launched using the experiment scripts provided in the repository.

Experiment configurations are recorded in:

```text
experiments.csv
```

Training logs are stored in:

```text
logs/
```

Model checkpoints are stored in:

```text
checkpoints/
```

## Parallel Scan Verification

The repository includes tests comparing:

* Sequential recurrence implementation
* Parallel scan implementation

Verification checks:

* Forward pass equivalence
* Gradient equivalence with respect to transition parameters
* Gradient equivalence with respect to inputs

## Results

Results and analyses can be found in:

```text
experiments.ipynb
```

## Notes

Large checkpoint files are excluded from version control through `.gitignore`.

Recommended directories to exclude:

```text
checkpoints/
saved_models/
logs/
```
