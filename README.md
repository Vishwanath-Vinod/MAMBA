# MAMBA Barebones

A lightweight PyTorch implementation of the Mamba State Space Model (SSM) architecture, including both sequential and parallel scan implementations for efficient sequence modelling.

## Overview

This repository contains:

* A minimal implementation of the Mamba SSM architecture.
* Sequential and parallel scan (Blleloch-scan) implementations of the Mamba recurrence.
* Experiment logging and evaluation notebooks.

## Repository Structure

```text
.
├── README.md                    # Project documentation
├── checkpoints/                 # Saved training checkpoints
├── configs/
│   └── config.py                # Training and model configuration
├── data/
│   └── wikitext-2/              # WikiText-2 dataset
│       ├── README.md
│       ├── train.txt
│       ├── valid.txt
│       └── test.txt
├── datasets.py                  # Dataset loading utilities
├── exp.py                       # Experiment runner
├── experiments.ipynb            # Result analysis and visualization
├── logs/                        # Training logs
├── model/
│   ├── blleloch_scan_concept.py # Blelloch scan demonstration
│   ├── block_skeleton.py        # Base model block structure
│   ├── ffn_mlp.py               # Feed-forward network module
│   ├── language_model.py        # Language modelling wrapper
│   ├── mamba_layer.py           # Single Mamba layer
│   ├── mamba_model.py           # Full Mamba architecture
│   └── parallel_scan.py         # Parallel scan implementation
├── requirements.txt            # Python dependencies
├── saved_models/               # Final trained models
├── test.py                     # Testing script
├── train.py                    # Training entry point
├── test_official.py            # Testing performance of official MAMBA implementation.
├── train_official.py           # Training official MAMBA model.
├── test_generation.py          # Tests autoregressive generation.
└── utils/
    ├── checkpoints.py          # Checkpoint utilities
    ├── test_utils.py           # Evaluation helpers
    └── train_utils.py          # Training utilities
    ├── sampling_utils.py       # Helper functions for facilitating sampling (generation).
    └── generation_utils.py     # Helper functions for Autoregressive Generation.
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
python3 train.py
```

Example:

```bash
python3 train.py \
    --d_model 128 \
    --n_layer 4 \
    --d_state 16 \
    --dropout 0.2 \
    --weight_decay 0.01
```

## Experiments

A detailed ablation study for multiple hyperparameter configurations with multiple architectural implementations are present in:

```text
experiments.ipynb
```
It also contains a comparison with the official MAMBA implementation as a sanity check.


## Parallel Scan Verification

The repository includes tests comparing:

* Sequential recurrence implementation
* Parallel scan implementation

Verification checks:

* Forward pass equivalence
* Gradient equivalence with respect to transition parameters
* Gradient equivalence with respect to inputs
