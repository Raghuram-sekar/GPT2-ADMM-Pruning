# GPT-2 Weight Pruning using ADMM
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![MATLAB](https://img.shields.io/badge/MATLAB-%23e05a00.svg?style=for-the-badge) ![Simulink](https://img.shields.io/badge/Simulink-%23e05a00.svg?style=for-the-badge)

## Overview
An optimization project implementing weight pruning/sparsity algorithms on GPT-2 models using the Alternating Direction Method of Multipliers (ADMM) framework in PyTorch. Achieved 40% weight sparsity with less than a 1.5% perplexity increase, backed by MATLAB Live Scripts and Simulink state-space models.

## System Architecture
```\n[Relational Database / Core API Architecture]\n```

## Features
- Structured weight pruning achieving up to 40% weight sparsity on GPT-2 attention and MLP weights.
- Perplexity analysis showing less than 1.5% accuracy drop under target compression.
- MATLAB and Simulink state-space models simulating optimization convergence trends.

## Tech Stack
- PyTorch for GPT-2 attention layer pruning and weight masking
- MATLAB Live Scripts for convergence optimization calculations
- Simulink for structural modeling

## Getting Started
To configure and run the project locally, clone the repository and execute the setup instructions:

```bash
git clone https://github.com/Raghuram-sekar/GPT2-ADMM-Pruning.git
cd GPT2-ADMM-Pruning

# Execute local setup commands:
# ADMM Optimization Formulation:
# min_W f(W) + g(Z) subject to W - Z = 0
# L_rho(W, Z, U) = f(W) + g(Z) + (rho/2)*||W - Z + U||_2^2 - (rho/2)*||U||_2^2

jupyter notebook GPT2_ADMM_Pruning_Implementation.ipynb
```
