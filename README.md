# GPT-2 Weight Pruning using ADMM

An optimization project implementing weight pruning/sparsity algorithms on GPT-2 models using the Alternating Direction Method of Multipliers (ADMM) framework in PyTorch. Achieved 40% weight sparsity with less than a 1.5% perplexity increase, backed by MATLAB Live Scripts and Simulink state-space models.

## Features
- Structured weight pruning achieving up to 40% weight sparsity on GPT-2 attention and MLP weights.
- Perplexity analysis showing less than 1.5% accuracy drop under target compression.
- MATLAB and Simulink state-space models simulating optimization convergence trends.

## Tech Stack
- PyTorch
- Python
- MATLAB Live Scripts
- Simulink
- ADMM Optimization

## Getting Started
To configure and run the project locally, clone the repository and execute the setup instructions:

```bash
git clone https://github.com/Raghuram-sekar/GPT2-ADMM-Pruning.git
cd GPT2-ADMM-Pruning

# Execute local setup commands:
# Open and run the Jupyter notebook:
jupyter notebook GPT2_ADMM_Pruning_Implementation.ipynb
```
