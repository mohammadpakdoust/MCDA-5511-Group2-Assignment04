# Mechanistic Interpretability with Sparse Autoencoders (SAEs)

This project implements a production-quality pipeline for decomposing the hidden activations of a Large Language Model (LLM) into sparse, interpretable features using Sparse Autoencoders (SAEs). This approach is inspired by Anthropic’s "Scaling Monosemanticity" Research.

## Overview

### What is Activation Space?
In a transformer model, the "activation space" refers to the high-dimensional vector representations of tokens at a given layer. While these vectors (residual stream) contain all the information the model uses for computation, they are dense and difficult for humans to interpret directly due to **superposition**—the phenomenon where the model packs more features into fewer dimensions than it has neurons.

### What is a Feature?
Anthropic defines a **feature** as a semantically coherent direction in activation space. A single neuron might be "polysemantic" (responding to multiple unrelated concepts), but an SAE identifies "monosemantic" Directions that correspond to single, clear concepts (e.g., "French grammar," "medical terminology," or "sentiment").

### How SAEs Map Activations to Features
An SAE is trained to reconstruct activations $x$ through a high-dimensional, sparse bottleneck $f$:
1. **Encoder**: Projects activations into a higher-dimensional space ($4\times$ to $64\times$ the original size) and applies a ReLU to ensure sparsity.
2. **Sparsity Penalty**: An $L_1$ regularization term is added to the loss function to encourage that only a few features are active at once.
3. **Decoder**: Reconstructs the original activation from the sparse features.

Formula: $x \approx \hat{x} = W_{dec} \cdot \text{ReLU}(W_{enc}(x - b_{dec}) + b_{enc}) + b_{dec}$

## Project Structure

```text
sae-interpretability/
├── main.py                 # Central execution pipeline
├── models/
│   └── model_wrapper.py    # HF Model loading and hook management
├── data/
│   └── dataset_generator.py# Activation extraction from wikitext
├── sae/
│   ├── sae_model.py        # PyTorch SAE implementation
│   └── train_sae.py        # Training logic and early stopping
├── features/
│   ├── feature_extractor.py # Top-k activating example retrieval
│   ├── feature_analyzer.py  # Interpretation and summary tools
│   └── intervention.py      # Causal feature clamping
├── utils/
│   └── helpers.py          # Visualization and logging
└── pyproject.toml          # Dependency management (uv)
```

## Setup & Usage

### 1. Environment

This project uses [uv](https://github.com/astral-sh/uv) for fast, reproducible environment management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install all dependencies
uv sync

# Activate the environment
source .venv/bin/activate
```

### 2. Run the Full Pipeline
```bash
# With the venv activated:
python main.py --max_samples 500 --epochs 20

# Or run directly via uv without activating:
uv run python main.py --max_samples 500 --epochs 20
```

This will:
1. Load `Qwen/Qwen2.5-1.5B`.
2. Extract activations from Layer 14.
3. Train an SAE to decompose these activations.
4. Print top-activating tokens for the most prominent features.
5. Perform a **Causal Intervention** (clamping) to see how forcing a feature changes model output.

## Experimental Setup

- **Model**: `Qwen2.5-1.5B` (running on MPS/Apple Silicon).
- **Layer**: Mid-layer residual stream (Layer 14).
- **SAE Expansion**: 1536 input dimensions -> 6144 feature dimensions.
- **Regularization**: $L_1$ penalty with $\lambda=0.1$.

## Causal Intervention (Feature Clamping)

To prove a feature is truly what we think it is, we modify the activations during the model's forward pass. By "clamping" a specific feature index to a high value, we force the model to behave as if that concept is present. If the output text shifts toward that concept, we've demonstrated a **causal influence**.

## Limitations & Future Work

- **Dictionary Learning**: This project uses a basic SAE; modern improvements like "JumpRELU" or "Gated SAEs" could improve results.
- **Expansion Factor**: We use $4\times$ expansion, but meaningful feature discovery often requires $32\times$ or more.
- **Compute**: Training on millions of tokens is required for high-fidelity feature discovery.
