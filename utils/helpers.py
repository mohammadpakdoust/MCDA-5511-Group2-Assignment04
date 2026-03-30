import torch
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Any

def plot_training_curves(history: Dict[str, List[float]], save_path: str = "plots/training_curves.png"):
    """
    Plots the training and validation loss curves.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('SAE Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Training curves saved to {save_path}")

def calculate_reconstruction_fidelity(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """
    Calculates the R-squared value for the reconstruction.
    """
    mse = torch.mean((original - reconstructed) ** 2)
    var = torch.var(original)
    r2 = 1 - (mse / var)
    return r2.item()

def get_config_summary(config: Dict[str, Any]) -> str:
    """
    Returns a human-readable summary of the current configuration.
    """
    summary = "Project Configuration:\n"
    for key, value in config.items():
        summary += f" - {key}: {value}\n"
    return summary

def ensure_dirs():
    """Ensure all required directories exist."""
    dirs = ["models", "data", "sae", "features", "utils", "plots", "results"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
