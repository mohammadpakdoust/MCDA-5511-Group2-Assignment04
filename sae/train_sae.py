import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import os

class SAETrainer:
    """
    Handles training of the SparseAutoencoder model.
    Includes data loading, optimization, and early stopping.
    """
    def __init__(
        self, 
        sae_model: nn.Module, 
        lr: float = 1e-4, 
        batch_size: int = 128,
        device: str = "cpu"
    ):
        self.model = sae_model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.device = device
        self.history = {"train_loss": [], "val_loss": [], "sparsity": []}

    def train(
        self, 
        activations: torch.Tensor, 
        epochs: int = 50, 
        val_split: float = 0.1,
        patience: int = 5
    ):
        """Standard training loop with validation and early stopping."""
        dataset = TensorDataset(activations)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        
        train_set, val_set = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size)
        
        best_val_loss = float('inf')
        no_improve_epochs = 0
        
        print(f"Starting training on {self.device}...")
        for epoch in range(epochs):
            self.model.train()
            train_total_loss = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                x = batch[0].to(self.device)
                
                # Forward pass
                _, total_loss, mse, l1, _ = self.model(x)
                
                # Backprop
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                # Post-step normalization of decoder weights
                self.model.normalize_decoder_weights()
                
                train_total_loss += total_loss.item()
                pbar.set_postfix({"loss": total_loss.item(), "mse": mse.item(), "l1": l1.item()})
            
            # Validation
            val_loss = self.evaluate(val_loader)
            self.history["train_loss"].append(train_total_loss / len(train_loader))
            self.history["val_loss"].append(val_loss)
            
            print(f"Epoch {epoch+1}: train_loss={train_total_loss/len(train_loader):.6f}, val_loss={val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_epochs = 0
                torch.save(self.model.state_dict(), "sae/best_sae_model.pt")
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    print("Early stopping triggered.")
                    break
                    
        # Load best model
        self.model.load_state_dict(torch.load("sae/best_sae_model.pt"))

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        """Evaluate model on validation/test data."""
        self.model.eval()
        total_loss = 0
        for batch in loader:
            x = batch[0].to(self.device)
            _, loss, _, _, _ = self.model(x)
            total_loss += loss.item()
        return total_loss / len(loader)

    def save_model(self, path: str = "sae/sae_model.pt"):
        """Save the trained SAE model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
