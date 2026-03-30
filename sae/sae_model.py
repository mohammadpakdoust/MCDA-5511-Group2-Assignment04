import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    """
    Implementation of a Sparse Autoencoder (SAE) for hidden activations.
    Decomposes activation space into sparse, interpretable features.
    
    Architecture:
    Encoder: x -> ReLU(W_enc * (x - b_dec) + b_enc)
    Decoder: f -> W_dec * f + b_dec
    """
    def __init__(self, input_dim: int, hidden_dim: int, l1_lambda: float = 0.1):
        super(SparseAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.l1_lambda = l1_lambda
        
        # Decoder bias initialized to the median of the data (best practice)
        self.b_dec = nn.Parameter(torch.zeros(input_dim))
        
        # Encoder weights
        self.W_enc = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.b_enc = nn.Parameter(torch.zeros(hidden_dim))
        
        # Decoder weights
        self.W_dec = nn.Parameter(torch.empty(hidden_dim, input_dim))
        
        # Tie weights: self.W_dec.data = self.W_enc.data.T
        # We initialize them using Kaiming normal
        nn.init.kaiming_normal_(self.W_enc)
        self.W_dec.data = self.W_enc.data.T
        
        # Standardize decoder weights to have unit norm
        self.normalize_decoder_weights()

    def normalize_decoder_weights(self):
        """Normalize decoder weights to unit norm."""
        with torch.no_grad():
            norms = torch.norm(self.W_dec, dim=1, keepdim=True)
            self.W_dec.data /= (norms + 1e-8)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Subtract decoder bias and project to hidden space with ReLU."""
        x_centered = x - self.b_dec
        features = F.relu(torch.matmul(x_centered, self.W_enc) + self.b_enc)
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Map features back to activation space."""
        return torch.matmul(features, self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple:
        """Full forward pass with reconstruction and latent features."""
        features = self.encode(x)
        reconstruction = self.decode(features)
        
        # Calculate losses
        reconstruction_loss = F.mse_loss(reconstruction, x)
        sparsity_loss = self.l1_lambda * features.abs().sum() / x.shape[0]
        
        total_loss = reconstruction_loss + sparsity_loss
        
        return reconstruction, total_loss, reconstruction_loss, sparsity_loss, features

    @torch.no_grad()
    def get_feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Utility for analysis to get hidden features."""
        return self.encode(x)
