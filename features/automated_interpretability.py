import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Any, Dict
import os

class AutomatedInterpretor:
    """
    Performs automated interpretability by clustering the geometric
    weight decoders of the learned SAE features. Features clustered together
    in the underlying activation space generally encode related semantic or syntactic concepts.
    """
    def __init__(self, sae: Any):
        self.sae = sae

    @torch.no_grad()
    def cluster_and_visualize(self, n_clusters: int = 10, save_path: str = "plots/feature_clusters.png") -> Dict[int, list]:
        """
        Runs PCA to reduce dimensionality to 2D for visualization, applies K-Means, 
        and plots the result. Returns a dict mapping cluster IDs to feature indices.
        """
        print("\n--- Running Automated Interpretability (Feature Clustering) ---")
        
        # Extract decoder weights [hidden_dim, input_dim]
        # The decoder weights are the geometric direction of each feature in activation space.
        w_dec = self.sae.W_dec.cpu().numpy()
        
        print("Applying PCA for 2D visualization...")
        pca = PCA(n_components=2)
        reduced_weights = pca.fit_transform(w_dec)
        
        print(f"Applying K-Means with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(w_dec)
        
        # Plotting
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(10, 8))
        
        scatter = plt.scatter(reduced_weights[:, 0], reduced_weights[:, 1], 
                           c=cluster_labels, cmap='tab10', alpha=0.3, s=15)
        
        plt.title('Automated Interpretability: SAE Feature Clusters (PCA Projection)')
        plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.colorbar(scatter, label='Cluster ID')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(save_path)
        input("\n[DEMO] Press ENTER to display the Feature Clusters graph...")
        plt.show(block=True)
        plt.close()
        print(f"Feature clusters visualization saved to {save_path}")
        
        # Organize clusters
        clusters = {i: [] for i in range(n_clusters)}
        for feature_idx, label in enumerate(cluster_labels):
            clusters[label].append(feature_idx)
            
        return clusters
