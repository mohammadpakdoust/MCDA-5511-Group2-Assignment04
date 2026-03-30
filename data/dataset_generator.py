import torch
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import os

class DatasetGenerator:
    """
    Handles prompt processing, activation extraction and saving the 
    dataset for SAE training.
    """
    def __init__(self, model_wrapper: Any, layer_idx: int = 14):
        self.model_wrapper = model_wrapper
        self.layer_idx = layer_idx
        self.activations_data = []
        self.token_data = []
        
        # Register hook on initialize
        self.model_wrapper.register_layer_hook(self.layer_idx)

    def generate_from_dataset(
        self, 
        dataset_name: str = "wikitext", 
        dataset_config: str = "wikitext-2-raw-v1",
        split: str = "train",
        max_samples: int = 100,
        max_seq_len: int = 128
    ):
        """
        Loads a standard dataset and extracts activations.
        """
        print(f"Loading {dataset_name} ({dataset_config}) subset...")
        dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
        
        samples_processed = 0
        pbar = tqdm(total=max_samples)
        
        for sample in dataset:
            text = sample['text'].strip()
            if not text:
                continue
                
            activations = self.model_wrapper.get_activations(text)
            # activations is [1, seq_len, hidden_dim]
            # tokens is [1, seq_len]
            tokens = self.model_wrapper.tokenizer(text, return_tensors="pt")["input_ids"]
            
            # Subsample for memory efficiency if needed
            # We flatten the sequence dimension
            self.activations_data.append(activations.squeeze(0).cpu())
            self.token_data.append(tokens.squeeze(0).cpu())
            
            samples_processed += 1
            pbar.update(1)
            
            if samples_processed >= max_samples:
                break
                
        pbar.close()
        
        # Concatenate everything into large tensors
        self.final_activations = torch.cat(self.activations_data, dim=0)
        self.final_tokens = torch.cat(self.token_data, dim=0)
        
        print(f"Extracted {len(self.final_activations)} token activations.")

    def save_dataset(self, save_path: str = "data/activations.pt"):
        """
        Saves the extracted data to a PyTorch file.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "activations": self.final_activations,
            "tokens": self.final_tokens,
            "layer_idx": self.layer_idx
        }, save_path)
        print(f"Dataset saved to {save_path}")

    def load_dataset(self, load_path: str = "data/activations.pt"):
        """
        Loads the dataset from disk.
        """
        data = torch.load(load_path)
        return data["activations"], data["tokens"]
