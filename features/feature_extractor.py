import torch
from typing import List, Dict, Tuple, Any

class FeatureExtractor:
    """
    Extracts and maps hidden activations of the SAE to their corresponding 
    tokens and text from the original model.
    """
    def __init__(self, sae: Any, tokenizer: Any):
        self.sae = sae
        self.tokenizer = tokenizer

    @torch.no_grad()
    def get_top_k_activating_examples(
        self, 
        activations: torch.Tensor, 
        token_ids: torch.Tensor, 
        feature_idx: int, 
        k: int = 10,
        context_window: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Identify the top-k activating tokens for a given feature index.
        Returns the token IDs, their decoded text, and activation strength for each.
        """
        # Get feature activations for all tokens: [num_tokens, feature_dim]
        feature_activations = self.sae.get_feature_activations(activations)
        
        # Get activations for the target feature: [num_tokens]
        target_feature_acts = feature_activations[:, feature_idx]
        
        # Sort activations to find top-k
        top_values, top_indices = torch.topk(target_feature_acts, k)
        
        results = []
        for i in range(k):
            idx = top_indices[i].item()
            val = top_values[i].item()
            
            if val == 0:
                continue
                
            # Get surrounding context for better interpretability
            start_idx = max(0, idx - context_window)
            end_idx = min(len(token_ids), idx + context_window + 1)
            
            context_tokens = token_ids[start_idx:end_idx]
            context_text = self.tokenizer.decode(context_tokens)
            target_token_text = self.tokenizer.decode([token_ids[idx]])
            
            results.append({
                "token_idx": idx,
                "activation_value": val,
                "target_token": target_token_text,
                "context_text": context_text
            })
            
        return results
