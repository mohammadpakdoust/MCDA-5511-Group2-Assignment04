import torch
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

class FeatureAnalyzer:
    """
    Analyzes all features found by the SAE to identify those that are 
    semantically coherent and likely to correspond to interpretable concepts.
    """
    def __init__(self, extractor: Any, activations: torch.Tensor, token_ids: torch.Tensor):
        self.extractor = extractor
        self.activations = activations
        self.token_ids = token_ids

    def analyze_all_features(self, max_features: int = 100, k: int = 5) -> List[Dict[str, Any]]:
        """
        Iterates over the first max_features to find top activating examples 
        for each.
        """
        results = []
        num_features = self.extractor.sae.hidden_dim
        num_features = min(num_features, max_features)
        
        print(f"Analyzing top {num_features} features...")
        for i in tqdm(range(num_features)):
            # Get top k examples for each feature
            examples = self.extractor.get_top_k_activating_examples(
                self.activations, 
                self.token_ids, 
                feature_idx=i, 
                k=k
            )
            
            if examples:
                results.append({
                    "feature_idx": i,
                    "examples": examples,
                    "max_activation": examples[0]["activation_value"]
                })
        
        # Sort features by activation strength
        results.sort(key=lambda x: x["max_activation"], reverse=True)
        return results

    def display_feature(self, feature_info: Dict[str, Any]):
        """
        Prints a summary of the feature and its top activating text snippets.
        """
        print(f"--- Feature {feature_info['feature_idx']} ---")
        print(f"Max Activation: {feature_info['max_activation']:.4f}")
        for i, ex in enumerate(feature_info['examples']):
            print(f" {i+1}. Token: '{ex['target_token']}' Val: {ex['activation_value']:.4f}")
            print(f"    Context: ... {ex['context_text']} ...")
        print("-" * 20)
