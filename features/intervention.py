import torch
from typing import List, Dict, Any, Optional

class InterventionHandler:
    """
    Implements feature clamping to test the causal influence of a 
    specific SAE feature on the model's output.
    """
    def __init__(self, model_wrapper: Any, sae: Any, layer_idx: int):
        self.model_wrapper = model_wrapper
        self.sae = sae
        self.layer_idx = layer_idx
        self.clamped_feature_idx = None
        self.clamped_value = 0.0

    def _clamping_hook(self, model, input, output):
        """
        Modifies the activations during the forward pass by increasing a 
        specific feature dimension.
        """
        # Output is usually [batch_size, seq_len, hidden_dim]
        # In Qwen2, output is a tuple (hidden_states, ...)
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None
            
        # Get feature activations for current hidden states
        # [batch, seq, input_dim] -> [batch, seq, hidden_dim]
        features = self.sae.encode(hidden_states)
        
        # Clamp the target feature to a high value (causal intervention)
        if self.clamped_feature_idx is not None:
            # Set target feature to a high value for ALL tokens
            features[:, :, self.clamped_feature_idx] = self.clamped_value
            
        # Reconstruct modified hidden states
        modified_hidden_states = self.sae.decode(features)
        
        # In case hidden_states were in half precision, ensure match
        modified_hidden_states = modified_hidden_states.to(hidden_states.dtype)
        
        if rest:
            return (modified_hidden_states,) + rest
        else:
            return modified_hidden_states

    @torch.no_grad()
    def run_intervention(
        self, 
        prompt: str, 
        feature_idx: int, 
        clamped_value: float = 10.0,
        max_new_tokens: int = 20
    ) -> Dict[str, str]:
        """
        Runs the model with and without clamping to compare outputs.
        """
        # Baseline output
        self.model_wrapper.remove_hooks()
        baseline_res = self.model_wrapper.generate_with_activations(prompt, max_new_tokens)
        baseline_text = baseline_res["text"]
        
        # Intervention output
        self.clamped_feature_idx = feature_idx
        self.clamped_value = clamped_value
        
        # Register the clamping hook
        layer = self.model_wrapper.model.model.layers[self.layer_idx]
        hook_handle = layer.register_forward_hook(self._clamping_hook)
        
        intervention_res = self.model_wrapper.generate_with_activations(prompt, max_new_tokens)
        intervention_text = intervention_res["text"]
        
        # Cleanup
        hook_handle.remove()
        self.clamped_feature_idx = None
        
        return {
            "prompt": prompt,
            "baseline": baseline_text,
            "intervention": intervention_text
        }
