import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple, Any

class ModelWrapper:
    """
    A wrapper for loading Qwen weights and hooking into the residual stream 
    to extract activations for Sparse Autoencoder training and analysis.
    """
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B", device: str = None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"Loading model {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Use float16/bfloat16 for efficiency on MPS/CUDA
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map=self.device
        )
        self.model.eval()
        self.activations = {}
        self.hooks = []

    def _get_activation_hook(self, name: str):
        """Standard hook to capture activations."""
        def hook(model, input, output):
            # For residual stream, output is usually a tuple (hidden_states, ...)
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach()
            else:
                self.activations[name] = output.detach()
        return hook

    def register_layer_hook(self, layer_idx: int):
        """
        Registers a hook on the residual stream output of a specific layer.
        In Qwen2, the layer output includes the residual connection.
        """
        if layer_idx < 0 or layer_idx >= len(self.model.model.layers):
            raise ValueError(f"Invalid layer index {layer_idx}")
            
        layer = self.model.model.layers[layer_idx]
        hook_handle = layer.register_forward_hook(self._get_activation_hook(f"layer_{layer_idx}"))
        self.hooks.append(hook_handle)
        print(f"Registered hook for layer {layer_idx}")

    def remove_hooks(self):
        """Clears all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}

    @torch.no_grad()
    def generate_with_activations(self, prompt: str, max_new_tokens: int = 50) -> Dict[str, Any]:
        """
        Generates text and captures activations collected via hooks.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # We capture activations for the input tokens specifically
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_hidden_states=True
        )
        
        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        return {
            "text": generated_text,
            "tokens": outputs.sequences[0],
            "activations": self.activations
        }

    @torch.no_grad()
    def get_activations(self, text: str) -> torch.Tensor:
        """
        Runs a forward pass and returns the captured activations for the input text.
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        self.model(**inputs)
        # Assuming only one hook is registered for now
        if not self.activations:
            raise RuntimeError("No activations captured. Did you register a hook?")
        
        # Returns [batch, seq_len, hidden_dim]
        return list(self.activations.values())[0]
