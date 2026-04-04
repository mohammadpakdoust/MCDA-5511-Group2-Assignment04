"""
Scans all SAE features to find one that fires dominantly on numeric tokens.
Run with: uv run python find_numeric_feature.py
"""
import torch
from models.model_wrapper import ModelWrapper
from sae.sae_model import SparseAutoencoder
from features.feature_extractor import FeatureExtractor

def main():
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Load tokenizer
    model_wrapper = ModelWrapper(model_name="Qwen/Qwen2.5-1.5B", device=device)

    # Load activations
    data = torch.load("data/activations_layer_14.pt", map_location="cpu")
    activations = data["activations"]
    token_ids   = data["tokens"]

    # Load SAE
    state_dict = torch.load("sae/sae_layer_14.pt", map_location=device)
    input_dim  = state_dict["W_enc"].shape[0]  # [input_dim, feature_dim]
    hidden_dim = state_dict["W_enc"].shape[1]
    sae = SparseAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
    state_dict.pop("W_dec", None)
    sae.load_state_dict(state_dict)
    sae.to(device)

    extractor = FeatureExtractor(sae, model_wrapper.tokenizer)

    print("Scanning features for numeric token activation...\n")

    # Pre-compute all feature activations once
    with torch.no_grad():
        feat_acts = sae.get_feature_activations(
            activations.to(device, dtype=torch.float32)
        ).cpu()  # [num_tokens, feature_dim]

    numeric_hits = {}

    for feat_idx in range(hidden_dim):
        col = feat_acts[:, feat_idx]
        top_vals, top_idxs = torch.topk(col, 5)

        numeric_count = 0
        for i in range(5):
            if top_vals[i].item() == 0:
                break
            tok = model_wrapper.tokenizer.decode([token_ids[top_idxs[i].item()]])
            if any(ch.isdigit() for ch in tok):
                numeric_count += 1

        if numeric_count >= 2:            # at least 2 of top-5 tokens are numeric
            max_act = col.max().item()
            numeric_hits[feat_idx] = (numeric_count, max_act)

    if not numeric_hits:
        print("No clear numeric feature found in this model. The SAE may need more training data.")
    else:
        # Sort by numeric_count desc, then by max activation desc
        ranked = sorted(numeric_hits.items(), key=lambda x: (-x[1][0], -x[1][1]))
        print(f"Found {len(ranked)} candidate numeric feature(s):\n")
        for feat_idx, (cnt, max_act) in ranked[:5]:
            print(f"  Feature {feat_idx}: {cnt}/5 top tokens are numeric | max activation = {max_act:.4f}")
            top_vals, top_idxs = torch.topk(feat_acts[:, feat_idx], 5)
            for i in range(5):
                tok = model_wrapper.tokenizer.decode([token_ids[top_idxs[i].item()]])
                print(f"    Token: '{tok}' | val: {top_vals[i].item():.4f}")
            print()

        best_feat = ranked[0][0]
        print(f">>> Best numeric feature index to use in main.py: {best_feat}")

if __name__ == "__main__":
    main()
