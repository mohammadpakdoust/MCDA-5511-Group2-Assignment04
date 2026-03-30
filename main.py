import torch
import os
import argparse
from models.model_wrapper import ModelWrapper
from data.dataset_generator import DatasetGenerator
from sae.sae_model import SparseAutoencoder
from sae.train_sae import SAETrainer
from features.feature_extractor import FeatureExtractor
from features.feature_analyzer import FeatureAnalyzer
from features.intervention import InterventionHandler
from utils.helpers import ensure_dirs, plot_training_curves

def parse_args():
    parser = argparse.ArgumentParser(description="SAE Mechanistic Interpretability Pipeline")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--layer_idx", type=int, default=14)
    parser.add_argument("--expansion_factor", type=int, default=4)
    parser.add_argument("--l1_lambda", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--train_only", action="store_true")
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--intervention_only", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    ensure_dirs()
    
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Model
    model_wrapper = ModelWrapper(model_name=args.model_name, device=device)
    hidden_dim = model_wrapper.model.config.hidden_size
    feature_dim = hidden_dim * args.expansion_factor
    
    # 2. Data Generation
    dataset_path = f"data/activations_layer_{args.layer_idx}.pt"
    if not os.path.exists(dataset_path):
        generator = DatasetGenerator(model_wrapper, layer_idx=args.layer_idx)
        generator.generate_from_dataset(max_samples=args.max_samples)
        generator.save_dataset(dataset_path)
    
    activations, token_ids = torch.load(dataset_path)["activations"], torch.load(dataset_path)["tokens"]
    
    # 3. SAE Model & Training
    sae = SparseAutoencoder(input_dim=hidden_dim, hidden_dim=feature_dim, l1_lambda=args.l1_lambda)
    sae_path = f"sae/sae_layer_{args.layer_idx}.pt"
    
    if not os.path.exists(sae_path) or args.train_only:
        trainer = SAETrainer(sae, lr=1e-4, batch_size=args.batch_size, device=device)
        trainer.train(activations, epochs=args.epochs)
        trainer.save_model(sae_path)
        plot_training_curves(trainer.history)
    else:
        sae.load_state_dict(torch.load(sae_path, map_location=device))
        sae.to(device)
    
    # 4. Feature Extraction & Analysis
    extractor = FeatureExtractor(sae, model_wrapper.tokenizer)
    analyzer = FeatureAnalyzer(extractor, activations, token_ids)
    
    if not args.intervention_only:
        top_features = analyzer.analyze_all_features(max_features=5, k=5)
        for feat in top_features:
            analyzer.display_feature(feat)
    
    # 5. Causal Intervention
    if not args.train_only:
        print("\n--- Running Causal Intervention ---")
        intervention = InterventionHandler(model_wrapper, sae, args.layer_idx)
        
        # Test finding: Try feature 0 (just as an example, in practice pick an interpretable one)
        test_prompt = "The capital of France is"
        res = intervention.run_intervention(test_prompt, feature_idx=0, clamped_value=20.0)
        
        print(f"Prompt: {res['prompt']}")
        print(f"Baseline: {res['baseline']}")
        print(f"Intervention: {res['intervention']}")

if __name__ == "__main__":
    main()
