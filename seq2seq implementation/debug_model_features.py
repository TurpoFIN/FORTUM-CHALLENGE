"""
Debug script to see exactly what features the model expects.
"""

import torch
import sys

model_path = sys.argv[1] if len(sys.argv) > 1 else "artifacts/models_monthly_a100/seq2seq_monthly_embeddings.pt"

print(f"Loading model from: {model_path}")
checkpoint = torch.load(model_path, map_location='cpu')

config = checkpoint['config']

print("\n=== MODEL CONFIGURATION ===")
print(f"Encoder input size: {config['model_kwargs']['enc_input_size']}")
print(f"Decoder known size: {config['model_kwargs']['dec_known_size']}")
print(f"Number of groups: {config['model_kwargs']['num_groups']}")
print(f"Group embedding dim: {config['model_kwargs']['group_emb_dim']}")

if 'categorical_config' in config['model_kwargs']:
    print(f"\nCategorical config: {config['model_kwargs']['categorical_config']}")

if 'categorical_encoders' in config:
    print(f"\nCategorical encoders:")
    for feat_name, encoder in config['categorical_encoders'].items():
        print(f"  {feat_name}: {len(encoder)} categories")

print(f"\nGroup to index mapping: {len(config['group_to_idx'])} groups")
print(f"Sample groups: {list(config['group_to_idx'].keys())[:5]}")

