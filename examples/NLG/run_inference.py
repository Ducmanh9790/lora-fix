#!/usr/bin/env python3
"""
Inference script - Load and use LoRA checkpoint
Demonstrates how to load a saved LoRA model for inference
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import loralib as lora
from run_training import SimpleGPT2WithLoRA

def load_lora_checkpoint(model, checkpoint_path, device):
    """Load LoRA weights from checkpoint"""
    print(f"📂 Loading LoRA checkpoint from: {checkpoint_path}")
    lora_state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(lora_state_dict, strict=False)
    print(f"✓ Checkpoint loaded successfully")
    return model

def main():
    # Hyperparameters (must match training)
    vocab_size = 50257
    hidden_dim = 768
    num_layers = 2
    lora_dim = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint_path = 'lora_model/pytorch_model.bin'
    
    print("=" * 80)
    print("GPT-2 with LoRA - INFERENCE DEMO")
    print("=" * 80)
    
    # Create model
    print(f"\n🤖 Creating model...")
    model = SimpleGPT2WithLoRA(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_lora=True,
        lora_dim=lora_dim
    ).to(device)
    
    # Load LoRA checkpoint
    print(f"\n💾 Loading checkpoint...")
    if os.path.exists(checkpoint_path):
        model = load_lora_checkpoint(model, checkpoint_path, device)
    else:
        print(f"⚠️  Checkpoint not found at {checkpoint_path}")
        print("   Using model with random LoRA weights")
    
    # Set to eval mode
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"  Frozen parameters: {total_params - trainable_params:,} ({100*(total_params-trainable_params)/total_params:.2f}%)")
    
    # Inference demo
    print(f"\n🔮 Running inference demo...")
    with torch.no_grad():
        # Create dummy input
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        
        # Forward pass
        logits = model(input_ids)
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Output logits shape: {logits.shape}")
        
        # Get predictions (argmax)
        predictions = torch.argmax(logits, dim=-1)
        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Sample predictions (first 10 tokens):")
        print(f"    {predictions[0, :10].cpu().numpy()}")
    
    # Merge LoRA weights (optional)
    print(f"\n🔗 Merging LoRA weights with base model...")
    model.eval()
    for module in model.modules():
        if isinstance(module, lora.Linear):
            if hasattr(module, 'merge_weights'):
                # Merge LoRA into the base weight
                if module.r > 0 and not module.merged:
                    def T(w):
                        return w.transpose(0, 1) if module.fan_in_fan_out else w
                    module.weight.data += T(module.lora_B @ module.lora_A) * module.scaling
                    module.merged = True
    
    print("✓ LoRA weights merged into base model")
    
    # Inference on merged model
    print(f"\n🚀 Running inference on merged model...")
    with torch.no_grad():
        logits_merged = model(input_ids)
        predictions_merged = torch.argmax(logits_merged, dim=-1)
        print(f"  Sample predictions (first 10 tokens):")
        print(f"    {predictions_merged[0, :10].cpu().numpy()}")
    
    print("\n" + "=" * 80)
    print("✨ Inference completed successfully!")
    print("=" * 80)

if __name__ == '__main__':
    main()
