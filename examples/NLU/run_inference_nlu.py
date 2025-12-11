#!/usr/bin/env python3
"""
LoRA Inference for RoBERTa on GLUE Tasks (NLU)
Load checkpoint and run inference
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

import torch
import torch.nn as nn
import loralib as lora
from run_training_nlu import SimpleRoBERTaWithLoRA
import argparse

print("✓ Imports successful")

def load_lora_checkpoint(model, checkpoint_path, device):
    """Load LoRA weights from checkpoint"""
    print(f"📂 Loading LoRA checkpoint from: {checkpoint_path}")
    lora_state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(lora_state_dict, strict=False)
    print(f"✓ Checkpoint loaded successfully")
    return model

def main():
    parser = argparse.ArgumentParser(description='RoBERTa LoRA Inference')
    parser.add_argument('--task', type=str, default='sst2', help='GLUE task')
    parser.add_argument('--checkpoint_dir', type=str, default='lora_nlu_model', help='checkpoint directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.task}_pytorch_model.bin')
    
    task_num_labels = {
        'sst2': 2, 'mnli': 3, 'qnli': 2, 'mrpc': 2, 
        'cola': 2, 'rte': 2, 'qqp': 2, 'stsb': 1
    }
    num_labels = task_num_labels.get(args.task, 2)
    
    print("=" * 80)
    print("RoBERTa with LoRA - GLUE INFERENCE DEMO")
    print("=" * 80)
    
    # Create model
    print(f"\n🤖 Creating model for {args.task.upper()} task...")
    model = SimpleRoBERTaWithLoRA(
        vocab_size=50265,
        hidden_dim=768,
        num_layers=12,
        num_labels=num_labels,
        use_lora=True,
        lora_dim=16
    ).to(device)
    
    # Load checkpoint
    print(f"\n💾 Loading checkpoint...")
    if os.path.exists(checkpoint_path):
        model = load_lora_checkpoint(model, checkpoint_path, device)
    else:
        print(f"⚠️  Checkpoint not found at {checkpoint_path}")
        print("   Using model with random LoRA weights")
    
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"  Frozen parameters: {total_params - trainable_params:,} ({100*(total_params-trainable_params)/total_params:.2f}%)")
    
    # Inference demo
    print(f"\n🔮 Running inference on {args.task.upper()} task...")
    with torch.no_grad():
        # Create dummy input
        batch_size = 4
        seq_len = 128
        input_ids = torch.randint(0, 50265, (batch_size, seq_len)).to(device)
        
        # Forward pass
        logits = model(input_ids)
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Output logits shape: {logits.shape}")
        
        # Get predictions
        predictions = torch.argmax(logits, dim=-1)
        print(f"  Predictions: {predictions.cpu().numpy()}")
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1)
        print(f"  Max probability: {probs.max().item():.4f}")
        print(f"  Min probability: {probs.min().item():.4f}")
    
    print("\n" + "=" * 80)
    print("✨ Inference completed successfully!")
    print("=" * 80)

if __name__ == '__main__':
    main()
