#!/usr/bin/env python3
"""
Comparison script - LoRA vs Full Fine-tuning
Shows the difference in parameters and model sizes
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import loralib as lora
from run_training import SimpleGPT2WithLoRA

def create_model_without_lora(vocab_size=50257, hidden_dim=768, num_layers=2):
    """Create standard model without LoRA"""
    layers = nn.ModuleList()
    for i in range(num_layers):
        layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        layers.append(layer)
    
    model = nn.Sequential(
        nn.Embedding(vocab_size, hidden_dim),
        *layers,
        nn.Linear(hidden_dim, vocab_size)
    )
    return model

def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return total, trainable, frozen

def get_model_size_mb(model):
    """Calculate model size in MB"""
    param_size = sum(p.numel() * 4 for p in model.parameters()) / (1024 * 1024)  # 4 bytes per float32
    return param_size

def main():
    print("=" * 100)
    print("📊 COMPARISON: LoRA vs Full Fine-tuning")
    print("=" * 100)
    
    vocab_size = 50257
    hidden_dim = 768
    num_layers = 2
    lora_dim = 16
    device = torch.device('cpu')
    
    # ============================================================================
    # FULL FINE-TUNING MODEL
    # ============================================================================
    print("\n📌 SCENARIO 1: Full Fine-tuning (No LoRA)")
    print("-" * 100)
    
    model_full = create_model_without_lora(vocab_size, hidden_dim, num_layers).to(device)
    
    # All parameters trainable
    for p in model_full.parameters():
        p.requires_grad = True
    
    total_full, trainable_full, frozen_full = count_parameters(model_full)
    size_full = get_model_size_mb(model_full)
    
    print(f"Total Parameters:       {total_full:>15,}  ({size_full:>8.2f} MB)")
    print(f"Trainable Parameters:   {trainable_full:>15,}  ({100*trainable_full/total_full:>7.2f}%)")
    print(f"Frozen Parameters:      {frozen_full:>15,}  ({100*frozen_full/total_full:>7.2f}%)")
    print(f"Model Size (weights):   {size_full:>8.2f} MB")
    print(f"GPU Memory needed:      ~{size_full * 2:.2f} MB (with gradients)")
    
    # ============================================================================
    # LORA MODEL
    # ============================================================================
    print("\n🎯 SCENARIO 2: LoRA Fine-tuning")
    print("-" * 100)
    
    model_lora = SimpleGPT2WithLoRA(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_lora=True,
        lora_dim=lora_dim
    ).to(device)
    
    # Mark only LoRA as trainable
    lora.mark_only_lora_as_trainable(model_lora)
    
    total_lora, trainable_lora, frozen_lora = count_parameters(model_lora)
    size_lora = get_model_size_mb(model_lora)
    lora_checkpoint_size = trainable_lora * 4 / (1024 * 1024)  # Only save trainable params
    
    print(f"Total Parameters:       {total_lora:>15,}  ({size_lora:>8.2f} MB)")
    print(f"Trainable Parameters:   {trainable_lora:>15,}  ({100*trainable_lora/total_lora:>7.2f}%)")
    print(f"Frozen Parameters:      {frozen_lora:>15,}  ({100*frozen_lora/total_lora:>7.2f}%)")
    print(f"Model Size (full):      {size_lora:>8.2f} MB")
    print(f"Checkpoint Size (LoRA): {lora_checkpoint_size:>8.2f} MB")
    print(f"GPU Memory needed:      ~{lora_checkpoint_size * 2:.2f} MB (with gradients)")
    
    # ============================================================================
    # COMPARISON
    # ============================================================================
    print("\n📈 COMPARISON & BENEFITS")
    print("-" * 100)
    
    param_reduction = (1 - trainable_lora / trainable_full) * 100
    size_reduction = (1 - lora_checkpoint_size / size_full) * 100
    memory_reduction = (1 - lora_checkpoint_size * 2 / (size_full * 2)) * 100
    
    print(f"\n✨ Parameter Reduction:")
    print(f"   Full Fine-tune trainable:  {trainable_full:>15,}")
    print(f"   LoRA trainable:            {trainable_lora:>15,}")
    print(f"   → Reduction: {param_reduction:>7.2f}%  ({trainable_full/trainable_lora:>7.2f}x smaller)")
    
    print(f"\n💾 Checkpoint Size Reduction:")
    print(f"   Full model size:           {size_full:>8.2f} MB")
    print(f"   LoRA checkpoint:           {lora_checkpoint_size:>8.2f} MB")
    print(f"   → Reduction: {size_reduction:>7.2f}%  ({size_full/lora_checkpoint_size:>7.2f}x smaller)")
    
    print(f"\n⚡ GPU Memory Usage Reduction:")
    print(f"   Full Fine-tune:            ~{size_full * 2:>8.2f} MB")
    print(f"   LoRA Fine-tune:            ~{lora_checkpoint_size * 2:>8.2f} MB")
    print(f"   → Reduction: {memory_reduction:>7.2f}%")
    
    # ============================================================================
    # KEY INSIGHTS
    # ============================================================================
    print("\n🔑 KEY INSIGHTS")
    print("-" * 100)
    
    print(f"""
1. PARAMETER EFFICIENCY
   • LoRA reduces trainable parameters by {param_reduction:.2f}%
   • Only {100*trainable_lora/total_lora:.2f}% of total parameters need to be trained
   • Base model (weights) remains frozen

2. STORAGE EFFICIENCY
   • LoRA checkpoints are {size_reduction:.2f}% smaller
   • Can easily store multiple LoRA adapters for different tasks
   • Each task-specific adapter is only ~{lora_checkpoint_size:.2f} MB

3. COMPUTATIONAL EFFICIENCY
   • Training is faster due to fewer parameters
   • GPU memory usage reduced by ~{memory_reduction:.2f}%
   • Enables training on smaller GPUs or larger batch sizes

4. PRACTICAL BENEFITS
   • Can adapt {int(size_full/lora_checkpoint_size):.0f}+ different LoRA adapters in same storage as 1 full model
   • Achieves comparable performance to full fine-tuning
   • Supports rapid task-switching during deployment

5. IDEAL USE CASES
   • Fine-tuning large pre-trained models (GPT, RoBERTa, DeBERTa)
   • Multi-task learning with shared base model
   • Resource-constrained environments
   • Rapid iteration and experimentation
    """)
    
    print("=" * 100)
    print("✅ Comparison completed!")
    print("=" * 100)

if __name__ == '__main__':
    main()
