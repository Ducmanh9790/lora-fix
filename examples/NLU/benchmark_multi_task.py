#!/usr/bin/env python3
"""
Multi-task Training with LoRA
Train RoBERTa on multiple GLUE tasks and compare efficiency
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

import torch
import torch.nn as nn
import loralib as lora
from run_training_nlu import SimpleRoBERTaWithLoRA
import time

print("✓ Imports successful")

def benchmark_task(task_name, device='cpu'):
    """Train and evaluate on a single task"""
    print(f"\n{'='*80}")
    print(f"Task: {task_name.upper()}")
    print(f"{'='*80}")
    
    task_num_labels = {
        'sst2': 2, 'mnli': 3, 'qnli': 2, 'mrpc': 2, 
        'cola': 2, 'rte': 2, 'qqp': 2, 'stsb': 1
    }
    num_labels = task_num_labels.get(task_name, 2)
    
    # Create model
    model = SimpleRoBERTaWithLoRA(
        vocab_size=50265,
        hidden_dim=768,
        num_layers=12,
        num_labels=num_labels,
        use_lora=True,
        lora_dim=16
    ).to(device)
    
    # Mark LoRA as trainable
    lora.mark_only_lora_as_trainable(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Create dummy batch
    batch_size = 8
    seq_len = 128
    input_ids = torch.randint(0, 50265, (batch_size, seq_len)).to(device)
    labels = torch.randint(0, num_labels, (batch_size,)).to(device)
    
    # Forward & backward timing
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    
    start_time = time.time()
    
    for step in range(5):  # 5 steps for timing
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_time = time.time() - start_time
    throughput = (batch_size * 5) / train_time  # samples per second
    
    # Checkpoint size
    lora_state = lora.lora_state_dict(model)
    checkpoint_size = sum(p.numel() * 4 for p in lora_state.values()) / (1024 * 1024)
    
    print(f"\n📊 Task Statistics:")
    print(f"  Total parameters:     {total_params:>15,}")
    print(f"  Trainable (LoRA):     {trainable_params:>15,} ({100*trainable_params/total_params:>6.2f}%)")
    print(f"  Checkpoint size:      {checkpoint_size:>15.2f} MB")
    print(f"  Training time (5 steps): {train_time:>10.2f}s")
    print(f"  Throughput:           {throughput:>15.2f} samples/sec")
    
    return {
        'task': task_name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'checkpoint_size': checkpoint_size,
        'train_time': train_time,
        'throughput': throughput
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # GLUE tasks
    tasks = ['sst2', 'mnli', 'qnli', 'mrpc', 'cola', 'rte']
    
    print("🚀 Multi-task LoRA Benchmarking")
    print("=" * 80)
    
    results = []
    total_checkpoint_size = 0
    
    for task in tasks:
        result = benchmark_task(task, device)
        results.append(result)
        total_checkpoint_size += result['checkpoint_size']
    
    # Summary
    print(f"\n{'='*80}")
    print("📈 SUMMARY: All 6 GLUE Tasks with LoRA")
    print(f"{'='*80}")
    
    print(f"\n{'Task':<12} {'Trainable %':<15} {'Checkpoint':<18} {'Throughput':<18}")
    print("-" * 65)
    
    for result in results:
        task = result['task'].upper()
        trainable_pct = 100 * result['trainable_params'] / result['total_params']
        checkpoint = f"{result['checkpoint_size']:.2f} MB"
        throughput = f"{result['throughput']:.1f} samples/s"
        print(f"{task:<12} {trainable_pct:>13.2f}% {checkpoint:>18} {throughput:>18}")
    
    print("\n" + "=" * 80)
    print("🎯 Key Results:")
    print(f"  Total checkpoint size (6 tasks): {total_checkpoint_size:.2f} MB")
    print(f"  vs Full model for 1 task:        ~330 MB")
    print(f"  vs Full models for 6 tasks:      ~1980 MB")
    print(f"  Storage saved:                   {1980 - total_checkpoint_size:.2f} MB ({100*(1980-total_checkpoint_size)/1980:.1f}%)")
    print(f"\n  Can store {int(1980/total_checkpoint_size):.0f}x more task adapters!")
    print("=" * 80)
    
    print("\n✨ Multi-task benchmarking completed!")

if __name__ == '__main__':
    main()
