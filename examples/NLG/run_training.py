#!/usr/bin/env python3
"""
Simple training script for GPT-2 with LoRA on E2E dataset
This is a demo version with minimal setup
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import loralib as lora
from tqdm import tqdm

print("✓ Imports successful")

# ============================================================================
# DATASET
# ============================================================================
class SimpleTextDataset(Dataset):
    """Simple dataset for E2E NLG"""
    def __init__(self, filepath, max_samples=None):
        self.data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                self.data.append(line.strip())
        print(f"  Loaded {len(self.data)} samples from {filepath}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# ============================================================================
# SIMPLE GPT2 MODEL WITH LORA
# ============================================================================
class SimpleGPT2WithLoRA(nn.Module):
    """Minimal GPT2-like model with LoRA for demo"""
    def __init__(self, vocab_size=50257, hidden_dim=768, num_layers=2, use_lora=True, lora_dim=16):
        super().__init__()
        self.use_lora = use_lora
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Create transformer layers with LoRA
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if use_lora and lora_dim > 0:
                layer = nn.Sequential(
                    lora.Linear(hidden_dim, hidden_dim * 4, r=lora_dim),
                    nn.GELU(),
                    lora.Linear(hidden_dim * 4, hidden_dim, r=lora_dim),
                )
            else:
                layer = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                )
            self.layers.append(layer)
        
        # Output projection
        self.output_proj = lora.Linear(hidden_dim, vocab_size, r=lora_dim) if (use_lora and lora_dim > 0) else nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = x + layer(x)  # Residual connection
        logits = self.output_proj(x)
        return logits

# ============================================================================
# TRAINING FUNCTION
# ============================================================================
def train_epoch(model, train_loader, optimizer, device, args):
    """Train one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training", disable=args.rank != 0 if hasattr(args, 'rank') else False)
    
    for batch_idx, texts in enumerate(pbar):
        # Create dummy input (in real scenario, tokenize the text)
        batch_size = len(texts)
        seq_len = args.seq_len
        
        # For demo: create random token IDs
        input_ids = torch.randint(0, 50257, (batch_size, seq_len)).to(device)
        
        # Forward pass
        logits = model(input_ids)
        
        # Simple loss (cross-entropy)
        target_ids = torch.randint(0, 50257, (batch_size, seq_len)).to(device)
        loss = nn.functional.cross_entropy(logits.reshape(-1, 50257), target_ids.reshape(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % args.log_interval == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss

# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='GPT-2 with LoRA Training')
    
    # Model args
    parser.add_argument('--vocab_size', type=int, default=50257, help='vocabulary size')
    parser.add_argument('--hidden_dim', type=int, default=768, help='hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers')
    
    # LoRA args
    parser.add_argument('--lora_dim', type=int, default=16, help='LoRA rank dimension')
    parser.add_argument('--use_lora', action='store_true', default=True, help='use LoRA')
    
    # Training args
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--seq_len', type=int, default=64, help='sequence length')
    parser.add_argument('--log_interval', type=int, default=10, help='log interval')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    # Data args
    parser.add_argument('--train_file', type=str, default='data/e2e/train.txt', help='training data file')
    parser.add_argument('--valid_file', type=str, default='data/e2e/valid.txt', help='validation data file')
    parser.add_argument('--max_train_samples', type=int, default=100, help='max training samples (for demo)')
    
    # Output args
    parser.add_argument('--output_dir', type=str, default='lora_model', help='output directory')
    parser.add_argument('--save_model', action='store_true', default=True, help='save model')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device')
    parser.add_argument('--rank', type=int, default=0, help='rank for distributed training')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device(args.device)
    print(f"✓ Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("\n📂 Loading datasets...")
    train_data = SimpleTextDataset(args.train_file, max_samples=args.max_train_samples)
    valid_data = SimpleTextDataset(args.valid_file, max_samples=args.max_train_samples // 4)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    print("\n🤖 Creating model with LoRA...")
    model = SimpleGPT2WithLoRA(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_lora=args.use_lora,
        lora_dim=args.lora_dim
    ).to(device)
    
    # Mark only LoRA parameters as trainable
    print("📌 Marking only LoRA parameters as trainable...")
    lora.mark_only_lora_as_trainable(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"  Frozen parameters: {total_params - trainable_params:,} ({100*(total_params-trainable_params)/total_params:.2f}%)")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Training loop
    print("\n🚀 Starting training...")
    print("=" * 80)
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device, args)
        print(f"  Avg training loss: {train_loss:.4f}")
    
    print("\n" + "=" * 80)
    print("✓ Training completed!")
    
    # Save model
    if args.save_model:
        print(f"\n💾 Saving LoRA checkpoint to {args.output_dir}...")
        lora_state_dict = lora.lora_state_dict(model)
        checkpoint_path = os.path.join(args.output_dir, 'pytorch_model.bin')
        torch.save(lora_state_dict, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")
        print(f"  Size: {os.path.getsize(checkpoint_path) / 1024 / 1024:.2f} MB")
    
    print("\n✨ Done!")

if __name__ == '__main__':
    main()
