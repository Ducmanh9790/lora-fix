#!/usr/bin/env python3
"""
LoRA Fine-tuning for RoBERTa on GLUE Tasks (NLU)
Simplified demo version
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import loralib as lora
from tqdm import tqdm
import argparse

print("✓ Imports successful")

# ============================================================================
# SIMPLE GLUE DATASET
# ============================================================================
class SimpleGLUEDataset(Dataset):
    """Simple dataset for GLUE tasks"""
    def __init__(self, task_name='sst2', split='train', max_samples=100):
        """
        task_name: 'sst2', 'mnli', 'qnli', etc.
        split: 'train', 'validation'
        """
        self.task = task_name
        self.split = split
        self.data = []
        
        # Create dummy data for demo
        if task_name == 'sst2':
            # Sentiment: text + label (0 or 1)
            self.data = [
                {'text': f'sample text {i}', 'label': i % 2}
                for i in range(max_samples)
            ]
        elif task_name == 'mnli':
            # Natural Language Inference: premise + hypothesis + label (0, 1, or 2)
            self.data = [
                {'premise': f'premise {i}', 'hypothesis': f'hypothesis {i}', 'label': i % 3}
                for i in range(max_samples)
            ]
        elif task_name == 'qnli':
            # Question NLI: question + sentence + label (0 or 1)
            self.data = [
                {'question': f'question {i}', 'sentence': f'sentence {i}', 'label': i % 2}
                for i in range(max_samples)
            ]
        else:
            # Generic binary classification
            self.data = [
                {'text': f'sample {i}', 'label': i % 2}
                for i in range(max_samples)
            ]
        
        print(f"  Loaded {len(self.data)} samples from {task_name} ({split})")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# ============================================================================
# SIMPLE ROBERTA WITH LORA
# ============================================================================
class SimpleRoBERTaWithLoRA(nn.Module):
    """Simplified RoBERTa-like model with LoRA"""
    def __init__(self, vocab_size=50265, hidden_dim=768, num_layers=12, num_labels=2, use_lora=True, lora_dim=16):
        super().__init__()
        self.use_lora = use_lora
        self.num_labels = num_labels
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if use_lora and lora_dim > 0:
                layer = nn.Sequential(
                    lora.Linear(hidden_dim, hidden_dim * 4, r=lora_dim),
                    nn.GELU(),
                    lora.Linear(hidden_dim * 4, hidden_dim, r=lora_dim),
                    nn.LayerNorm(hidden_dim),
                )
            else:
                layer = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                )
            self.layers.append(layer)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_labels)
        )
    
    def forward(self, input_ids):
        # Embedding
        x = self.embedding(input_ids)
        x = x.mean(dim=1)  # Simple pooling: average across tokens
        
        # Transformer layers
        for layer in self.layers:
            x = x + layer(x)  # Residual connection
        
        # Classification
        logits = self.classifier(x)
        return logits

# ============================================================================
# TRAINING FUNCTION
# ============================================================================
def train_epoch(model, train_loader, optimizer, num_labels, device, args):
    """Train one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training", disable=False)
    
    for batch in pbar:
        batch_size = len(batch)
        seq_len = 128
        
        # Create dummy input
        input_ids = torch.randint(0, 50265, (batch_size, seq_len)).to(device)
        
        # Create labels based on task
        if num_labels == 2:
            labels = torch.randint(0, 2, (batch_size,)).to(device)
        elif num_labels == 3:
            labels = torch.randint(0, 3, (batch_size,)).to(device)
        else:
            labels = torch.randint(0, num_labels, (batch_size,)).to(device)
        
        # Forward pass
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss

# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='RoBERTa with LoRA on GLUE')
    
    # Task args
    parser.add_argument('--task', type=str, default='sst2', 
                       choices=['sst2', 'mnli', 'qnli', 'mrpc', 'cola', 'rte', 'qqp', 'stsb'],
                       help='GLUE task name')
    
    # Model args
    parser.add_argument('--vocab_size', type=int, default=50265, help='RoBERTa vocab size')
    parser.add_argument('--hidden_dim', type=int, default=768, help='hidden dimension')
    parser.add_argument('--num_layers', type=int, default=12, help='number of transformer layers')
    parser.add_argument('--num_labels', type=int, default=2, help='number of output labels')
    
    # LoRA args
    parser.add_argument('--lora_dim', type=int, default=16, help='LoRA rank')
    parser.add_argument('--use_lora', action='store_true', default=True, help='use LoRA')
    
    # Training args
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=2, help='number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    
    # Data args
    parser.add_argument('--max_train_samples', type=int, default=200, help='max training samples')
    
    # Output args
    parser.add_argument('--output_dir', type=str, default='lora_nlu_model', help='output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Device
    device = torch.device(args.device)
    print(f"✓ Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine number of labels based on task
    task_num_labels = {
        'sst2': 2, 'mnli': 3, 'qnli': 2, 'mrpc': 2, 
        'cola': 2, 'rte': 2, 'qqp': 2, 'stsb': 1  # STSB is regression
    }
    num_labels = task_num_labels.get(args.task, 2)
    
    # Load data
    print(f"\n📂 Loading datasets for {args.task.upper()}...")
    train_data = SimpleGLUEDataset(task_name=args.task, split='train', max_samples=args.max_train_samples)
    val_data = SimpleGLUEDataset(task_name=args.task, split='validation', max_samples=args.max_train_samples // 4)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    print(f"\n🤖 Creating RoBERTa model with LoRA...")
    model = SimpleRoBERTaWithLoRA(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_labels=num_labels,
        use_lora=args.use_lora,
        lora_dim=args.lora_dim
    ).to(device)
    
    # Mark only LoRA as trainable
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
    print(f"\n🚀 Starting training on {args.task.upper()} task...")
    print("=" * 80)
    
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, num_labels, device, args)
        print(f"  Avg training loss: {train_loss:.4f}")
    
    print("\n" + "=" * 80)
    print("✓ Training completed!")
    
    # Save model
    print(f"\n💾 Saving LoRA checkpoint to {args.output_dir}...")
    lora_state_dict = lora.lora_state_dict(model)
    checkpoint_path = os.path.join(args.output_dir, f'{args.task}_pytorch_model.bin')
    torch.save(lora_state_dict, checkpoint_path)
    print(f"  Checkpoint saved: {checkpoint_path}")
    print(f"  Size: {os.path.getsize(checkpoint_path) / 1024 / 1024:.2f} MB")
    
    print("\n✨ Done!")

if __name__ == '__main__':
    main()
