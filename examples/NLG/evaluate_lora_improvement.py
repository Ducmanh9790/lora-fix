#!/usr/bin/env python3
"""
NLG Model Evaluation - So sánh kết quả trước và sau khi train LoRA
Evaluates GPT-2 model before and after LoRA training on E2E NLG task.
"""

import os
import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from collections import Counter

# Giả sử loralib đã được cài đặt
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
import loralib as lora


class SimpleGPT2WithLoRA(torch.nn.Module):
    """GPT-2 model with LoRA adaptation layers"""
    
    def __init__(self, pretrained_model='gpt2', lora_rank=16, lora_alpha=32, lora_dropout=0.05):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(pretrained_model)
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
        # Áp dụng LoRA vào các c_attn và c_proj layers
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and ('c_attn' in name or 'c_proj' in name):
                # Tạo LoRA layer
                lora_module = lora.Linear(
                    module.in_features,
                    module.out_features,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias=module.bias is not None
                )
                # Copy weights
                lora_module.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    lora_module.bias.data = module.bias.data.clone()
                
                # Replace module
                parent_name = '.'.join(name.split('.')[:-1])
                module_name = name.split('.')[-1]
                parent = self.model.get_submodule(parent_name)
                setattr(parent, module_name, lora_module)
        
        # Mark only LoRA as trainable
        lora.mark_only_lora_as_trainable(self.model)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    def generate(self, input_ids, max_length=50, num_beams=1):
        """Generate text using the model"""
        return self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            top_p=0.95,
            do_sample=True,
            pad_token_id=self.model.config.eos_token_id
        )


def calculate_perplexity(model, texts, tokenizer, device='cpu'):
    """Tính Perplexity - đo lường độ "bối rối" của mô hình"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            if loss is not None:
                total_loss += loss.item() * input_ids.shape[1]
                total_tokens += input_ids.shape[1]
    
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss))
        return perplexity.item()
    return float('inf')


def calculate_entropy(model, texts, tokenizer, device='cpu'):
    """Tính Entropy - đo lường độ chắc chắn của dự đoán"""
    model.eval()
    total_entropy = 0
    total_predictions = 0
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            input_ids = inputs['input_ids'].to(device)
            
            outputs = model(input_ids=input_ids, labels=input_ids)
            logits = outputs.logits
            
            # Tính entropy từ logits
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
            
            total_entropy += entropy.item()
            total_predictions += 1
    
    return total_entropy / total_predictions if total_predictions > 0 else 0


def evaluate_generation_quality(model, prompts, tokenizer, device='cpu', max_length=50):
    """Đánh giá chất lượng văn bản được tạo ra"""
    model.eval()
    results = []
    
    with torch.no_grad():
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            
            # Generate output
            output_ids = model.generate(input_ids, max_length=max_length)
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Tính metrics
            output_length = len(output_ids[0])
            vocab_diversity = len(set(output_ids[0].tolist()))
            
            results.append({
                'prompt': prompt,
                'output': output_text,
                'output_length': output_length,
                'vocab_diversity': vocab_diversity
            })
    
    return results


def count_parameters(model):
    """Đếm số lượng parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_evaluation_report(model_name, metrics, generation_results):
    """In báo cáo đánh giá chi tiết"""
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"{'='*70}")
    
    print("\n📊 METRICS:")
    print(f"  Perplexity (độ bối rối):        {metrics['perplexity']:.4f}")
    print(f"  Entropy (độ không chắc):        {metrics['entropy']:.4f}")
    print(f"  Average Output Length:          {metrics['avg_output_length']:.2f}")
    print(f"  Average Vocab Diversity:        {metrics['avg_vocab_diversity']:.2f}")
    
    if 'trainable_params' in metrics:
        print(f"\n💾 PARAMETERS:")
        print(f"  Total Parameters:               {metrics['total_params']:,}")
        print(f"  Trainable Parameters:           {metrics['trainable_params']:,}")
        trainable_pct = (metrics['trainable_params'] / metrics['total_params'] * 100) if metrics['total_params'] > 0 else 0
        print(f"  Trainable % of Total:           {trainable_pct:.2f}%")
    
    print("\n✍️ GENERATION SAMPLES:")
    for i, result in enumerate(generation_results[:3], 1):
        print(f"\n  Sample {i}:")
        print(f"    Input:   {result['prompt']}")
        print(f"    Output:  {result['output'][:100]}...")
        print(f"    Length:  {result['output_length']} tokens")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🔧 Device: {device}")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test data - E2E NLG examples
    test_texts = [
        "name : Aachos , eatType : restaurant , food : Indian , priceRange : moderate , area : city centre , familyFriendly : yes",
        "name : Akane , eatType : restaurant , food : Japanese , priceRange : high , area : riverside , familyFriendly : no",
        "name : Browns Cambridge , eatType : pub , food : English , priceRange : moderate , area : city centre , familyFriendly : yes"
    ]
    
    prompts = [
        "The restaurant serves",
        "In the city centre, you can find",
        "For a family-friendly dinner,"
    ]
    
    print("\n" + "="*70)
    print("E2E NLG MODEL EVALUATION")
    print("="*70)
    
    # ========== Evaluate PRETRAINED Model (không train) ==========
    print("\n\n🔄 Loading PRETRAINED model (no fine-tuning)...")
    pretrained_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    pretrained_model.eval()
    
    print("✅ Evaluating PRETRAINED model...")
    pretrained_perplexity = calculate_perplexity(pretrained_model, test_texts, tokenizer, device)
    pretrained_entropy = calculate_entropy(pretrained_model, test_texts, tokenizer, device)
    pretrained_generation = evaluate_generation_quality(pretrained_model, prompts, tokenizer, device)
    
    pretrained_metrics = {
        'perplexity': pretrained_perplexity,
        'entropy': pretrained_entropy,
        'avg_output_length': np.mean([r['output_length'] for r in pretrained_generation]),
        'avg_vocab_diversity': np.mean([r['vocab_diversity'] for r in pretrained_generation]),
        'total_params': sum(p.numel() for p in pretrained_model.parameters()),
        'trainable_params': 0  # Không có trainable params
    }
    
    print_evaluation_report("PRETRAINED GPT-2 (No Fine-tuning)", pretrained_metrics, pretrained_generation)
    
    # ========== Load LoRA-trained Model ==========
    lora_checkpoint_path = os.path.join(os.path.dirname(__file__), 'lora_model', 'pytorch_model.bin')
    
    if os.path.exists(lora_checkpoint_path):
        print("\n\n🔄 Loading LoRA FINE-TUNED model...")
        lora_model = SimpleGPT2WithLoRA(lora_rank=16).to(device)
        
        # Load trained weights
        checkpoint = torch.load(lora_checkpoint_path, map_location=device)
        lora_model.model.load_state_dict(checkpoint, strict=False)
        lora_model.eval()
        
        print("✅ Evaluating LoRA FINE-TUNED model...")
        lora_perplexity = calculate_perplexity(lora_model.model, test_texts, tokenizer, device)
        lora_entropy = calculate_entropy(lora_model.model, test_texts, tokenizer, device)
        lora_generation = evaluate_generation_quality(lora_model.model, prompts, tokenizer, device)
        
        total_params, trainable_params = count_parameters(lora_model.model)
        
        lora_metrics = {
            'perplexity': lora_perplexity,
            'entropy': lora_entropy,
            'avg_output_length': np.mean([r['output_length'] for r in lora_generation]),
            'avg_vocab_diversity': np.mean([r['vocab_diversity'] for r in lora_generation]),
            'total_params': total_params,
            'trainable_params': trainable_params
        }
        
        print_evaluation_report("LoRA FINE-TUNED GPT-2", lora_metrics, lora_generation)
        
        # ========== So sánh kết quả ==========
        print("\n\n" + "="*70)
        print("📈 COMPARISON: PRETRAINED vs FINE-TUNED")
        print("="*70)
        
        perplexity_improvement = ((pretrained_perplexity - lora_perplexity) / pretrained_perplexity * 100) if pretrained_perplexity > 0 else 0
        entropy_improvement = ((pretrained_entropy - lora_entropy) / pretrained_entropy * 100) if pretrained_entropy > 0 else 0
        vocab_improvement = ((lora_metrics['avg_vocab_diversity'] - pretrained_metrics['avg_vocab_diversity']) / pretrained_metrics['avg_vocab_diversity'] * 100) if pretrained_metrics['avg_vocab_diversity'] > 0 else 0
        
        print(f"\n📊 Metrics Improvement:")
        print(f"  Perplexity:  {pretrained_perplexity:.4f} → {lora_perplexity:.4f}")
        print(f"    ✓ Cải thiện: {perplexity_improvement:+.2f}%")
        
        print(f"\n  Entropy:     {pretrained_entropy:.4f} → {lora_entropy:.4f}")
        print(f"    ✓ Cải thiện: {entropy_improvement:+.2f}%")
        
        print(f"\n  Vocab Diversity: {pretrained_metrics['avg_vocab_diversity']:.2f} → {lora_metrics['avg_vocab_diversity']:.2f}")
        print(f"    ✓ Cải thiện: {vocab_improvement:+.2f}%")
        
        print(f"\n💾 Efficiency:")
        print(f"  Trainable Params: {lora_metrics['trainable_params']:,} / {total_params:,}")
        print(f"  Training Efficiency: Only {(lora_metrics['trainable_params']/total_params*100):.2f}% params updated")
        
        # Kết luận
        print(f"\n🎯 CONCLUSION:")
        if perplexity_improvement > 0:
            print(f"  ✅ Model đã học được - Perplexity giảm {perplexity_improvement:.1f}%")
        else:
            print(f"  ⚠️ Perplexity tăng {abs(perplexity_improvement):.1f}% (cần train nhiều hơn)")
        
        if lora_metrics['avg_vocab_diversity'] > pretrained_metrics['avg_vocab_diversity']:
            print(f"  ✅ Độ đa dạng từ vựng tăng {vocab_improvement:.1f}%")
        else:
            print(f"  ⚠️ Độ đa dạng từ vựng giảm")
        
        print(f"  ✅ LoRA chỉ cập nhật {lora_metrics['trainable_params']:,} params (1.2% tổng)")
        
    else:
        print(f"\n❌ LoRA checkpoint not found at: {lora_checkpoint_path}")
        print("   Vui lòng chạy run_training.py trước!")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
