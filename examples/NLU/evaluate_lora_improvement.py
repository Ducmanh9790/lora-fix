#!/usr/bin/env python3
"""
NLU Model Evaluation - So sánh kết quả trước và sau khi train LoRA
Evaluates RoBERTa model before and after LoRA training on GLUE tasks.
"""

import os
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
import loralib as lora


class SimpleRoBERTaWithLoRA(torch.nn.Module):
    """RoBERTa model with LoRA adaptation layers"""
    
    def __init__(self, num_labels=2, lora_rank=16, lora_alpha=32, lora_dropout=0.05):
        super().__init__()
        self.model = RobertaForSequenceClassification.from_pretrained(
            'roberta-base',
            num_labels=num_labels
        )
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.num_labels = num_labels
        
        # Apply LoRA to attention and dense layers
        lora.mark_only_lora_as_trainable(self.model)
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and ('attention' in name or 'dense' in name):
                if not hasattr(module, 'lora_a'):
                    lora_module = lora.Linear(
                        module.in_features,
                        module.out_features,
                        r=lora_rank,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        bias=module.bias is not None
                    )
                    lora_module.weight.data = module.weight.data.clone()
                    if module.bias is not None:
                        lora_module.bias.data = module.bias.data.clone()
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


def evaluate_classification_metrics(model, texts, labels, tokenizer, device='cpu', task='sst2'):
    """Đánh giá metrics phân loại - Accuracy, F1, Precision, Recall"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1).item()
            predictions.append(pred)
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Tính các metrics
    accuracy = accuracy_score(labels, predictions)
    
    # Cho các task multi-class
    if len(np.unique(labels)) > 2:
        f1 = f1_score(labels, predictions, average='weighted')
        precision = precision_score(labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    else:
        f1 = f1_score(labels, predictions, average='binary')
        precision = precision_score(labels, predictions, average='binary', zero_division=0)
        recall = recall_score(labels, predictions, average='binary', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'predictions': predictions
    }


def calculate_model_confidence(model, texts, tokenizer, device='cpu'):
    """Tính độ tự tin của mô hình trong dự đoán"""
    model.eval()
    confidences = []
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            max_prob = torch.max(probs).item()
            confidences.append(max_prob)
    
    return np.mean(confidences), np.std(confidences)


def count_parameters(model):
    """Đếm số lượng parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_evaluation_report(model_name, metrics, task='sst2'):
    """In báo cáo đánh giá chi tiết"""
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name} (Task: {task.upper()})")
    print(f"{'='*70}")
    
    print("\n📊 CLASSIFICATION METRICS:")
    print(f"  Accuracy:          {metrics['accuracy']*100:.2f}%")
    print(f"  F1 Score:          {metrics['f1']:.4f}")
    print(f"  Precision:         {metrics['precision']:.4f}")
    print(f"  Recall:            {metrics['recall']:.4f}")
    
    if 'confidence_mean' in metrics:
        print(f"\n🎯 MODEL CONFIDENCE:")
        print(f"  Mean Confidence:   {metrics['confidence_mean']*100:.2f}%")
        print(f"  Std Deviation:     {metrics['confidence_std']*100:.2f}%")
    
    if 'trainable_params' in metrics:
        print(f"\n💾 PARAMETERS:")
        print(f"  Total Parameters:  {metrics['total_params']:,}")
        print(f"  Trainable Params:  {metrics['trainable_params']:,}")
        trainable_pct = (metrics['trainable_params'] / metrics['total_params'] * 100) if metrics['total_params'] > 0 else 0
        print(f"  Trainable %:       {trainable_pct:.2f}%")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🔧 Device: {device}")
    
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Test data - SST-2 examples (Sentiment Analysis)
    test_texts = [
        "This movie was absolutely wonderful and I loved every minute of it.",
        "Terrible film, a complete waste of time.",
        "It was okay, nothing special but not bad either.",
        "Best movie I've ever seen, highly recommend!",
        "Boring and predictable, disappointed with this one."
    ]
    
    # True labels: 1 = positive, 0 = negative
    test_labels = [1, 0, 0, 1, 0]
    
    print("\n" + "="*70)
    print("SST-2 SENTIMENT CLASSIFICATION MODEL EVALUATION")
    print("="*70)
    
    # ========== Evaluate PRETRAINED Model ==========
    print("\n\n🔄 Loading PRETRAINED RoBERTa (no fine-tuning)...")
    pretrained_model = RobertaForSequenceClassification.from_pretrained(
        'roberta-base',
        num_labels=2
    ).to(device)
    pretrained_model.eval()
    
    print("✅ Evaluating PRETRAINED model...")
    pretrained_metrics = evaluate_classification_metrics(
        pretrained_model, test_texts, test_labels, tokenizer, device, 'sst2'
    )
    pretrained_conf_mean, pretrained_conf_std = calculate_model_confidence(
        pretrained_model, test_texts, tokenizer, device
    )
    pretrained_metrics['confidence_mean'] = pretrained_conf_mean
    pretrained_metrics['confidence_std'] = pretrained_conf_std
    pretrained_metrics['total_params'] = sum(p.numel() for p in pretrained_model.parameters())
    pretrained_metrics['trainable_params'] = 0
    
    print_evaluation_report("PRETRAINED RoBERTa (No Fine-tuning)", pretrained_metrics)
    
    # ========== Load LoRA-trained Model ==========
    lora_checkpoint_path = os.path.join(os.path.dirname(__file__), 'lora_nlu_model', 'sst2_pytorch_model.bin')
    
    if os.path.exists(lora_checkpoint_path):
        print("\n\n🔄 Loading LoRA FINE-TUNED model...")
        lora_model = SimpleRoBERTaWithLoRA(num_labels=2, lora_rank=16).to(device)
        
        # Load trained weights
        checkpoint = torch.load(lora_checkpoint_path, map_location=device)
        lora_model.model.load_state_dict(checkpoint, strict=False)
        lora_model.eval()
        
        print("✅ Evaluating LoRA FINE-TUNED model...")
        lora_metrics = evaluate_classification_metrics(
            lora_model.model, test_texts, test_labels, tokenizer, device, 'sst2'
        )
        lora_conf_mean, lora_conf_std = calculate_model_confidence(
            lora_model.model, test_texts, tokenizer, device
        )
        lora_metrics['confidence_mean'] = lora_conf_mean
        lora_metrics['confidence_std'] = lora_conf_std
        
        total_params, trainable_params = count_parameters(lora_model.model)
        lora_metrics['total_params'] = total_params
        lora_metrics['trainable_params'] = trainable_params
        
        print_evaluation_report("LoRA FINE-TUNED RoBERTa", lora_metrics)
        
        # ========== So sánh kết quả ==========
        print("\n\n" + "="*70)
        print("📈 COMPARISON: PRETRAINED vs FINE-TUNED")
        print("="*70)
        
        accuracy_improvement = (lora_metrics['accuracy'] - pretrained_metrics['accuracy']) * 100
        f1_improvement = (lora_metrics['f1'] - pretrained_metrics['f1'])
        confidence_improvement = (lora_metrics['confidence_mean'] - pretrained_metrics['confidence_mean']) * 100
        
        print(f"\n📊 Metrics Improvement:")
        print(f"  Accuracy:        {pretrained_metrics['accuracy']*100:.2f}% → {lora_metrics['accuracy']*100:.2f}%")
        print(f"    ✓ Cải thiện:   {accuracy_improvement:+.2f} percentage points")
        
        print(f"\n  F1 Score:        {pretrained_metrics['f1']:.4f} → {lora_metrics['f1']:.4f}")
        print(f"    ✓ Cải thiện:   {f1_improvement:+.4f}")
        
        print(f"\n  Precision:       {pretrained_metrics['precision']:.4f} → {lora_metrics['precision']:.4f}")
        print(f"\n  Recall:          {pretrained_metrics['recall']:.4f} → {lora_metrics['recall']:.4f}")
        
        print(f"\n🎯 Confidence:")
        print(f"  Mean:            {pretrained_metrics['confidence_mean']*100:.2f}% → {lora_metrics['confidence_mean']*100:.2f}%")
        print(f"    ✓ Cải thiện:   {confidence_improvement:+.2f} percentage points")
        
        print(f"\n💾 Efficiency:")
        print(f"  Trainable Params: {lora_metrics['trainable_params']:,} / {total_params:,}")
        print(f"  Training Efficiency: Only {(lora_metrics['trainable_params']/total_params*100):.2f}% params updated")
        
        # Kết luận
        print(f"\n🎯 CONCLUSION:")
        if accuracy_improvement > 0:
            print(f"  ✅ Model đã học được - Accuracy tăng {accuracy_improvement:.2f}%")
        else:
            print(f"  ⚠️ Accuracy giảm {abs(accuracy_improvement):.2f}% (cần train nhiều hơn)")
        
        if confidence_improvement > 0:
            print(f"  ✅ Mô hình tự tin hơn trong dự đoán - {confidence_improvement:+.2f}%")
        else:
            print(f"  ⚠️ Độ tự tin giảm")
        
        print(f"  ✅ LoRA chỉ cập nhật {lora_metrics['trainable_params']:,} params ({(lora_metrics['trainable_params']/total_params*100):.2f}% tổng)")
        
        # Hiển thị chi tiết dự đoán
        print(f"\n📝 DETAILED PREDICTIONS:")
        print(f"{'Text':<50} | {'True':<5} | {'Pretrained':<12} | {'LoRA':<12}")
        print("-" * 90)
        
        for text, true_label, pred_pretrained, pred_lora in zip(
            test_texts,
            test_labels,
            pretrained_metrics['predictions'],
            lora_metrics['predictions']
        ):
            text_display = text[:45] + "..." if len(text) > 45 else text
            true_str = "+" if true_label == 1 else "-"
            pred_pre = "+" if pred_pretrained == 1 else "-"
            pred_lora_str = "+" if pred_lora == 1 else "-"
            
            match_pre = "✓" if pred_pretrained == true_label else "✗"
            match_lora = "✓" if pred_lora == true_label else "✗"
            
            print(f"{text_display:<50} | {true_str:<5} | {pred_pre} {match_pre:<10} | {pred_lora_str} {match_lora:<10}")
        
    else:
        print(f"\n❌ LoRA checkpoint not found at: {lora_checkpoint_path}")
        print("   Vui lòng chạy run_training_nlu.py trước!")
    
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
