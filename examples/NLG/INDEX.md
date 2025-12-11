# 📖 INDEX - LoRA NLG Examples Documentation

## 📋 Quick Navigation

### 📝 Documentation Files

| File | Mục đích | Status |
|------|---------|--------|
| `RUN_DEMO.md` | Hướng dẫn chi tiết cách chạy | ✅ |
| `EXECUTION_SUMMARY.md` | Tóm tắt kết quả chạy | ✅ |
| `INDEX.md` | File này - Navigation | ✅ |

---

### 🐍 Python Scripts

| Script | Dòng lệnh | Chức năng |
|--------|-----------|----------|
| `run_training.py` | `python run_training.py` | Training GPT-2 với LoRA |
| `run_inference.py` | `python run_inference.py` | Load checkpoint & inference |
| `compare_lora_vs_full.py` | `python compare_lora_vs_full.py` | So sánh LoRA vs Full FT |
| `src/gpt2_ft.py` | (chính thức) | Original training code |
| `src/model.py` | (chính thức) | GPT-2 model definition |

---

### 📁 Data Files

```
data/e2e/
├── train.txt          ← Training data
├── valid.txt          ← Validation data  
└── test.txt           ← Test data (chưa dùng)
```

**Dữ liệu**: E2E NLG Challenge dataset

---

### 💾 Output Files

```
lora_model/
└── pytorch_model.bin  (4.06 MB) - LoRA checkpoint
    ├─ Tạo bởi: run_training.py
    ├─ Kích thước: 4.06 MB
    └─ Load bởi: run_inference.py
```

---

## 🚀 Quick Start

### 1️⃣ Training

```bash
# Chạy training với default params
python run_training.py

# Hoặc tùy chỉnh
python run_training.py \
    --num_epochs 3 \
    --batch_size 8 \
    --lora_dim 32 \
    --lr 2e-4
```

**Kết quả**: 
- Logs để track training loss
- Checkpoint lưu tại `lora_model/pytorch_model.bin`

### 2️⃣ Inference

```bash
# Load checkpoint và chạy inference
python run_inference.py
```

**Kết quả**:
- Model load thành công
- Inference trên sample inputs
- Hiển thị predictions

### 3️⃣ Comparison

```bash
# So sánh LoRA vs Full Fine-tuning
python compare_lora_vs_full.py
```

**Kết quả**:
- Parameter statistics
- Memory usage comparison
- Storage efficiency analysis

---

## 📊 Key Metrics

### LoRA Performance

```
Training Time:        ~4 seconds/epoch (demo setup)
Trainable Parameters: 1,062,160 (1.21% of total)
Checkpoint Size:      4.06 MB
GPU Memory:           ~8.1 MB

Full Fine-tuning:     86,689,873 params, 330.70 MB, ~661 MB GPU
LoRA vs Full:         99% reduction in trainable params
```

---

## 🔍 Architecture Details

### Model Structure

```
SimpleGPT2WithLoRA(
    vocab_size=50257
    hidden_dim=768
    num_layers=2
    lora_dim=16
)

Layers:
├─ Embedding: 50257 → 768
├─ 2× Transformer:
│  ├─ lora.Linear: 768 → 3072 (LoRA: 16 × 768 + 3072 × 16)
│  ├─ GELU
│  └─ lora.Linear: 3072 → 768 (LoRA: 16 × 3072 + 768 × 16)
└─ lora.Linear: 768 → 50257 (LoRA: 16 × 768 + 50257 × 16)
```

### LoRA Configuration

```python
lora_dim = 16                    # Rank
lora_alpha = 128                 # Scaling factor
lora_dropout = 0.0               # Dropout
scaling = lora_alpha / lora_dim  # = 8
```

---

## 📚 LoRA Concepts

### Low-Rank Adaptation Formula

```
Output = Base_Weight @ Input + (α/r) × B @ A @ Input

where:
- Base_Weight: Original frozen weight
- A: [r × d_in] LoRA matrix
- B: [d_out × r] LoRA matrix
- α: Scaling factor
- r: Rank dimension
```

### Why LoRA Works

```
1. Neural networks operate in low-rank regime
2. Fine-tuning updates are also low-rank
3. So we can represent updates as B @ A
4. Dramatically reduces trainable parameters
5. Still achieves competitive performance
```

---

## 🎯 Use Cases

### Multi-Task Learning

```python
# Same base model, different LoRA adapters
base_model = load_gpt2()

tasks = ['e2e', 'dart', 'webnlg']
adapters = {}

for task in tasks:
    adapter = load_lora_checkpoint(f'{task}_adapter.bin')
    adapters[task] = adapter

# Use different adapters for different tasks
for task in tasks:
    predictions = base_model(input_ids)  # Base
    predictions += adapters[task](input_ids)  # + LoRA
```

### Resource-Constrained Training

```python
# Train on CPU/edge device with limited memory
model = GPT2WithLoRA()
lora.mark_only_lora_as_trainable(model)

# Training only needs memory for:
# - LoRA parameters: 1.06M
# - Gradients: 1.06M
# - Optimizer states: ~4.24M
# Total: ~8.1 MB (vs 661 MB for full)
```

---

## ⚙️ Parameters Reference

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_epochs` | 2 | Training epochs |
| `--batch_size` | 4 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--seq_len` | 64 | Sequence length |
| `--log_interval` | 10 | Logging interval |
| `--device` | auto | cuda/cpu |

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--vocab_size` | 50257 | Vocabulary size |
| `--hidden_dim` | 768 | Hidden dimension |
| `--num_layers` | 2 | Number of layers |
| `--lora_dim` | 16 | LoRA rank |

### LoRA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lora_dim` | 16 | Rank dimension (r) |
| `--lora_alpha` | 128 | Scaling factor |
| `--lora_dropout` | 0.0 | LoRA dropout |

---

## 🔗 Related Files

### In Repository

```
loralib/
├── __init__.py        - Package entry
├── layers.py          - LoRA layer implementations
└── utils.py           - Utility functions (mark_only_lora_as_trainable, etc.)

examples/NLG/
├── src/
│   ├── model.py       - GPT-2 model
│   ├── gpt2_ft.py     - Full training script
│   ├── data_utils.py  - Data loading
│   └── ... (other utilities)
└── data/
    ├── e2e/           - E2E NLG data
    ├── dart/          - DART data
    └── webnlg_challenge_2017/  - WebNLG data
```

### External References

- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **GitHub**: https://github.com/microsoft/LoRA
- **HuggingFace PEFT**: https://github.com/huggingface/peft
- **E2E Dataset**: http://www.macs.hw.ac.uk/InteractiveSystemsGroup/projects/e2e-dataset/

---

## 🧪 Testing Checklist

- [x] Import libraries (torch, numpy, tqdm, loralib)
- [x] Load data from E2E dataset
- [x] Create model with LoRA layers
- [x] Mark only LoRA as trainable
- [x] Forward pass works
- [x] Backward pass & optimization works
- [x] Save checkpoint
- [x] Load checkpoint
- [x] Inference works
- [x] Merge LoRA weights
- [x] Parameter counting
- [x] Memory calculation

---

## 🛠️ Troubleshooting

### Issue: "Module torch not found"
```bash
# Solution: Install dependencies
pip install torch numpy tqdm
```

### Issue: "Data file not found"
```bash
# Solution: Make sure you're in examples/NLG/ directory
cd examples/NLG/
python run_training.py
```

### Issue: "CUDA out of memory"
```python
# Solution: Use CPU
python run_training.py --device cpu

# Or reduce batch size
python run_training.py --batch_size 2
```

---

## 📈 Performance Tips

1. **Increase Batch Size**: 4 → 8 (if memory allows)
2. **Increase LoRA Rank**: 16 → 32 (more expressive, slower)
3. **Use Gradient Accumulation**: Multiple backward before step
4. **Mixed Precision**: Use fp16 for faster training
5. **Larger Hidden Dim**: 768 → 1024 (more capacity)

---

## 🎓 Learning Path

1. **Basics** → Read EXECUTION_SUMMARY.md
2. **Setup** → Follow RUN_DEMO.md
3. **Implementation** → Study run_training.py
4. **Theory** → Read loralib/layers.py
5. **Comparison** → Run compare_lora_vs_full.py
6. **Advanced** → Explore official examples/NLG/ code

---

## 📞 Quick Reference

### Common Commands

```bash
# Training
python run_training.py --num_epochs 5 --batch_size 8

# Inference
python run_inference.py

# Comparison
python compare_lora_vs_full.py

# Check checkpoint
ls -lh lora_model/
```

### Quick Python Imports

```python
import loralib as lora
import torch
from run_training import SimpleGPT2WithLoRA

# Create model
model = SimpleGPT2WithLoRA(lora_dim=16)

# Mark LoRA as trainable
lora.mark_only_lora_as_trainable(model)

# Get state dict
state = lora.lora_state_dict(model)
```

---

## ✅ Status

- ✅ All scripts tested
- ✅ Data available
- ✅ Training works
- ✅ Inference works
- ✅ Checkpoints saved
- ✅ Comparison analysis complete
- ✅ Documentation ready

---

**Last Updated**: 2025-12-11  
**Status**: Production Ready ✨
