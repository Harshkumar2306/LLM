# 🧠 LLM Experiments: Building GPT from Scratch

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Google Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)

A comprehensive, end-to-end framework for building, training, and evaluating Large Language Models (LLMs) from scratch. This repository contains the mathematical implementation of a custom GPT-style autoregressive language model, integrated with a fully automated Google Colab + Google Drive training pipeline.

The purpose of this repository is to conduct rigorous architectural experiments (Baseline vs. RoPE vs. SwiGLU) on a ~17.6M parameter model trained on the TinyShakespeare dataset.

---

## 🔬 Architectural Experiments & Results

We trained three distinct architectures to exactly 10,000 iterations on a T4 GPU using FlashAttention. To ensure scientific validity, all models were mathematically scaled to maintain parity at **~17.6 Million Parameters**.

| Architecture | Parameters | Training Time | Validation Loss | Perplexity | Generation Speed | Branch |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** (Absolute + GELU) | 17.67M | ~60 mins | **4.86** | **130.0** | 318 tok/sec | `main` |
| **SwiGLU** (Absolute + SwiGLU) | 17.67M | ~57 mins | **4.93** | **139.4** | 278 tok/sec | `exp/swiglu` |
| **RoPE** (Rotary + GELU) | 17.60M | ~60 mins | **5.34** | **208.8** | 242 tok/sec | `exp/rope` |

> **Research Takeaway:** 
> Modern LLM techniques like RoPE and SwiGLU are designed to extrapolate efficiently across trillions of tokens and billions of parameters. When downscaled to a tiny 17M parameter model on a small dataset, the rigid, highly-parameterized "Baseline" architecture (Absolute Embeddings) actually overfits the short-context positions slightly better. This perfectly reproduces known scaling-law phenomena.

---

## 🚀 Features

- **Custom PyTorch GPT Engine**: A fully functional, mathematical implementation of multi-head causal self-attention.
- **FlashAttention**: Natively dispatches to PyTorch's optimized `F.scaled_dot_product_attention` to fuse QKV kernels and drastically accelerate GPU training.
- **Automated Colab Pipeline**: A notebook (`colab_train.ipynb`) that automatically clones the repository, mounts Google Drive, pre-processes the dataset, and seamlessly resumes training from the latest checkpoint.
- **Mixed-Precision Training**: Utilizes `torch.amp.GradScaler` (FP16) for reduced memory footprint.
- **Gradient Accumulation**: Bypasses GPU memory limits by accumulating gradients across micro-batches.
- **Rigorous Evaluation**: Sliding-window validation loss calculation to accurately measure model perplexity.

---

## 📂 Repository Structure

```text
LLM/
├── core/                  # Neural Network Architecture
│   ├── model.py           # The primary GPT module
│   ├── attention.py       # Causal Self-Attention & FlashAttention
│   ├── block.py           # Transformer Blocks
│   ├── embeddings.py      # Positional & Token Embeddings
│   └── ffn.py             # Feed-Forward Networks (GELU/SwiGLU)
├── engine/                # Training Infrastructure
│   └── trainer.py         # Advanced Training Loop (Checkpoints, AMP)
├── data/                  # Data Processing
│   ├── dataset.py         # PyTorch Dataset implementation
│   └── tokenizer.py       # Character-level tokenization
├── scripts/               # Execution Scripts
│   ├── prepare_data.py    # Downloads and compresses the dataset
│   ├── train.py           # Main training script
│   ├── evaluate.py        # Rigorous validation loss calculation
│   ├── benchmark.py       # System speed/memory profiling
│   └── generate.py        # Autoregressive text generation
└── colab_train.ipynb      # The automated cloud pipeline
```

---

## 💻 Usage & Reproduction

To reproduce these experiments locally or on a cloud GPU:

### 1. Setup & Data Preparation
```bash
git clone https://github.com/Harshkumar2306/LLM.git
cd LLM
pip install -r requirements.txt

# Download and tokenize the TinyShakespeare dataset
python scripts/prepare_data.py --input data/input.txt --outdir data/
```

### 2. Training
Train the model. The trainer will automatically checkpoint to the `--out_dir`.
```bash
python scripts/train.py \
    --train_bin data/train.bin \
    --val_bin data/val.bin \
    --device cuda \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --out_dir runs/my_experiment
```

### 3. Evaluation & Benchmarking
Run the rigorous sliding-window evaluation and measure system throughput.
```bash
python scripts/evaluate.py --checkpoint runs/my_experiment/best.pt --val_bin data/val.bin --device cuda
python scripts/benchmark.py --device cuda
```

### 4. Text Generation
Test the model's predictive capabilities by giving it a prompt.
```bash
python scripts/generate.py \
    --checkpoint runs/my_experiment/best.pt \
    --prompt "ROMEO:" \
    --device cuda
```

---

## 🤝 Branches

To explore the specific mathematical implementations of the different architectures, check out their respective branches:
- **`main`**: The standard GPT-2 Baseline (Absolute Embeddings + GELU).
- **`exp/rope`**: The implementation of Rotary Position Embeddings via polar complex geometry.
- **`exp/swiglu`**: The implementation of the Swish-Gated Linear Unit activation mechanism.

*Developed with ❤️ as a deep dive into Large Language Model Architectures.*
