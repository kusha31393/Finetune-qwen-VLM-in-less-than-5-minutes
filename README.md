# Qwen Vision Language Model Fine-tuning for LaTeX OCR

This project fine-tunes the Qwen2-VL-7B-Instruct model to convert mathematical equation images into their corresponding LaTeX representations using LoRA (Low-Rank Adaptation) for efficient parameter-efficient training.

## 📖 Overview

The script implements a complete pipeline for fine-tuning a vision-language model on the task of mathematical equation recognition. It transforms images of mathematical expressions into LaTeX code, making it useful for digitizing mathematical content from documents, whiteboards, or handwritten notes.

### Key Features

- **Memory Efficient**: Uses 4-bit quantization to reduce memory usage from ~28GB to ~7GB
- **Parameter Efficient**: Employs LoRA adaptation to train only a small subset of parameters
- **Vision-Language Integration**: Handles both image and text inputs simultaneously
- **Real-time Feedback**: Streaming text generation for immediate results
- **Comprehensive Evaluation**: Pre and post-training testing for performance comparison

## 🏗️ Architecture

### Model Details
- **Base Model**: `unsloth/Qwen2-VL-7B-Instruct-bnb-4bit`
- **Architecture**: Vision-Language Transformer with cross-modal attention
- **Quantization**: 4-bit quantization using bitsandbytes
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)

### Training Configuration
- **LoRA Rank**: 16
- **LoRA Alpha**: 16
- **Batch Size**: 2 (per device) with 4 gradient accumulation steps
- **Learning Rate**: 2e-4
- **Training Steps**: 30 (demo configuration)
- **Precision**: bf16 (if supported) or fp16

## 📊 Dataset

**Dataset**: [unsloth/Latex_OCR](https://huggingface.co/datasets/unsloth/Latex_OCR)
- Contains thousands of mathematical equation images paired with LaTeX code
- Covers various mathematical expressions, formulas, and symbols
- Pre-processed for vision-language model training

### Data Format
Each sample contains:
- `image`: PIL Image of mathematical equation
- `text`: Corresponding LaTeX representation

## 🚀 Getting Started

### Prerequisites

#### Hardware Requirements
- **GPU**: NVIDIA GPU with at least 8GB VRAM (RTX 3080/4070 or better recommended)
- **RAM**: 16GB+ system RAM
- **Storage**: 10GB+ free space for model and dependencies

#### Software Requirements
- Python 3.8+
- CUDA 11.8+ or 12.x
- PyTorch 2.0+

### Installation

1. **Clone or download the script**:
   ```bash
   # If using git
   git clone <your-repo-url>
   
   # Or simply download finetune_qwen_vl.py
   ```

2. **Install dependencies** (automatically handled by the script):
   ```python
   # The script automatically installs:
   # - bitsandbytes: 4-bit quantization support
   # - accelerate: Multi-GPU training acceleration  
   # - xformers: Memory-efficient attention
   # - peft: Parameter-Efficient Fine-Tuning
   # - trl: Transformer Reinforcement Learning
   # - unsloth: Fast training library
   # - datasets: Dataset loading and processing
   # - transformers: Model implementation
   ```

### Usage

#### Option 1: Jupyter Notebook/Google Colab
1. Upload `finetune_qwen_vl.py` to your notebook environment
2. Run cells sequentially
3. Monitor training progress and generation outputs

#### Option 2: Python Script
```bash
python finetune_qwen_vl.py
```

**Note**: The script is currently formatted for interactive execution (with `!pip` commands). For standalone script usage, replace `!pip` with `subprocess.run()` calls.

## 📋 Script Workflow

### 1. Environment Setup
- Installs required dependencies
- Imports necessary libraries
- Sets up CUDA environment

### 2. Model Loading
- Loads pre-quantized Qwen2-VL model
- Configures LoRA adapters
- Sets up tokenizer

### 3. Dataset Preparation
- Downloads LaTeX OCR dataset
- Converts samples to conversation format
- Prepares data for training

### 4. Pre-training Evaluation
- Tests model performance before fine-tuning
- Generates sample LaTeX output
- Establishes baseline performance

### 5. Fine-tuning
- Configures SFT trainer with optimized settings
- Trains LoRA adapters on LaTeX OCR task
- Monitors training progress

### 6. Post-training Evaluation
- Tests fine-tuned model performance
- Compares with pre-training results
- Demonstrates improvement

## ⚙️ Configuration Options

### LoRA Parameters
```python
r=16,                # Rank of adaptation matrices
lora_alpha=16,       # LoRA scaling parameter
lora_dropout=0,      # Dropout rate for LoRA layers
bias="none",         # Bias configuration
```

### Training Parameters
```python
per_device_train_batch_size=2,    # Batch size per GPU
gradient_accumulation_steps=4,    # Gradient accumulation
warmup_steps=5,                   # Learning rate warmup
max_steps=30,                     # Total training steps
learning_rate=2e-4,               # Learning rate
```

### Generation Parameters
```python
max_new_tokens=128,    # Maximum LaTeX length
temperature=1.5,       # Sampling temperature
min_p=0.1,            # Minimum probability threshold
use_cache=True,       # Enable KV caching
```

## 🔧 Customization

### Extending Training Steps
For production use, increase `max_steps`:
```python
max_steps=1000,  # or more depending on dataset size
```

### Adjusting Batch Size
For different GPU memory:
```python
# For 16GB+ VRAM
per_device_train_batch_size=4,
gradient_accumulation_steps=2,

# For 8GB VRAM  
per_device_train_batch_size=1,
gradient_accumulation_steps=8,
```

### Custom Datasets
To use your own dataset:
1. Format data with `image` and `text` fields
2. Replace dataset loading:
   ```python
   dataset = load_dataset("your-dataset-name", split="train")
   ```

### Different Models
To use alternative models:
```python
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",  # Alternative model
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth"
)
```

## 📈 Performance Optimization

### Memory Optimization
- **4-bit Quantization**: Reduces memory by ~75%
- **Gradient Checkpointing**: Trades compute for memory
- **LoRA**: Trains only ~1% of total parameters

### Speed Optimization
- **Unsloth Library**: 2x faster training than standard implementations
- **xFormers**: Memory-efficient attention mechanisms
- **Mixed Precision**: bf16/fp16 for faster computation

## 🐛 Troubleshooting

### Common Issues

#### Out of Memory Error
```bash
# Reduce batch size
per_device_train_batch_size=1
gradient_accumulation_steps=8

# Enable gradient checkpointing
use_gradient_checkpointing="unsloth"
```

#### CUDA Issues
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Installation Issues
```bash
# Install dependencies manually
pip install bitsandbytes accelerate
pip install transformers datasets
pip install unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
```

### Performance Issues
- **Slow Training**: Reduce `max_seq_length` or increase batch size
- **Poor Quality**: Increase `max_steps` or adjust `learning_rate`
- **Memory Leaks**: Restart kernel between runs

## 📊 Expected Results

### Training Metrics
- **Training Loss**: Should decrease steadily
- **Memory Usage**: ~7-8GB VRAM with 4-bit quantization
- **Training Time**: ~2-5 minutes for 30 steps

### Output Quality
- **Pre-training**: Basic LaTeX recognition with errors
- **Post-training**: Improved accuracy and mathematical symbol recognition
- **Convergence**: Noticeable improvement after 20-30 steps

## 🤝 Contributing

### Areas for Improvement
1. **Dataset Expansion**: Add more mathematical domains
2. **Evaluation Metrics**: Implement BLEU/ROUGE scoring
3. **Model Variants**: Support for different model sizes
4. **Production Deployment**: Add inference API and model serving

### Development Setup
```bash
# For development
pip install -e .
pip install pytest black isort

# Run tests
pytest tests/

# Format code  
black finetune_qwen_vl.py
isort finetune_qwen_vl.py
```

## 📚 References

- [Qwen2-VL Model](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- [Unsloth Library](https://github.com/unslothai/unsloth)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Original Colab Notebook](https://colab.research.google.com/drive/1Ng4PP2AMkL69IApMyKt7QM2u-YF6sPHy)

## 📄 License

This project is provided for educational and research purposes. Please check the licenses of the underlying models and datasets:
- Qwen2-VL: Apache 2.0 License
- Unsloth: Apache 2.0 License
- LaTeX OCR Dataset: Check dataset card for licensing

## 🙏 Acknowledgments

- **Alibaba Cloud**: For the Qwen2-VL model
- **Unsloth AI**: For the efficient training library
- **Hugging Face**: For the datasets and transformers library
- **Community**: For the LaTeX OCR dataset

---

**Note**: This is a demonstration script optimized for educational purposes. For production use, consider implementing proper error handling, logging, evaluation metrics, and model validation.
