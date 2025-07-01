# Supervised Fine-Tuning with Unsloth: Medical Q&A Model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sap156/Supervised-Fine-Tuning-UnSloth/blob/main/SFT.ipynb)

A comprehensive guide to fine-tuning large language models for medical question answering using Unsloth, HuggingFace, and Google Colab. This project demonstrates how to customize a DeepSeek-R1 model to provide medical expertise with chain-of-thought reasoning.

## 🎯 Project Overview

This project shows you how to:
- Fine-tune the DeepSeek-R1-Distill-Llama-8B model for medical applications
- Implement chain-of-thought reasoning for complex medical queries
- Use Unsloth for efficient and fast fine-tuning
- Deploy your model to HuggingFace Hub

## 🚀 Quick Start

### Prerequisites

- Google Colab account (recommended) or local GPU with CUDA support
- HuggingFace account and API token
- Basic understanding of Python and machine learning concepts

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/sap156/Supervised-Fine-Tuning-UnSloth.git
   cd Supervised-Fine-Tuning-UnSloth
   ```

2. **Open in Google Colab**
   - Click the "Open in Colab" badge above
   - Or upload the `SFT.ipynb` file to your Google Colab

3. **Set up GPU Runtime**
   - In Colab: Runtime → Change runtime type → Hardware accelerator → GPU (T4 recommended)

## 📋 Step-by-Step Guide

### Step 1: Install Dependencies

```python
!pip install unsloth
```

The notebook installs Unsloth, which provides optimized fine-tuning capabilities with significant speed improvements and memory efficiency.

### Step 2: Import Required Libraries

```python
from unsloth import FastLanguageModel
import torch
```

### Step 3: Configure Model Parameters

```python
max_seq_length = 2048  # Maximum sequence length for training
dtype = None          # Auto-detect optimal data type
load_in_4bit = True   # Enable 4-bit quantization for memory efficiency
```

### Step 4: Authenticate with HuggingFace

```python
from huggingface_hub import login
from google.colab import userdata

hf_token = userdata.get('HuggingFace')  # Store your token in Colab secrets
login(hf_token)
```

**Setting up HuggingFace Token in Colab:**
1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token with write permissions
3. In Colab: Go to the key icon (🔑) on the left sidebar
4. Add a new secret named `HuggingFace` with your token value

### Step 5: Load Base Model

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = hf_token,
)
```

### Step 6: Define Prompt Template

```python
prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.

Write a response that appropriately completes the request.

Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:

You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.

Please answer the following medical question.

### Question:

{}

### Response:

<think>{}"""
```

### Step 7: Test Base Model (Optional)

Before fine-tuning, you can test the base model's performance on medical questions to establish a baseline.

### Step 8: Configure LoRA for Fine-Tuning

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=9001,
    use_rslora=False,
    loftq_config=None,
)
```

**LoRA Parameters Explained:**
- `r=16`: Rank of adaptation - higher values = more parameters but better performance
- `target_modules`: Which layers to adapt
- `lora_alpha=16`: Scaling parameter for LoRA
- `lora_dropout=0`: Dropout rate (0 for optimization)

### Step 9: Prepare Training Data

```python
def formatting_prompts_func(examples):
    inputs = examples["Question"]          # Medical questions
    cots = examples["Complex_CoT"]         # Chain of thought reasoning
    outputs = examples["Response"]         # Final answers
    texts = []

    for input, cot, output in zip(inputs, cots, outputs):
        text = train_prompt_style.format(input, cot, output) + EOS_TOKEN
        texts.append(text)

    return { "text": texts }
```

### Step 10: Load and Format Dataset

```python
from datasets import load_dataset

# Load subset of medical dataset
dataset = load_dataset(
    "FreedomIntelligence/medical-o1-reasoning-SFT",
    "en",
    split="train[0:500]",  # Using first 500 examples for demo
    trust_remote_code=True
)

# Format dataset
dataset = dataset.map(formatting_prompts_func, batched=True)
```

### Step 11: Configure Training

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,

    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,  # Increase for longer training
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none"
    ),
)
```

### Step 12: Start Training

```python
trainer_stats = trainer.train()
```

**Training Tips:**
- Monitor the loss curve - it should decrease over time
- For production use, increase `max_steps` or use `num_train_epochs`
- Adjust `per_device_train_batch_size` based on your GPU memory

### Step 13: Test Fine-Tuned Model

After training, test your model with the same question to see improvements:

```python
# Test the fine-tuned model
FastLanguageModel.for_inference(model)
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")
outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,
    use_cache=True,
)
response = tokenizer.batch_decode(outputs)
print(response[0].split("### Response:")[1])
```

### Step 14: Save and Deploy Model

```python
# Save locally
model.save_pretrained("DeepSeek-R1-Medical-INSTAGRAM")
tokenizer.save_pretrained("DeepSeek-R1-Medical-INSTAGRAM")

# Push to HuggingFace Hub
model.push_to_hub("your-username/your-model-name")
tokenizer.push_to_hub("your-username/your-model-name")
```

## 📊 Dataset Information

This project uses the `FreedomIntelligence/medical-o1-reasoning-SFT` dataset, which contains:
- Medical questions and scenarios
- Chain-of-thought reasoning processes
- Expert-level medical responses

**Dataset Structure:**
- `Question`: Medical query or case study
- `Complex_CoT`: Detailed reasoning process
- `Response`: Final medical answer or diagnosis

## 🔧 Customization Options

### Adjusting Model Size
```python
# For different model sizes
model_options = {
    "small": "unsloth/llama-3-8b-bnb-4bit",
    "medium": "unsloth/DeepSeek-R1-Distill-Llama-8B",
    "large": "unsloth/llama-3-70b-bnb-4bit"
}
```

### Training Parameters
```python
# For longer, more thorough training
TrainingArguments(
    per_device_train_batch_size=1,  # Reduce if out of memory
    gradient_accumulation_steps=8,   # Increase for larger effective batch size
    num_train_epochs=3,             # Train for multiple epochs
    max_steps=-1,                   # Let epochs control training length
    learning_rate=1e-4,             # Lower for more stable training
    warmup_ratio=0.1,               # 10% of steps for warmup
)
```

### Custom Datasets
To use your own dataset:
1. Format data with `Question`, `Complex_CoT`, and `Response` columns
2. Modify the `formatting_prompts_func` to match your data structure
3. Adjust the prompt template for your domain

## 💡 Best Practices

### Memory Management
- Use `load_in_4bit=True` for memory efficiency
- Reduce `per_device_train_batch_size` if you encounter OOM errors
- Enable gradient checkpointing with `use_gradient_checkpointing="unsloth"`

### Training Optimization
- Start with small datasets (100-500 examples) for testing
- Monitor training loss - it should decrease consistently
- Use warmup steps to stabilize training
- Save checkpoints regularly for long training runs

### Quality Assurance
- Test your model on held-out examples
- Compare performance before and after fine-tuning
- Validate outputs for factual accuracy in medical contexts

## 🚨 Important Notes

### Medical Disclaimer
This model is for educational and research purposes only. Do not use for actual medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.

### Resource Requirements
- **GPU**: NVIDIA T4 (free in Colab) or better
- **RAM**: 12GB+ recommended
- **Storage**: 10GB+ for model and dataset
- **Time**: 30-60 minutes for the demo (500 examples, 60 steps)

### Common Issues and Solutions

**Out of Memory Errors:**
```python
# Reduce batch size
per_device_train_batch_size=1

# Increase gradient accumulation
gradient_accumulation_steps=8

# Use smaller model
model_name = "unsloth/llama-3-8b-bnb-4bit"
```

**Slow Training:**
```python
# Enable optimizations
use_gradient_checkpointing="unsloth"
optim="adamw_8bit"
fp16=True  # or bf16=True if supported
```

**Authentication Issues:**
- Ensure your HuggingFace token has write permissions
- Store token securely in Colab secrets
- Check token validity on HuggingFace website

