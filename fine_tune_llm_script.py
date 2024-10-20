import os
import torch
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from load_model_tokenizer import load_model_and_tokenizer
from model_config import get_config
from flash_attn.modules.mha import FlashSelfAttention
import torch.nn as nn

# Load Config
config = get_config("gpt2")

# Load Dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Load Pre-trained Model and Tokenizer
model, tokenizer = load_model_and_tokenizer(config.model_name, config.task)

# Replace Attention with Flash Attention
for module in model.modules():
    if isinstance(module, nn.MultiheadAttention):
        flash_attention = FlashSelfAttention(module.embed_dim, num_heads=module.num_heads)
        module.forward = flash_attention.forward

# Tokenize Dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set Training Arguments
training_args = TrainingArguments(
    output_dir=config.output_dir,
    overwrite_output_dir=True,
    num_train_epochs=config.num_train_epochs,
    per_device_train_batch_size=config.batch_size,
    save_steps=config.save_steps,
    save_total_limit=2,
    logging_dir=config.logging_dir,
    evaluation_strategy="epoch",
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Fine-tune the Model
trainer.train()

# Evaluate the Model
validation_metrics = trainer.evaluate()
validation_loss = validation_metrics.get("eval_loss", None)
if validation_loss is not None:
    print(f"Validation Loss: {validation_loss}")

# Save the Model
model_output_dir = config.output_dir
os.makedirs(model_output_dir, exist_ok=True)
model.save_pretrained(model_output_dir)
tokenizer.save_pretrained(model_output_dir)

print("Model fine-tuned and saved to", model_output_dir)
