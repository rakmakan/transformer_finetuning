#!/bin/bash

# Create directories
mkdir -p transformer_finetuning
cd transformer_finetuning

mkdir -p flash_attention
mkdir -p kaggle_datasets
mkdir -p results_gpt2
mkdir -p results_bert
mkdir -p logs_gpt2
mkdir -p logs_bert

# Create empty Python scripts
touch fine_tune_llm_script.py
touch load_model_tokenizer.py
touch model_config.py
touch experiment_with_kaggle.py

touch requirements.txt

# Make scripts executable
chmod +x fine_tune_llm_script.py
chmod +x load_model_tokenizer.py
chmod +x model_config.py
chmod +x experiment_with_kaggle.py

echo "Project structure created successfully."
