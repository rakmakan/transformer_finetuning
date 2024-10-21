import os
import torch
from transformers import Trainer, TrainingArguments, BertForSequenceClassification
from datasets import load_dataset
import evaluate
from load_model_tokenizer import load_model_and_tokenizer
from model_config import get_config
import kaggle
import pandas as pd
import torch.nn as nn
import mlflow
import mlflow.pytorch
from torch.utils.tensorboard import SummaryWriter

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to download dataset from Kaggle
def download_kaggle_dataset(dataset_name, download_path="./kaggle_datasets"):
    os.makedirs(download_path, exist_ok=True)
    kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True)

# Load Config
config = get_config("bert-base-uncased")
config.save_steps = 125  # Save checkpoints at each epoch assuming 125 iterations per epoch

# Kaggle Dataset to Use
dataset_name = "amananandrai/ag-news-classification-dataset"
download_kaggle_dataset(dataset_name)

# Load Dataset (using train.csv and test.csv)
dataset = load_dataset("csv", data_files={"train": "./kaggle_datasets/train.csv", "validation": "./kaggle_datasets/test.csv"})

# Limit the number of records in training and validation datasets
dataset["train"] = dataset["train"].select(range(1000))  # Limit to 1000 records
dataset["validation"] = dataset["validation"].select(range(500))  # Limit to 500 records

# Load Pre-trained Model and Tokenizer
model, tokenizer = load_model_and_tokenizer(config.model_name, config.task)

# Replace model with BertForSequenceClassification
id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
label2id = {"World": 0, "Sports": 1, "Business": 2, "Sci/Tech": 3}
label2id = {v: k for k, v in id2label.items()}
model = BertForSequenceClassification.from_pretrained(
    config.model_name, 
    num_labels=4, 
    id2label=id2label, 
    label2id=label2id
)

# Move model to the appropriate device
model.to(device)

# Add a padding token if not already present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Calculate max length based on the dataset length distribution
max_length = int(sum([len(tokenizer.tokenize(text)) for text in dataset["train"]["Description"]]) / len(dataset["train"]))

# Tokenize Dataset
def tokenize_function(examples):
    return tokenizer(examples["Description"], padding="max_length", truncation=True, max_length=max_length)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Add labels to the tokenized dataset
def add_labels(examples):
    # Ensure the class index starts from 0
    examples["labels"] = [label - 1 for label in examples["Class Index"]]
    return examples

tokenized_datasets = tokenized_datasets.map(add_labels, batched=True)

# Set Training Arguments
training_args = TrainingArguments(
    output_dir=config.output_dir,
    overwrite_output_dir=True,
    num_train_epochs=config.num_train_epochs,
    per_device_train_batch_size=config.batch_size,
    save_steps=config.save_steps,
    save_total_limit=2,
    logging_dir=config.logging_dir,
    evaluation_strategy="steps",
    logging_steps=10,
)

# Start TensorBoard SummaryWriter
writer = SummaryWriter(config.logging_dir)

# Load metrics for evaluation
accuracy_metric = evaluate.load("accuracy")

# Start MLflow experiment
mlflow.set_experiment("bert_fine_tuning_experiment")
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model_name", config.model_name)
    mlflow.log_param("num_train_epochs", config.num_train_epochs)
    mlflow.log_param("batch_size", config.batch_size)

    # Define Trainer
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), axis=-1)
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        return accuracy

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
    )

    # Fine-tune the Model
    for step, output in enumerate(trainer.train()):
        writer.add_scalar("Training Loss", output.loss, step)

    # Evaluate the Model
    validation_metrics = trainer.evaluate()
    validation_loss = validation_metrics.get("eval_loss", None)
    validation_accuracy = validation_metrics.get("eval_accuracy", None)

    if validation_loss is not None:
        print(f"Validation Loss: {validation_loss}")
        mlflow.log_metric("validation_loss", validation_loss)
        writer.add_scalar("Validation Loss", validation_loss)

    if validation_accuracy is not None:
        print(f"Validation Accuracy: {validation_accuracy}")
        mlflow.log_metric("validation_accuracy", validation_accuracy)
        writer.add_scalar("Validation Accuracy", validation_accuracy)

    # Calculate training accuracy
    train_metrics = trainer.evaluate(eval_dataset=tokenized_datasets["train"])
    train_accuracy = train_metrics.get("eval_accuracy", None)
    if train_accuracy is not None:
        print(f"Training Accuracy: {train_accuracy}")
        mlflow.log_metric("training_accuracy", train_accuracy)
        writer.add_scalar("Training Accuracy", train_accuracy)

    # Log the trained model
    mlflow.pytorch.log_model(model, "model")

# Save the Model
model_output_dir = config.output_dir
os.makedirs(model_output_dir, exist_ok=True)
model.save_pretrained(model_output_dir)
tokenizer.save_pretrained(model_output_dir)

# Close TensorBoard writer
writer.close()

print("Model fine-tuned and saved to", model_output_dir)
