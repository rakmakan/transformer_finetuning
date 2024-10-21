import os
import json

class Config:
    def __init__(self, model_name, task, output_dir, num_train_epochs, batch_size, save_steps, logging_dir):
        self.model_name = model_name
        self.task = task
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        self.save_steps = save_steps
        self.logging_dir = logging_dir

    def save(self, filepath):
        config_dict = {
            "model_name": self.model_name,
            "task": self.task,
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_train_epochs,
            "batch_size": self.batch_size,
            "save_steps": self.save_steps,
            "logging_dir": self.logging_dir,
        }
        with open(filepath, 'w') as config_file:
            json.dump(config_dict, config_file, indent=4)

# Example configurations for different models
config_gpt2 = Config(
    model_name="gpt2",
    task="causal-lm",
    output_dir="./results_gpt2",
    num_train_epochs=3,
    batch_size=4,
    save_steps=10_000,
    logging_dir="./logs_gpt2"
)

config_bert = Config(
    model_name="bert-base-uncased",
    task="sequence-classification",
    output_dir="./results_bert",
    num_train_epochs=10,
    batch_size=8,
    save_steps=5000,
    logging_dir="./logs_bert"
)

# Add more configurations as needed

# Function to get configuration by model name
def get_config(model_name):
    if model_name == "gpt2":
        return config_gpt2
    elif model_name == "bert-base-uncased":
        return config_bert
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

# Example usage
if __name__ == "__main__":
    config = get_config("gpt2")
    print(f"Loaded config for {config.model_name}")
    config.save("./config_gpt2.json")
