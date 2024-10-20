from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

def load_model_and_tokenizer(model_name: str, task: str):
    """
    Load model and tokenizer from Hugging Face hub based on the specified task.

    Args:
        model_name (str): The name or path of the pre-trained model.
        task (str): The task type, such as 'causal-lm', 'seq2seq-lm', 'sequence-classification', 'token-classification'.

    Returns:
        model, tokenizer: The loaded model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if task == 'causal-lm':
        model = AutoModelForCausalLM.from_pretrained(model_name)
    elif task == 'seq2seq-lm':
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    elif task == 'sequence-classification':
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    elif task == 'token-classification':
        model = AutoModelForTokenClassification.from_pretrained(model_name)
    else:
        raise ValueError(f"Unsupported task: {task}. Supported tasks: 'causal-lm', 'seq2seq-lm', 'sequence-classification', 'token-classification'.")
    
    return model, tokenizer

# Example usage
if __name__ == "__main__":
    model_name = "gpt2"
    task = "causal-lm"
    model, tokenizer = load_model_and_tokenizer(model_name, task)
    print(f"Model and tokenizer for {model_name} ({task}) loaded successfully.")
