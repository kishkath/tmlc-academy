import json
from datasets import Dataset
from transformers import PreTrainedTokenizerBase


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    print(f"Loaded {len(data)} examples from {file_path}")
    return data


def create_dataset(train_file, eval_file):
    train_data = load_jsonl(train_file)
    eval_data = load_jsonl(eval_file)

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    print("Datasets created.\n")
    return train_dataset, eval_dataset


def tokenize_dataset(dataset, tokenizer: PreTrainedTokenizerBase, max_length=512):
    def preprocess_function(example):
        question = example.get("question", "")
        answer = example.get("answer", "")
        prompt = f"User: {question}\nAssistant: {answer}"
        return tokenizer(prompt, truncation=True, padding="max_length", max_length=max_length)

    tokenized_dataset = dataset.map(preprocess_function, batched=False)
    return tokenized_dataset
