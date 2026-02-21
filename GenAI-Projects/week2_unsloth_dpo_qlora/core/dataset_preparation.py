from datasets import load_dataset, DatasetDict
from configurations.config import logger

def clean_record(example):
    example["prompt"] = str(example.get("question", "") or "").strip()
    example["chosen"] = str(example.get("chosen", "") or "").strip()
    example["rejected"] = str(example.get("rejected", "") or "").strip()
    return example

def prepare_dataset(train_file, val_file, train_limit=700, test_limit=100):
    logger.info(f"📚 Loading dataset: train={train_file}, val={val_file}")
    dataset = load_dataset(
        "json",
        data_files={"train": train_file, "val": val_file}
    )
    dataset = DatasetDict({"train": dataset["train"], "val": dataset["val"]})
    dataset = dataset.map(clean_record)

    dataset = dataset.filter(
        lambda x: len(x["prompt"]) > 0 and len(x["chosen"]) > 0 and len(x["rejected"]) > 0
    )

    dataset["train"] = dataset["train"].select(range(min(train_limit, len(dataset["train"]))))
    dataset["val"] = dataset["val"].select(range(min(test_limit, len(dataset["val"]))))
    logger.info(f"✅ Dataset prepared: train={len(dataset['train'])}, val={len(dataset['val'])}")

    return dataset
