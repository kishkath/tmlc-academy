import pandas as pd
import json

def create_jsonl_files_from_excel(excel_path="gym_workout_dataset.xlsx"):
    # Read training and validation sheets
    train_df = pd.read_excel(excel_path, sheet_name='training_data')
    val_df = pd.read_excel(excel_path, sheet_name='validation_data')

    def write_jsonl(df, filename):
        with open(filename, "w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                json_line = json.dumps({
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that provides expert guidance on gym workouts, exercises, and related nutrition advice."
                        },
                        {
                            "role": "user",
                            "content": row["prompt"]
                        },
                        {
                            "role": "assistant",
                            "content": row["completion"]
                        }
                    ]
                })
                f.write(json_line + "\n")

    # Write training and validation JSONL files
    train_jsonl_file_path, val_jsonl_file_path = "processed_datasets/training.jsonl", "processed_datasets/validation.jsonl"
    write_jsonl(train_df, train_jsonl_file_path)
    write_jsonl(val_df, val_jsonl_file_path)

    return train_jsonl_file_path, val_jsonl_file_path
