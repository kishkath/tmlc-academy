import csv
import difflib
import json
import random

from dataset.dataset import create_jsonl_files_from_excel
from finetuning.finetune import finetune, manage_fine_tune_jobs, query_fine_tuned_model, wait_for_finetuned_model


def read_random_validation_prompt(val_file_path):
    with open(val_file_path, "r", encoding="utf-8") as f:
        val_data = [json.loads(line) for line in f]

    random_example = random.choice(val_data)
    user_prompt = ""
    expected = ""
    for msg in random_example["messages"]:
        if msg["role"] == "user":
            user_prompt = msg["content"]
        elif msg["role"] == "assistant":
            expected = msg["content"]

    return user_prompt, expected


def evaluate_prompt(model_name, prompt, expected=None):
    actual = query_fine_tuned_model(model_name, prompt)
    similarity = difflib.SequenceMatcher(None, expected, actual).ratio() if expected else None

    result = {
        "prompt": prompt,
        "expected_completion": expected,
        "actual_completion": actual,
        "similarity_score": round(similarity, 3) if similarity else None
    }

    with open("results/evaluation_results_log.jsonl", "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(result, ensure_ascii=False) + "\n")

    return result


def evaluate_on_validation(model_name, val_file_path, output_json="evaluation_report.json",
                           output_csv="evaluation_report.csv"):
    print("\n🔍 Running evaluation on validation data...")
    results = []

    with open(val_file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            prompt = data["messages"][1]["content"]
            expected = data["messages"][2]["content"]
            result = evaluate_prompt(model_name, prompt, expected)
            results.append(result)

    with open(output_json, "w", encoding="utf-8") as f_json:
        json.dump(results, f_json, indent=2, ensure_ascii=False)

    with open(output_csv, "w", newline='', encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv,
                                fieldnames=["prompt", "expected_completion", "actual_completion", "similarity_score"])
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Evaluation complete.\n📄 JSON saved to: {output_json}\n📄 CSV saved to: {output_csv}")


def save_finetune_metadata(details, model_name):
    with open("finetuning/finetune_details/finetune_job_metadata.json", "w", encoding="utf-8") as f:
        json.dump(details, f, indent=4, default=str)

    if model_name:
        with open("finetuning/finetune_details/finetuned_model_name.txt", "w") as f:
            f.write(model_name)
        print("📁 Model name saved to: finetuned_model_name.txt")


def load_existing_model():
    try:
        with open("finetuning/finetune_details/finetuned_model_name.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


if __name__ == '__main__':
    model_name = load_existing_model()

    if model_name:
        print(f"✅ Using existing fine-tuned model: {model_name}")
    else:
        train_file, val_file = create_jsonl_files_from_excel("dataset/input_data/gym_workout_dataset.xlsx")
        print("📄 Training file created:", train_file)
        print("🧪 Validation file created:", val_file)

        finetuned_details = finetune(train_file)
        if not finetuned_details:
            print("❌ Fine-tuning failed to start.")
            exit()

        print("🛠️ Fine-tune job started:", finetuned_details)
        jobs_result = manage_fine_tune_jobs("list_jobs", finetuned_details)
        print("📋 Recent fine-tuning jobs:", jobs_result)

        model_name = wait_for_finetuned_model(finetuned_details.id)

        if model_name:
            print("✅ Fine-tuned model is ready:", model_name)
            save_finetune_metadata(finetuned_details, model_name)
        else:
            print("⚠️ Fine-tuned model is not ready yet. Please check again later.")
            exit()

    val_file = r"C:\Users\saikir\learnings\TMLC-GENAI\week_1_finetuning_llm\dataset\processed_datasets\validation.jsonl"

    user_choice = input("\nWould you like to test the model with your own prompt? (Y/N): ").strip().upper()

    if user_choice == "Y":
        user_prompt = input("👉 Enter your test prompt: ").strip()
        result = evaluate_prompt(model_name, user_prompt, expected=None)
        print("\n🤖 Response from fine-tuned model:", result["actual_completion"])
    else:
        user_prompt, expected = read_random_validation_prompt(val_file)
        print(f"\n🔁 Using random validation prompt:\n🗣️ Prompt: {user_prompt}\n📥 Expected: {expected}")
        result = evaluate_prompt(model_name, user_prompt, expected)
        print("\n🤖 Response from fine-tuned model:", result["actual_completion"])
        print("\n✅ Expected response for comparison:", expected)

    evaluate_on_validation(model_name, val_file)
