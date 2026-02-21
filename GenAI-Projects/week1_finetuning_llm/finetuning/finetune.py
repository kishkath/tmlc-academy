import os
import time
from datetime import timedelta

from dotenv import load_dotenv
from openai import OpenAI

from config.config import OPENAI_API_KEY, FINE_TUNE_SUFFIX, DEFAULT_TIMEOUT, POLL_INTERVAL

load_dotenv()

MODEL = os.getenv("LLM")

# Create OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


def upload_training_file(filepath: str):
    try:
        with open(filepath, "rb") as f:
            file_obj = client.files.create(file=f, purpose="fine-tune")

        print("✅ File uploaded successfully.")
        return file_obj
    except Exception as e:
        print(f"❌ Error uploading file: {e}")
        return None


def finetune(filepath: str):
    file_obj = upload_training_file(filepath)
    if not file_obj or not client:
        print("❌ Fine-tuning aborted due to file upload error.")
        return None

    try:
        fine_tune_job = client.fine_tuning.jobs.create(
            training_file=file_obj.id,
            suffix=FINE_TUNE_SUFFIX,
            model=MODEL
        )
        print(f"🚀 Fine-tuning started successfully. Job ID: {fine_tune_job.id}")
        return fine_tune_job
    except Exception as e:
        print(f"❌ Failed to start fine-tuning: {e}")
        return None


def manage_fine_tune_jobs(action: str, finetuned_job: str = None, model_id: str = None, limit: int = 10,
                          messages: list = None):
    job_id = finetuned_job.id if finetuned_job else None
    try:
        if action == "list_jobs":
            return client.fine_tuning.jobs.list(limit=limit)
        elif action == "retrieve_job" and job_id:
            return client.fine_tuning.jobs.retrieve(job_id)
        elif action == "cancel_job" and job_id:
            return client.fine_tuning.jobs.cancel(job_id)
        elif action == "list_events" and job_id:
            return client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=limit)
        elif action == "delete_model" and model_id:
            return client.models.delete(model_id)
        elif action == "use_model" and model_id and messages:
            return client.chat.completions.create(model=model_id, messages=messages)
        else:
            print("⚠️ Invalid parameters provided for action:", action)
            return None
    except Exception as e:
        print(f"❌ Error during '{action}': {e}")
        return None


def get_finetuned_model_name(job_id: str):
    try:
        job = client.fine_tuning.jobs.retrieve(job_id)
        if job and job.fine_tuned_model:
            print(f"✅ Fine-tuned model available: {job.fine_tuned_model}")
            return job.fine_tuned_model
        else:
            print("⚠️ Fine-tuned model is not yet available. The job might still be running or failed.")
            return None
    except Exception as e:
        print(f"❌ Error retrieving fine-tuned model name: {e}")
        return None


def wait_for_finetuned_model(job_id: str, timeout: int = DEFAULT_TIMEOUT, poll_interval: int = POLL_INTERVAL):
    start_time = time.time()
    print(f"⏳ Waiting for fine-tuned model to be ready (timeout: {timeout // 60} minutes)...")

    while time.time() - start_time < timeout:
        try:
            job = client.fine_tuning.jobs.retrieve(job_id)
            elapsed = time.time() - start_time
            elapsed_str = str(timedelta(seconds=int(elapsed)))

            if job.fine_tuned_model:
                print(f"\n✅ Fine-tuned model is now ready after {elapsed_str}: {job.fine_tuned_model}")
                return job.fine_tuned_model
            else:
                print(f"⏱️ {elapsed_str} elapsed... still waiting...")

        except Exception as e:
            print(f"❌ Error checking job: {e}")

        time.sleep(poll_interval)

    total_wait = str(timedelta(seconds=int(time.time() - start_time)))
    print(f"⌛ Timeout after {total_wait}: Fine-tuned model not ready.")
    return None


def query_fine_tuned_model(
        fine_tuned_model: str,
        user_message: str,
        system_message: str = "You are a helpful assistant which acts as an expert guidance on gym workouts, exercises, and related nutrition advice."
):
    try:
        completion = client.chat.completions.create(
            model=fine_tuned_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )

        response_text = completion.choices[0].message.content.strip()
        print("🧠 Fine-tuned model response:", response_text)
        return response_text

    except Exception as e:
        print(f"❌ Error querying model: {e}")
        return None
