# 📈 Customizing OpenAI Models for Gym & Fitness Guidance

This solution fine-tunes OpenAI's powerful language model to serve as a personalized **virtual gym coach**. The trained model understands user queries about workouts, plans, and fitness advice and delivers **accurate, domain-aligned responses**.

---

## 📂 Scenario Overview

The dataset is designed for the domain of **gym workouts and fitness planning**. It consists of user-assistant conversation samples focused on:

* Workout routines (e.g., beginner gym plans, splits, rest days)
* Exercise recommendations for specific goals (e.g., muscle gain, weight loss)
* Nutrition and diet tips (e.g., protein intake, meal timing)
* Gym-related questions from both beginners and regular fitness enthusiasts

The assistant responses are curated to be friendly, supportive, and factually correct based on widely accepted fitness practices.

Example entry from dataset:

```
User: What is the best gym plan for a beginner?
Assistant: The best plan for a beginner is to train three days a week with a focus on compound movements like squats, pushups, and rows...
```

This structured and annotated dataset helps the model learn domain-specific terminology and provide expert-like responses to user questions.

---

## 🚀 What This Solution Offers

* ✅ Custom fine-tuned model for **gym & workout** scenarios
* 📂 One-click training from Excel data
* 🧠 Smart responses tailored to beginner and expert gym-goers
* 📊 Automated model evaluation using validation data
* 📁 All results logged and tracked for transparency

---

## 🧩 How It Works

### 1. **Dataset Preparation**

* An Excel sheet containing Q\&A pairs is processed.
* It is split into **training** and **validation** `.jsonl` files.
* Format follows OpenAI's expected structure using "messages" for dialog.

### 2. **Fine-Tuning the Model**

* Training file is uploaded to OpenAI.
* A fine-tuning job is created and tracked until the model is ready.
* Metadata and model ID are saved for reuse.

### 3. **Model Usage**

* The fine-tuned model responds to **user prompts** related to gym routines.
* Example:
  *“What is the best gym plan for a beginner?”*
  → *“The best plan for a beginner includes 3-day splits focusing on compound movements…”*

### 4. **Evaluation & Quality Scoring**

* Automated evaluation is run on validation examples.
* Each response is compared against expected answers using a **similarity score**.
* Reports are saved as both JSON and CSV for easy review.

---

## 📁 Folder Structure

```
project-root/
├── dataset/
│   └── input_data/gym_workout_dataset.xlsx
├── finetuning/
│   └── finetune_details/
│       ├── finetune_job_metadata.json
│       └── finetuned_model_name.txt
├── results/
│   └── evaluation_report.json
│   └── evaluation_report.csv
├── config/
│   └── config.py
```

---

## ✅ Key Benefits for Your Business

| Benefit               | Description                                                |
| --------------------- | ---------------------------------------------------------- |
| 🌟 Accuracy           | Fine-tuned on your domain, answers are highly relevant.    |
| 🕒 Time-Saving        | Model is reusable. No need to retrain for each session.    |
| 📊 Evaluation Reports | Transparency on how well the model performs.               |
| 👥 Personalization    | Model can be aligned to your brand tone or training style. |

---

## 🛠 How to Run

1. Place your gym Q\&A Excel file at:
   `dataset/input_data/gym_workout_dataset.xlsx`

2. Run the script:

   ```bash
   python main.py
   ```

3. Interact with the model or run full validation automatically.

---

## 📌 Notes

* Environment variables are loaded from `.env` and `config.py`
* Model waits automatically until fine-tuning completes (or times out)
* Easy option to test with your own prompt or use a validation set

---

## 📬 Final Output

* ✅ A ready-to-use OpenAI fine-tuned model
* 📁 Evaluation reports with performance insights
* 💬 Real-time interactive query capability
