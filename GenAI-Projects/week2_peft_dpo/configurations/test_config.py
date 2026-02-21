import json


class TestConfig:
    """
    Loads inference configuration from a JSON file.
    Supports:
      - mode: 'lora' or 'merged'
      - base_model_name
      - adapter_path
      - merged_model_path
      - generation parameters (max_new_tokens, temperature, top_p)
      - list of questions
    """

    def __init__(self, path="week2_peft_dop/configurations/inference_configs.json"):
        with open(path, "r") as f:
            cfg = json.load(f)

        self.mode = cfg.get("mode", "lora")
        self.base_model_name = cfg.get("base_model_name")
        self.adapter_path = cfg.get("adapter_path")
        self.merged_model_path = cfg.get("merged_model_path", "./merged_model")
        self.generation = cfg.get("generation", {"max_new_tokens": 512, "temperature": 0.7, "top_p": 0.9})
        self.questions = cfg.get("questions", [])




