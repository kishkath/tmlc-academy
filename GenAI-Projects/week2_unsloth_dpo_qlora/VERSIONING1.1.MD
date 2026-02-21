# Version 1.1 Change Documentation (Short)

## config.json

| Change                                                                   | Purpose                            |
| ------------------------------------------------------------------------ | ---------------------------------- |
| Added `eval_steps: 400`                                                  | Enable periodic validation         |
| Warmup ratio → 0.2, epochs → 4, learning rate → 3e-5, scheduler → cosine | Better convergence and stable loss |

## trainer.py

| Change                                              | Purpose                               |
| --------------------------------------------------- | ------------------------------------- |
| eval_dataset → val                                  | Correct validation dataset            |
| Added eval_steps, validation loop, final validation | Track validation metrics periodically |
| WandB validation logging                            | Better experiment tracking            |
| Improved logs, safety check                         | Debug clarity and error prevention    |

## main.py

| Change                         | Purpose              |
| ------------------------------ | -------------------- |
| Updated for trainer.py changes | Ensure compatibility |

## dataset_preparation.py

| Change                       | Purpose                  |
| ---------------------------- | ------------------------ |
| Renamed test → val           | Consistency              |
| Added cleaning and filtering | Data quality improvement |

## model_loader.py

| Change            | Purpose                        |
| ----------------- | ------------------------------ |
| Minor adjustments | Ensure model loading stability |

## inference_utils.py

| Change           | Purpose             |
| ---------------- | ------------------- |
| No major changes | Maintain efficiency |

## inference.py

| Change           | Purpose                |
| ---------------- | ---------------------- |
| No major changes | Maintain compatibility |

**Summary:** Added validation steps, improved config, dataset cleaning, and logging for better training tracking and stability.
