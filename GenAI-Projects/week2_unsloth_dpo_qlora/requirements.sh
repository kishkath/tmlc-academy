%%capture
!pip install pip3-autoremove
!pip install datasets trl wandb fastapi uvicorn pydantic loguru
!pip-autoremove torch torchvision torchaudio -y
!pip install "torch==2.4.0" "xformers==0.0.27.post2" triton torchvision torchaudio
!pip install "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install numpy scipy
