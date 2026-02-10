# *[Specify the config file path and the GPU devices to use]
export CUDA_VISIBLE_DEVICES=3

# *[Specify the config file path]
export OMINI_CONFIG=./train/config/fill_subject_signs.yaml

# *[Specify the WANDB API key]
# export WANDB_API_KEY='YOUR_WANDB_API_KEY'

export XDG_CACHE_HOME=/export/scratch/emironov/cache
export HF_HOME=/export/scratch/emironov/huggingface
export HF_DATASETS_CACHE=/export/scratch/emironov/huggingface/datasets

echo $OMINI_CONFIG
export TOKENIZERS_PARALLELISM=true

accelerate launch --main_process_port 41353 -m omini.train_flux.train_fill_subject_with_signs
