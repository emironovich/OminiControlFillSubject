# *[Specify the config file path and the GPU devices to use]
export CUDA_VISIBLE_DEVICES=

export XDG_CACHE_HOME=/export/scratch/emironov/cache
export HF_HOME=/export/scratch/emironov/huggingface
export HF_DATASETS_CACHE=/export/scratch/emironov/huggingface/datasets

export TOKENIZERS_PARALLELISM=true

accelerate launch --main_process_port 41353 -m omini.train_flux.process_200k
