export TORCH_HUB=/fs/nexus-scratch/kwyang3/models
export TRANSFORMERS_CACHE=/fs/nexus-scratch/kwyang3/models
export HF_DATASETS_CACHE=/fs/nexus-scratch/kwyang3/data

python blip_gpt_main.py  \
    --data_root=/fs/nexus-scratch/kwyang3/data/VCR \
    --exp_tag=vcr_blip2 \
    --dataset=vcr_val \
    --device_id=0 \
    --prompt_setting=v1a \
    --data_partition=0_10 \
    --vqa_model=instructblip  \
    --temp_gpt=0.0  \
    --data_subset=./misc/exp_data/vcr_val_random500_annoid.yaml  \
    --openai_key="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"