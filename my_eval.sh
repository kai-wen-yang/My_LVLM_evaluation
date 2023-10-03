export TORCH_HUB=/fs/nexus-scratch/kwyang3/models
export TRANSFORMERS_CACHE=/fs/nexus-scratch/kwyang3/models
export HF_DATASETS_CACHE=/fs/nexus-scratch/kwyang3/data

cd /fs/nexus-scratch/kwyang3/My_LVLM_evaluation/models
export PYTHONPATH="$PYTHONPATH:$PWD"

cd ..

#python my_eval.py --model_name Cheetah --batch_size 64 --eval_cls --dataset_name Flowers102 --device 0 --sample_num 50000

python my_eval.py --model_name Cheetah --batch_size 64 --eval_cls --dataset_name ImageNet --device 0 --sample_num 50000

