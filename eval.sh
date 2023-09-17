export TORCH_HUB=/fs/nexus-scratch/kwyang3/models
export TRANSFORMERS_CACHE=/fs/nexus-scratch/kwyang3/models
export HF_DATASETS_CACHE=/fs/nexus-scratch/kwyang3/data

python eval.py --model_name Cheetah --batch_size 16 --eval_cls --dataset_name ImageNet --device 0 --sample_num 1000

#python eval.py --model_name InstructBLIP --batch_size 2 --eval_vqa --dataset_name VCR1_MCI --device 0 --sample_num 1000

#python eval.py --model_name InstructBLIP --batch_size 16 --eval_cls --dataset_name ImageNet --device 0 --sample_num 1000

#python eval.py --model_name InstructBLIP --batch_size 2 --eval_vqa --dataset_name ScienceQA --device 0 --sample_num 1000