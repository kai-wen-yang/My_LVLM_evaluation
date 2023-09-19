export TORCH_HUB=/fs/nexus-scratch/kwyang3/models
export TRANSFORMERS_CACHE=/fs/nexus-scratch/kwyang3/models
export HF_DATASETS_CACHE=/fs/nexus-scratch/kwyang3/data

cd /fs/nexus-scratch/kwyang3/My_LVLM_evaluation/models
export PYTHONPATH="$PYTHONPATH:$PWD"

cd ..

python eval.py --model_name InstructBLIP --batch_size 16 --eval_cls_clip --dataset_name ImageNet --device 0 --sample_num 1000 --max_new_tokens 64 
#--cot 'Let us think step by step. '

#python eval.py --model_name InstructBLIP --batch_size 16 --eval_cls_clip --dataset_name ImageNet --device 0 --sample_num 1000 --max_new_tokens 64 \
#--cot 'Sub-question 1: What is the primary color of the object?\nSub-question 2: What is the shape of the object?\nSub-question 3: Is the object typically found indoors or outdoors?\nSub-question 4: Does the object have any text or writing on it?\nSub-question 5: Is the object commonly used for a specific purpose?\nSub-question 6: Does the object have any distinct patterns or designs on it?\n '

#python eval.py --model_name Cheetah --batch_size 8 --eval_cls --dataset_name ImageNet --device 0 --sample_num 1000
#python eval.py --model_name InstructBLIP --batch_size 16 --eval_cls --dataset_name ImageNet --device 0 --sample_num 1000

#python eval.py --model_name InstructBLIP --batch_size 2 --eval_vqa --dataset_name VCR1_MCI --device 0 --sample_num 1000

#python eval.py --model_name InstructBLIP --batch_size 16 --eval_cls --dataset_name ImageNet --device 0 --sample_num 1000

#python eval.py --model_name InstructBLIP --batch_size 2 --eval_vqa --dataset_name ScienceQA --device 0 --sample_num 1000
