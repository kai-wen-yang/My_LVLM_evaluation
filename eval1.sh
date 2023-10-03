#!/bin/bash
DATE=$(date "+%m%d%y_%H_%M_%S")
source ~/.bashrc
module load cuda/11.7.0 cudnn/v8.8.0
conda activate base
python --version
nvidia-smi
export TORCH_HUB=/fs/nexus-scratch/kwyang3/models
export TRANSFORMERS_CACHE=/fs/nexus-scratch/kwyang3/models
export HF_DATASETS_CACHE=/fs/nexus-scratch/kwyang3/data

cd /fs/nexus-scratch/kwyang3/My_LVLM_evaluation/models
export PYTHONPATH="$PYTHONPATH:$PWD"

cd ..

python eval.py --model_name InstructBLIP --batch_size 8 --dataset_name ImageNetOption --device 0 --max_new_tokens 64 \
--expname 7b_top5_option_base --top_option 5 --sample_num 1000 --task_name classification_instruct \
--cot 'Question: What is the object in the image?'

python eval.py --model_name InstructBLIP --batch_size 8 --dataset_name ImageNetOption --device 0 --max_new_tokens 64 \
--expname 7b_top5_option_rationale7 --top_option 5 --sample_num 1000 --task_name classification_instruct \
--cot 'Question: What is the environment or background of the object in the image? Does the image appear to be taken indoors or outdoors? What are the surrounding objects or elements in the image? Does the image appear to be taken during the day or at night? What is the object in the image?'


python eval.py --model_name InstructBLIP --batch_size 8 --dataset_name Flowers102Option --device 0 --max_new_tokens 64 \
--expname 7b_top5_option_base --top_option 5 --sample_num 1000 --task_name classification_instruct \
--cot 'Question: What is the object in the image?'

python eval.py --model_name InstructBLIP --batch_size 8 --dataset_name Flowers102Option --device 0 --max_new_tokens 64 \
--expname 7b_top5_option_rationale7 --top_option 5 --sample_num 1000 --task_name classification_instruct \
--cot 'Question: What is the environment or background of the object in the image? Does the image appear to be taken indoors or outdoors? What are the surrounding objects or elements in the image? Does the image appear to be taken during the day or at night? What is the object in the image?'

python eval.py --model_name InstructBLIP --batch_size 8 --dataset_name IwildOODOption --device 0 --max_new_tokens 64 \
--expname 7b_top5_option_base --top_option 5 --sample_num 1000 --task_name classification_instruct \
--cot 'Question: What is the object in the image?'

python eval.py --model_name InstructBLIP --batch_size 8 --dataset_name IwildOODOption --device 0 --max_new_tokens 64 \
--expname 7b_top5_option_rationale7 --top_option 5 --sample_num 1000 --task_name classification_instruct \
--cot 'Question: What is the environment or background of the object in the image? Does the image appear to be taken indoors or outdoors? What are the surrounding objects or elements in the image? Does the image appear to be taken during the day or at night? What is the object in the image?'


#python eval.py --model_name InstructBLIP --batch_size 8 --dataset_name ImageNetOption --device 0 --max_new_tokens 64 \
#--expname 7b_top5_option_rationale8 --top_option 5 --sample_num 1000 --task_name classification_instruct \
#--cot 'Describe the background and surroundings of the object and than answer the following question: What is the object in the image?'


#
#python eval.py --model_name InstructBLIP --batch_size 8 --dataset_name ImageNetOption --device 0 --max_new_tokens 64 \
#--expname 7b_top5_option_base --top_option 5 --sample_num 1000 --task_name classification_instruct \
#--cot 'Question: What is the object in the image?'
#
#

#python eval.py --model_name InstructBLIP --batch_size 8 --eval_cls_clip --dataset_name ImageNetOption --device 0 --max_new_tokens 64 \
#--expname 7b_top5_option_base --top_option 5 --sample_num 1000
#
#python eval.py --model_name InstructBLIP --batch_size 8 --eval_cls_clip --dataset_name ImageNetOption --device 0 --max_new_tokens 64 \
#--expname 7b_top5_option_background1 --top_option 5 --sample_num 1000 \
#--cot 'Is the lighting in the image natural (e.g., sunlight) or artificial (e.g., indoor lighting)? Does the image appear to be taken indoors or outdoors? Are there any notable shadows cast in the image? Does the image appear to be taken during the day or at night? '
#
#python eval.py --model_name InstructBLIP --batch_size 8 --eval_cls_clip --dataset_name ImageNetOption --device 0 --max_new_tokens 64 \
#--expname 7b_top5_option_background2 --top_option 5 --sample_num 1000 \
#--cot 'What is the environment or background of the object in the image? Does the image appear to be taken indoors or outdoors? What are the surrounding objects or elements in the image? Does the image appear to be taken during the day or at night? '
#
#

#python eval.py --model_name InstructBLIP --batch_size 4 --eval_cls_clip --dataset_name ImageNet --device 0 --sample_num 50000 --max_new_tokens 64 \
#--expname backques1 \
#--cot 'Is the lighting in the image natural (e.g., sunlight) or artificial (e.g., indoor lighting)? Does the image appear to be taken indoors or outdoors? Are there any notable shadows cast in the image? Does the image appear to be taken during the day or at night? '
#
#python eval.py --model_name InstructBLIP --batch_size 4 --eval_cls_clip --dataset_name ImageNet --device 0 --sample_num 50000 --max_new_tokens 64 \
#--expname backques2 \
#--cot 'Is the lighting in the image natural (e.g., sunlight) or artificial (e.g., indoor lighting)? Does the image appear to be taken indoors or outdoors? '
#
##python eval.py --model_name InstructBLIP --batch_size 16 --eval_cls_clip --dataset_name ImageNet --device 0 --sample_num 50000 --max_new_tokens 64 \
##--expname backques3 \
##--cot 'Is the lighting in the image natural (e.g., sunlight) or artificial (e.g., indoor lighting)? '
##
##python eval.py --model_name InstructBLIP --batch_size 16 --eval_cls_clip --dataset_name ImageNet --device 0 --sample_num 50000 --max_new_tokens 64  --expname baseline \
##
##python eval.py --model_name InstructBLIP --batch_size 1 --eval_cls_clip --dataset_name ImageNet --device 0 --sample_num 50000 --max_new_tokens 64 \
##--expname alques1 \
##--cot 'Sub-question 1: What is the primary color of the object?\nSub-question 2: What is the shape of the object?\nSub-question 3: Is the object typically found indoors or outdoors?\nSub-question 4: Does the object have any text or writing on it?\nSub-question 5: Is the object commonly used for a specific purpose?\nSub-question 6: Does the object have any distinct patterns or designs on it?\nSub-question 7: Is the object made of natural materials or synthetic materials?\nSub-question 8: Does the object have any moving parts or components?\nSub-question 9: Is the object associated with a particular season or holiday?\nSub-question 10: Does the object emit any light or sound?\nMain question: '
##
##python eval.py --model_name InstructBLIP --batch_size 1 --eval_cls_clip --dataset_name ImageNet --device 0 --sample_num 50000 --max_new_tokens 64 \
##--expname alques2 \
##--cot 'What is the primary color of the object? What is the shape of the object? Is the object typically found indoors or outdoors? Does the object have any text or writing on it? Is the object commonly used for a specific purpose? Does the object have any distinct patterns or designs on it? Is the object made of natural materials or synthetic materials? Does the object have any moving parts or components?  Is the object associated with a particular season or holiday? Does the object emit any light or sound? '
#
#
#
##python eval.py --model_name InstructBLIP --batch_size 16 --eval_cls --dataset_name ImageNet --device 0 --sample_num 1000
#
##python eval.py --model_name InstructBLIP --batch_size 2 --eval_vqa --dataset_name VCR1_MCI --device 0 --sample_num 1000
#
##python eval.py --model_name InstructBLIP --batch_size 16 --eval_cls --dataset_name ImageNet --device 0 --sample_num 1000
#
