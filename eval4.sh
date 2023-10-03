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

python eval.py --model_name InstructBLIP --batch_size 8 --eval_cls_clip --dataset_name ImageNetOption --device 0 --max_new_tokens 64 \
--expname 7b_top2_option_background1 --top_option 2 --sample_num 1000 \
--cot 'Is the lighting in the image natural (e.g., sunlight) or artificial (e.g., indoor lighting)? Does the image appear to be taken indoors or outdoors? Are there any notable shadows cast in the image? Does the image appear to be taken during the day or at night? '

python eval.py --model_name InstructBLIP --batch_size 8 --eval_cls_clip --dataset_name ImageNetOption --device 0 --max_new_tokens 64 \
--expname 7b_top3_option_background1 --top_option 3 --sample_num 1000 \
--cot 'Is the lighting in the image natural (e.g., sunlight) or artificial (e.g., indoor lighting)? Does the image appear to be taken indoors or outdoors? Are there any notable shadows cast in the image? Does the image appear to be taken during the day or at night? '

python eval.py --model_name InstructBLIP --batch_size 8 --eval_cls_clip --dataset_name ImageNetOption --device 0 --max_new_tokens 64 \
--expname 7b_top4_option_background1 --top_option 4 --sample_num 1000 \
--cot 'Is the lighting in the image natural (e.g., sunlight) or artificial (e.g., indoor lighting)? Does the image appear to be taken indoors or outdoors? Are there any notable shadows cast in the image? Does the image appear to be taken during the day or at night? '




#python eval.py --model_name InstructBLIP --batch_size 4 --eval_vqa --dataset_name TextVQA --device 0 \
#--expname 13b_context --sample_num 500 \
#--cot 'huaping'
#
#python eval.py --model_name InstructBLIP --batch_size 4 --eval_vqa --dataset_name TextVQA --device 0 \
#--expname 13b_base --sample_num 500
#
#python eval.py --model_name InstructBLIP --batch_size 4 --eval_vqa --dataset_name VCR1_MCI --device 0 \
#--expname 13b_context --sample_num 500 \
#--cot 'huaping'
#
#python eval.py --model_name InstructBLIP --batch_size 4 --eval_vqa --dataset_name VCR1_MCI --device 0 \
#--expname 13b_base --sample_num 500
#
#python eval.py --model_name InstructBLIP --batch_size 4 --eval_vqa --dataset_name VCR1_OC --device 0 \
#--expname 13b_context --sample_num 500 \
#--cot 'huaping'
#
#python eval.py --model_name InstructBLIP --batch_size 4 --eval_vqa --dataset_name VCR1_OC --device 0 \
#--expname 13b_base --sample_num 500




##python eval.py --model_name InstructBLIP --batch_size 16 --eval_cls_clip --dataset_name ImageNet --device 0 --sample_num 50000 --max_new_tokens 64 \
##--expname backques1 \
##--cot 'Is the lighting in the image natural (e.g., sunlight) or artificial (e.g., indoor lighting)? Does the image appear to be taken indoors or outdoors? Are there any notable shadows cast in the image? Does the image appear to be taken during the day or at night? '
##
##python eval.py --model_name InstructBLIP --batch_size 16 --eval_cls_clip --dataset_name ImageNet --device 0 --sample_num 50000 --max_new_tokens 64 \
##--expname backques2 \
##--cot 'Is the lighting in the image natural (e.g., sunlight) or artificial (e.g., indoor lighting)? Does the image appear to be taken indoors or outdoors? '
##
##python eval.py --model_name InstructBLIP --batch_size 16 --eval_cls_clip --dataset_name ImageNet --device 0 --sample_num 50000 --max_new_tokens 64 \
##--expname backques3 \
##--cot 'Is the lighting in the image natural (e.g., sunlight) or artificial (e.g., indoor lighting)? '
##
##python eval.py --model_name InstructBLIP --batch_size 16 --eval_cls_clip --dataset_name ImageNet --device 0 --sample_num 50000 --max_new_tokens 64  --expname baseline \
##
##python eval.py --model_name InstructBLIP --batch_size 1 --eval_cls_clip --dataset_name ImageNet --device 0 --sample_num 50000 --max_new_tokens 64 \
##--expname alques1 \
##--cot 'Sub-question 1: What is the primary color of the object?\nSub-question 2: What is the shape of the object?\nSub-question 3: Is the object typically found indoors or outdoors?\nSub-question 4: Does the object have any text or writing on it?\nSub-question 5: Is the object commonly used for a specific purpose?\nSub-question 6: Does the object have any distinct patterns or designs on it?\nSub-question 7: Is the object made of natural materials or synthetic materials?\nSub-question 8: Does the object have any moving parts or components?\nSub-question 9: Is the object associated with a particular season or holiday?\nSub-question 10: Does the object emit any light or sound?\nMain question: '
#
#python eval.py --model_name InstructBLIP --batch_size 4 --eval_cls_clip --dataset_name ImageNet --device 0 --sample_num 50000 --max_new_tokens 64 \
#--expname alques2 \
#--cot 'What is the primary color of the object? What is the shape of the object? Is the object typically found indoors or outdoors? Does the object have any text or writing on it? Is the object commonly used for a specific purpose? Does the object have any distinct patterns or designs on it? Is the object made of natural materials or synthetic materials? Does the object have any moving parts or components?  Is the object associated with a particular season or holiday? Does the object emit any light or sound? '
#
#
#
##python eval.py --model_name InstructBLIP --batch_size 16 --eval_cls --dataset_name ImageNet --device 0 --sample_num 1000
#
##python eval.py --model_name InstructBLIP --batch_size 2 --eval_vqa --dataset_name VCR1_MCI --device 0 --sample_num 1000
#
##python eval.py --model_name InstructBLIP --batch_size 16 --eval_cls --dataset_name ImageNet --device 0 --sample_num 1000
#
##python eval.py --model_name InstructBLIP --batch_size 2 --eval_vqa --dataset_name ScienceQA --device 0 --sample_num 1000
