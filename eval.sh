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
--expname 7b_top5_option_base_answer --top_option 5 --sample_num 1000 \
--cot 'Questions: what is the object in the image?'

python eval.py --model_name InstructBLIP --batch_size 8 --eval_cls_clip --dataset_name ImageNetOption --device 0 --max_new_tokens 64 \
--expname 7b_top5_option_instruct5 --top_option 5 --sample_num 1000 \
--cot 'Choose the most likely answer from the given choices to answer the following questions: what is the object in the image?'

python eval.py --model_name InstructBLIP --batch_size 8 --eval_cls_clip --dataset_name ImageNetOption --device 0 --max_new_tokens 64 \
--expname 7b_top5_option_instruct6 --top_option 5 --sample_num 1000 \
--cot 'Choose the most likely answer from the given choices to answer the following questions. Your final answer should be in the form \boxed{{answer}}\nQuestion : what is the object in the image?'


#

#python eval.py --model_name InstructBLIP --batch_size 4 --eval_cls_clip --dataset_name ImageNetOption --device 0 --max_new_tokens 64 \
#--expname 13b_top10_general2 --top_option 10 --sample_num 1000 \
#--cot '''Look at the object in the picture and its surrounding background area, and then based on the information about the object itself and other background information to answer the following question: '''
#
#python eval.py --model_name InstructBLIP --batch_size 8 --eval_vqa --dataset_name TextVQA --device 0 \
#--expname back --sample_num 500
#
#python eval.py --model_name InstructBLIP --batch_size 8 --eval_cls_clip --dataset_name ImageNetOption --device 0 --max_new_tokens 64 \
#--expname 13b_top10_option --top_option 10 --sample_num 1000 \
#--cot 'Is the lighting in the image natural (e.g., sunlight) or artificial (e.g., indoor lighting)? Does the image appear to be taken indoors or outdoors? Are there any notable shadows cast in the image? Does the image appear to be taken during the day or at night? '
#
#
#
#python eval.py --model_name InstructBLIP --batch_size 8 --eval_vqa --dataset_name TextVQA --device 0 \
#--expname back --sample_num 1000 \
#--cot 'Is the lighting in the image natural (e.g., sunlight) or artificial (e.g., indoor lighting)? Does the image appear to be taken indoors or outdoors? Are there any notable shadows cast in the image? Does the image appear to be taken during the day or at night? '
#
##python eval.py --model_name InstructBLIP --batch_size 8 --eval_cls_clip --dataset_name ImageNetOption --device 0 --max_new_tokens 64 \
##--expname 13b_top3_option --top_option 3 --sample_num 1000 \
##--cot 'Is the lighting in the image natural (e.g., sunlight) or artificial (e.g., indoor lighting)? Does the image appear to be taken indoors or outdoors? Are there any notable shadows cast in the image? Does the image appear to be taken during the day or at night? '
#
##python eval.py --model_name InstructBLIP --batch_size 8 --eval_cls_clip --dataset_name ImageNetOption --device 0 --max_new_tokens 64 \
##--expname 13b_top4_option --top_option 4 --sample_num 1000 \
##--cot 'Is the lighting in the image natural (e.g., sunlight) or artificial (e.g., indoor lighting)? Does the image appear to be taken indoors or outdoors? Are there any notable shadows cast in the image? Does the image appear to be taken during the day or at night? '
#
#python eval.py --model_name InstructBLIP --batch_size 8 --eval_cls_clip --dataset_name ImageNetOption --device 0 --max_new_tokens 64 \
#--expname 13b_top5_option --top_option 5 --sample_num 1000 \
#--cot 'Is the lighting in the image natural (e.g., sunlight) or artificial (e.g., indoor lighting)? Does the image appear to be taken indoors or outdoors? Are there any notable shadows cast in the image? Does the image appear to be taken during the day or at night? '
#
