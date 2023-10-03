#!/bin/bash
DATE=$(date "+%m%d%y_%H_%M_%S")
source ~/.bashrc
module load cuda/11.7.0 cudnn/v8.8.0
conda activate base
python --version
nvidia-smi

export PYTHONPATH="$PYTHONPATH:$PWD"
ln -s /fs/nexus-scratch/kwyang3/data/iwildcam_v2.0 ./datasets/data/iwildcam_v2.0

python src/main.py --train-dataset=IWildCamIDVal --epochs=20 --lr=1e-5 --wd=0.2 --batch-size=128 --model=ViT-B/16 \
--eval-datasets=IWildCamID,IWildCamOOD --template=iwildcam_template  --save=./checkpoints/ \
--data-location=./datasets/data/ --ft_data="./datasets/csv/iwildcam_v2.0/train.csv" --csv-img-key filepath --csv-caption-key title --exp_name=iwildcam/main
