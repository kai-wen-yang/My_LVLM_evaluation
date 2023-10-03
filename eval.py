import os
import json
import argparse
import datetime
from functools import partial

import torch
import numpy as np

from utils import task_class_dict
from task_datasets import ocrDataset, dataset_class_dict
from models import get_model
import wandb
torch.hub.set_dir('/fs/nexus-scratch/kwyang3/models')
import pdb


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    # models
    parser.add_argument("--model_name", type=str, default="LLaMA-Adapter-v2")
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=-1)

    # datasets
    parser.add_argument("--ocr_dataset_name", type=str, default="IIIT5K SVT IC13 IC15 SVTP CUTE80 COCO-Text Total-Text WordArt CTW HOST WOST")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--sample_num", type=int, default=-1)
    parser.add_argument("--sample_seed", type=int, default=0)

    # result_path
    parser.add_argument("--answer_path", type=str, default="./answers")

    # eval choices
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--cot", type=str, default=None)
    parser.add_argument("--cot_position", type=str, default='begin')
    parser.add_argument("--expname", type=str, default=None)
    parser.add_argument("--top_option", type=int, default=5)
    args = parser.parse_args()
    return args


def sample_dataset(dataset, max_sample_num=5000, seed=0):
    if max_sample_num == -1:
        return dataset

    if len(dataset) > max_sample_num:

        drop_size = len(dataset) - max_sample_num
        dataset, _ = torch.utils.data.random_split(dataset, [max_sample_num, drop_size],
                                                                    generator=torch.Generator().manual_seed(42))
    return dataset


def get_eval_function(args):
    eval_func = task_class_dict[args.task_name]

    if args.max_new_tokens != -1:
        eval_func = partial(eval_func, max_new_tokens=args.max_new_tokens)
    
    if args.question is not None:
        eval_func = partial(eval_func, question=args.question)
    if args.cot is not None:
        eval_func = partial(eval_func, cot=args.cot, cot_position=args.cot_position)
    
    return eval_func


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    model = get_model(args.model_name)
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    result = {}

    eval_function = get_eval_function(args)

    args.expname = args.dataset_name + args.task_name + args.expname
    wandb.init(name=args.expname, config=args)
    answer_path = f"{args.answer_path}/{args.model_name}/{args.expname}"

    if eval_function is not None:
        dataset = dataset_class_dict[args.dataset_name]()
        dataset = sample_dataset(dataset, args.sample_num, args.sample_seed)
        metrics = eval_function(model, dataset, args.model_name, args.dataset_name, time, args.batch_size, answer_path=answer_path, args=args)
        result[args.dataset_name] = metrics
    
    result_path = os.path.join(os.path.join(answer_path, time), 'result.json')
    with open(result_path, "w") as f:
        f.write(json.dumps(result, indent=4))


if __name__ == "__main__":
    args = parse_args()
    main(args)
