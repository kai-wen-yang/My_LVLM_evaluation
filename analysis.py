import os
import json
import argparse
import datetime
from functools import partial

import torch
import numpy as np

from utils import evaluate_OCR, evaluate_VQA, evaluate_Caption, evaluate_KIE, evaluate_MRR, evaluate_embodied, \
    evaluate_zero_shot_image_classification, evaluate_zero_shot_image_classification_clip
from task_datasets import ocrDataset, dataset_class_dict
from models import get_model
import wandb
from my_eval import clip, zeroshot_classifier, openai_classnames, imagenet_templates
torch.hub.set_dir('/fs/nexus-scratch/kwyang3/models')
import pdb
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--dataset_name", type=str, default='ImageNetOption')
    args = parser.parse_args()
    return args


def main(args):
    dataset = dataset_class_dict[args.dataset_name]()
    unmatch = []
    match = []
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        if data['options'][0] == openai_classnames[data['label']]:
            match.append(data['confidence'])
        else:
            unmatch.append(data['confidence'])
    print(f'matched sample: {len(match)}')
    print(f'unmatched sample: {len(unmatch)}')

    plt.hist(match, bins=100, alpha=0.5, label='correct')
    plt.hist(unmatch, bins=100, alpha=0.5, label='wrong')

    plt.title('clip confidence')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('figures/clip_conf.png')


if __name__ == "__main__":
    args = parse_args()
    main(args)
