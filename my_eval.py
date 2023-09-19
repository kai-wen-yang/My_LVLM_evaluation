import clip.clip as clip
import os
import json
import argparse
import datetime
from functools import partial

import torch
import numpy as np

from utils import evaluate_OCR, evaluate_VQA, evaluate_Caption, evaluate_KIE, evaluate_MRR, evaluate_embodied, evaluate_zero_shot_image_classification
from task_datasets import ocrDataset, dataset_class_dict
from models import get_model
torch.hub.set_dir('/fs/nexus-scratch/kwyang3/models')
import pdb

from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Optional
from collections import defaultdict
from imagenet_classnames import openai_classnames
imagenet_templates = [
    'a photo of a {}.',
    'a photo of {}.',
]
from model import get_image

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
    parser.add_argument("--eval_ocr", action="store_true", help="Whether to evaluate on ocr.")
    parser.add_argument("--eval_vqa", action="store_true", help="Whether to evaluate on vqa.")
    parser.add_argument("--eval_caption", action="store_true", help="Whether to evaluate on caption.")
    parser.add_argument("--eval_kie", action="store_true", default=False, help="Whether to evaluate on kie.")
    parser.add_argument("--eval_mrr", action="store_true", default=False, help="Whether to evaluate on mrr.")
    parser.add_argument("--eval_embod", action="store_true", default=False, help="Whether to evaluate on embodied.")
    parser.add_argument("--eval_cls", action="store_true", default=False, help="Whether to evaluate on zero-shot classification.")

    args = parser.parse_args()
    return args


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))


def sample_dataset(dataset, max_sample_num=5000, seed=0):
    if max_sample_num == -1:
        return dataset

    if len(dataset) > max_sample_num:
        np.random.seed(seed)
        random_indices = np.random.choice(
            len(dataset), max_sample_num, replace=False
        )
        dataset = torch.utils.data.Subset(dataset, random_indices)
    return dataset


def zeroshot_classifier(classnames, templates):
	with torch.no_grad():
		zeroshot_weights = []
		i = 0
		for classname in tqdm(classnames):
			texts = [template.format(classname) for template in templates] #format with class
			texts = clip_model.tokenize(texts).cuda() #tokenize
			class_embeddings = clip_model.encode_text(texts) #embed with text encoder
			class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
			class_embedding = class_embeddings.mean(dim=0)
			class_embedding /= class_embedding.norm()
			zeroshot_weights.append(class_embedding)
			i += 1
		zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
	return zeroshot_weights


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    clip_model, train_preprocess, val_preprocess = clip.load(
      "ViT-B/16")
    clip_model.eval()
    clip_model.cuda()

    dataset = dataset_class_dict[args.dataset_name]()
    dataset = sample_dataset(dataset, args.sample_num, args.sample_seed)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
    zeroshot_weights_base = zeroshot_classifier(openai_classnames, imagenet_templates)
    	 
    outputs=[]
    targets=[]
    for batch in tqdm(dataloder):
        image_path, _, y = batch
        image = get_image(image_path)
        images = images.cuda()
        target = target.cuda()

        # predict
        image_features = clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        logits_base = image_features @ zeroshot_weights_base
        outputs.append(logits_base)
        targets.append(targets)
    acc=accuracy(torch.cat(outputs,dim=0), torch.cat(targets,dim=0))
    print(acc)
	
	    

if __name__ == "__main__":
    args = parse_args()
    main(args)
