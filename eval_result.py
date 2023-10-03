import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Optional
from collections import defaultdict
import pdb
import torch
from utils.tools import has_word, remove_special_chars
import sys
sys.path.append("..")
from my_eval import clip, zeroshot_classifier, openai_classnames, imagenet_templates
from models import get_image
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    # models
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--answer_path", type=str, default=None)
    args = parser.parse_args()
    return args

def main(args):
    answer_path = args.answer_path
    dataset_name = 'imagenet'
    correct = 0
    num = 0
    exact_match = 0
    clip_match = 0
    clip_unmatch_llm_match = 0
    clip_unmatch = 0
    clip_match_conf = 0
    clip_unmatch_conf = 0
    high_clip_low_llm = 0
    per_class_dict = defaultdict(lambda : defaultdict(int))
    with open(answer_path, 'r') as f:
        dict = json.load(f)
        for i in range(len(dict)):

            answer = dict[i]['answer']
            answer = remove_special_chars(answer).lower()
            gt_answers = dict[i]['gt_answers']
            if type(gt_answers) is str:
                cls_name = gt_answers
                gt_answers = [remove_special_chars(gt_answers).lower()]
            else:
                cls_name = gt_answers[0]
                gt_answers = [remove_special_chars(x).lower() for x in gt_answers]
            per_class_dict[cls_name]['total'] += 1
            if any([has_word(answer, x) for x in gt_answers]):
                per_class_dict[cls_name]['correct'] += 1
                correct+=1
            if any([answer == x for x in gt_answers]):
                exact_match += 1
            if dict[i]['clip_prediction'] == openai_classnames[dict[i]['label']]:
                clip_match += 1
                clip_match_conf += dict[i]['confidence']
            else:
                clip_unmatch += 1
                clip_unmatch_conf += dict[i]['confidence']
                if any([has_word(answer, x) for x in gt_answers]):
                    clip_unmatch_llm_match+=1
            if (dict[i]['confidence']>args.confidence and dict[i]['clip_prediction'] == openai_classnames[dict[i]['label']]) or (dict[i]['confidence']<=args.confidence and any([has_word(answer, x) for x in gt_answers])):
                high_clip_low_llm +=1

            num+=1
    acc_has_word = correct / num * 100
    acc_exact_match = exact_match / num * 100
    print(f'{dataset_name} of has_word: {acc_has_word:.2f}%')
    print(f'{dataset_name} of exact match: {acc_exact_match:.2f}%')
    print(f'{dataset_name} of clip match: {clip_match / num * 100:.2f}%')
    print(f'{dataset_name} of high clip low llm: {high_clip_low_llm / num * 100:.2f}%')
    print(f'{dataset_name} of clip unmatch: {clip_unmatch / num * 100:.2f}%')
    print(f'{dataset_name} of clip unmatch llm match: {clip_unmatch_llm_match / clip_unmatch * 100:.2f}%')
    print(f'{dataset_name} of clip match confidence: {clip_match_conf / clip_match:.2f}%')
    print(f'{dataset_name} of clip unmatch confidence: {clip_unmatch_conf / clip_unmatch:.2f}%')


if __name__ == "__main__":
    args = parse_args()
    main(args)
