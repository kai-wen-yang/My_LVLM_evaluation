import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Optional
from collections import defaultdict
import pdb
import torch
from .tools import has_word, remove_special_chars
import sys 
sys.path.append("..") 
from my_eval import clip, zeroshot_classifier, openai_classnames, imagenet_templates
from models import get_image


def evaluate_zero_shot_image_classification_clip(
    model,
    dataset,
    model_name,
    dataset_name,
    time,
    batch_size=1,
    answer_path='answers',
    question='What is the object in the image?',
    max_new_tokens=16,
    per_class_acc=True,
):
    ## load clip model
    clip_model, train_preprocess, val_preprocess = clip.load("ViT-B/16", 'cuda', jit=False)
    clip_model.eval()
    clip_model.cuda()
    zeroshot_weights_base = zeroshot_classifier(clip_model, openai_classnames, imagenet_templates)
  
    predictions=[]
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
    i = 0
    for batch in tqdm(dataloader, desc="Running inference"):
        ## clip inference ##
        images = [get_image(img) for img in batch['image_path']]
        images = [val_preprocess(x) for x in images]
        images = torch.stack(images, dim=0).cuda()

        with torch.no_grad():
             image_features = clip_model.encode_image(images)
             image_features /= image_features.norm(dim=-1, keepdim=True)

        logits_base = image_features @ zeroshot_weights_base
        _, y_pred = logits_base.topk(k=5, dim=1)
        questions=[]
        for i in range(y_pred.size(0)):
             options = '\n- '.join([openai_classnames[ind] for ind in y_pred[i].tolist()])
             questions.append(f"Question: What is the object in the image?\nChoose the best answer from the following choices:\n- {options}")	
        ####
        pdb.set_trace()
        outputs = model.batch_generate(batch['image_path'], questions, max_new_tokens=max_new_tokens)
        j = 0
        for image_path, gt_answer, output in zip(batch['image_path'], batch['gt_answers'], outputs):
            if type(image_path) is not str:
                image_path = f'batch#{i} sample#{j}'
            answer_dict={'question': question, 'answer': output,
            'gt_answers': gt_answer, 'image_path': image_path,
            'model_name': model_name}
            predictions.append(answer_dict)
            j += 1
        i += 1
    answer_dir = os.path.join(answer_path, time)
    os.makedirs(answer_dir, exist_ok=True)
    answer_path = os.path.join(answer_dir, f"{dataset_name}.json")
    with open(answer_path, "w") as f:
        f.write(json.dumps(predictions, indent=4))
    correct = 0
    num = 0
    exact_match = 0
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
            num+=1
    acc_has_word = correct / num * 100
    acc_exact_match = exact_match / num * 100
    print(f'{dataset_name} of has_word: {acc_has_word:.2f}%')
    print(f'{dataset_name} of exact match: {acc_exact_match:.2f}%')

    metrics = {
        'has_word': acc_has_word,
        'exact match': acc_exact_match,
    }
    if per_class_acc:
        num_classes = len(per_class_dict)
        acc_sum = 0.0
        for val in per_class_dict.values():
            acc_sum += val['correct'] / val['total']
        mean_per_class_acc = acc_sum / num_classes * 100
        metrics['mean_per_class_acc'] = mean_per_class_acc
        print(f'{dataset_name} of mean per-class: {mean_per_class_acc:.2f}%')
    return metrics
