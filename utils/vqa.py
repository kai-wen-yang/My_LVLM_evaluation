import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

from .tools import VQAEval
import pdb

def evaluate_VQA(
    model,
    dataset,
    model_name,
    dataset_name,
    time,
    batch_size=1,
    answer_path='./answers',
    max_new_tokens=256,
    cot=None,
    cot_position='end',
    args=None
):
    predictions=[]
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda batch: {key: [dict[key] for dict in batch] for key in batch[0]})
    for batch in tqdm(dataloader, desc="Running inference"):

        if not cot:
            questions = batch['question']
        else:
            pre_questions = []
            # for i in range(len(batch['image_path'])):
            #     pre_questions.append(cot)
            # pre_outputs = model.batch_generate(batch['image_path'], pre_questions, max_new_tokens=128)
            questions = batch['question']
            for i in range(len(batch['image_path'])):
                pre_questions.append(f'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions.\\nUSER: You are an AI assistant who has rich visual knowledge and strong reasoning abilities.\nYour goal is to help me answer a question about an image.\nThe question is:{questions[i]}\nYou should tell me which part of the image should I focus and which visual features should I based on to answer the question. You should answer in short.\nASSISTANT:')
            pre_outputs = model.text_generate(pre_questions)

            questions = []
            for i in range(len(batch['image_path'])):
                questions.append(f"{pre_outputs[i]}\n{batch['question'][i]}")

        outputs = model.batch_generate(batch['image_path'], questions, max_new_tokens=max_new_tokens)
        for image_path, question, gt_answer, output in zip(batch['image_path'], batch['question'], batch['gt_answers'], outputs):
            answer_dict={'question': question, 'answer': output,
            'gt_answers': gt_answer, 'image_path': image_path,
            'model_name': model_name}
            predictions.append(answer_dict)
    answer_dir = os.path.join(answer_path, time)
    os.makedirs(answer_dir, exist_ok=True)
    answer_path = os.path.join(answer_dir, f"{dataset_name}.json")
    with open(answer_path, "w") as f:
        f.write(json.dumps(predictions, indent=4))
    eval = VQAEval()
    correct = 0
    num = 0
    with open(answer_path, 'r') as f:
        dict = json.load(f)
        for i in range(len(dict)):
            gt_answers = dict[i]['gt_answers']
            answer = dict[i]['answer']
            if eval.evaluate(answer, gt_answers)==1:
                correct+=1
            num+=1
    print(f'{dataset_name}:{float(correct)/num}')
    return float(correct)/num