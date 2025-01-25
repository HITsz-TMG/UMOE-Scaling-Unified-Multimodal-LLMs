import os
import argparse
import json
import re

from m4c_evaluator import TextVQAAccuracyEvaluator


# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--annotation-file', type=str)
#     parser.add_argument('--result-file', type=str)
#     parser.add_argument('--result-dir', type=str)
#     return parser.parse_args()


def prompt_processor(prompt):
    if prompt.startswith('OCR tokens: '):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        question = match.group(1)
    elif 'Reference OCR token: ' in prompt and len(prompt.split('\n')) == 3:
        if prompt.startswith('Reference OCR token:'):
            question = prompt.split('\n')[1]
        else:
            question = prompt.split('\n')[0]
    elif len(prompt.split('\n')) == 2:
        question = prompt.split('\n')[0]
    else:
        assert False

    return question.lower()


def eval_single(annotation_file, result_file):
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    print(experiment_name)
    annotations = json.load(open(annotation_file))['data']
    annotations = {(annotation['image_id'], annotation['question'].lower()): annotation for annotation in annotations}
    results = json.load(open(result_file))

    pred_list = []
    for result in results:
        annotation = annotations[(result['question_id'], prompt_processor(result['text']))]
        pred_list.append({
            "pred_answer": result['answer'],
            "gt_answers": annotation['answers'],
        })

    evaluator = TextVQAAccuracyEvaluator()
    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * evaluator.eval_pred_list(pred_list)))


if __name__ == "__main__":
    # args = get_args()
    eval_single("TextVQA_0.5.1_val.json", \
                "textvqa_result.json")