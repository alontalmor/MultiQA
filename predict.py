import argparse
import models
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.common.file_utils import cached_path
from  datasets.multiqa_factory import MultiQAFactory
from common.official_eval import read_answers
from common.official_eval import evaluate
import numpy as np
import os
import gzip
import json
from allennlp.common.tqdm import Tqdm

if __name__ == "__main__":
    parse = argparse.ArgumentParser("")
    parse.add_argument("model")
    parse.add_argument("multiqa_dataset")
    parse.add_argument("dataset_name")
    parse.add_argument("--prediction_filepath", type=str, default=None)
    parse.add_argument("--cuda_device", type=int, default=-1)
    parse.add_argument("--sample_size", type=int, default=-1)
    args = parse.parse_args()

    file_path = cached_path(args.model)
    archive = load_archive(file_path, cuda_device=args.cuda_device)
    predictor = Predictor.from_archive(archive, 'multiqa_predictor')
    all_predictions = {}
    contexts = []
    single_file_path_cached = cached_path(args.multiqa_dataset)
    with gzip.open(single_file_path_cached, 'rb') as myzip:
        for example in myzip:
            context = json.loads(example)
            if 'header' in context:
                continue
            contexts.append(context)

            if args.sample_size != -1 and \
                    sum([len(context['qas']) for context in contexts]) >= args.sample_size:
                break

    # predict
    answers = {}
    for context in Tqdm.tqdm(contexts, total = len(contexts)):
        all_predictions.update(predictor.predict_json(context))

        # saving official answers for this context
        for qa in context['qas']:
            qid = qa['qid'].split('_q_')[1]
            if qid not in answers:
                answers[qid] = []

            if 'answer_candidates' in qa['answers']['open-ended']:
                for ans_cand in qa['answers']['open-ended']['answer_candidates']:
                    if 'extractive' in ans_cand and 'single_answer' in ans_cand['extractive']:
                        answers[qid] += [(ans_cand['extractive']['single_answer']['answer'])]
                        if 'aliases' in ans_cand['extractive']['single_answer']:
                            answers[qid] += ans_cand['extractive']['single_answer']['aliases']
                    elif 'yesno' in ans_cand and 'single_answer' in ans_cand['yesno']:
                        answers[qid] += [(ans_cand['yesno']['single_answer'])]

            elif 'cannot_answer' in qa['answers']['open-ended']:
                answers[qid] += ['cannot_answer']

    # running the official evaluation script:
    metrics = evaluate(answers, all_predictions, True)
    print(json.dumps(metrics))

    # automatic filename generation / or manual
    if args.prediction_filepath == None:
        if not os.path.exists('results/' + args.dataset_name):
            os.makedirs('results/' + args.dataset_name)
        output_filepath = 'results/' + args.dataset_name + '/' + '_'.join(args.model.split('/')[-2:]).split('.')[0] + '__on__' + \
                               args.multiqa_dataset.split('/')[-1].split('.')[0]
    else:
        output_filepath = args.output_filepath

    # formatting the predictions in the specific dataset format in order to run the official eval_script
    factory = MultiQAFactory()
    all_predictions = factory.format_predictions(args.dataset_name, all_predictions)

    # running dataset specific eval script

    # saving predictions
    with open(output_filepath + '_predictions.json', 'w') as f:
        json.dump(all_predictions, f)

    # storing results
    with open(output_filepath + '_eval_results.json', 'w') as f:
        json.dump(metrics, f)







