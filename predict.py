import argparse
import models
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from allennlp.common.file_utils import cached_path
from  datasets.multiqa_factory import MultiQAFactory
import numpy as np
import gzip
import json
from allennlp.common.tqdm import Tqdm

if __name__ == "__main__":
    parse = argparse.ArgumentParser("")
    parse.add_argument("model")
    parse.add_argument("dataset")
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
    single_file_path_cached = cached_path(args.dataset)
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
    for context in Tqdm.tqdm(contexts,total = len(contexts)):
        all_predictions.update(predictor.predict_json(context))

    # automatic filename generation / or manual
    if args.prediction_filepath == None:
        prediction_filepath = 'datasets/' + args.dataset_name + '/_'.join(args.model.split('/')[-2:]).split('.')[0] + '__on__' + \
                               args.dataset.split('/')[-1].split('.')[0] + '.json'
    else:
        prediction_filepath = args.prediction_filepath

    # formatting the predictions in the specific dataset format in order to run the official eval_script
    factory = MultiQAFactory()
    all_predictions = factory.format_predictions(args.dataset_name, all_predictions)

    with open(prediction_filepath, 'w') as f:
        json.dump(all_predictions, f)





