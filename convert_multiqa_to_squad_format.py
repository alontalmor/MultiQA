from typing import Dict, Any
import argparse
import logging
import json
import os
import gzip
import _jsonnet
import tqdm
from allennlp.common.file_utils import cached_path
from allennlp.commands.train import train_model_from_file
from allennlp.common.util import import_submodules
from allennlp.common.util import prepare_environment
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.models.archival import load_archive
from allennlp.training.util import evaluate
from allennlp.common import Params

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def multi_example_to_squad(example):
    squad_example = {}

    offsets = [{} for d in example['context']['documents']]
    squad_context = '[aug] yes no '
    aug_offests = {'yes':6, 'no':10}
    for doc_id, doc in enumerate(example['context']['documents']):
        for part in ['title','text','snippet']:
            if part in doc:
                squad_context += ' [' + part + '] '
                offsets[doc_id][part] = len(squad_context)
                squad_context += doc[part]


    new_qas = []
    for qa in example['qas']:
        new_qa = {'id':qa['qid'],'question':qa['question'],'answers':[],'is_impossible':False}
        if 'cannot_answer' in qa['answers']["open-ended"]:
            new_qa['is_impossible'] = True
        else:
            for answer_cand in qa['answers']["open-ended"]['annotators_answer_candidates']:
                if 'extractive' in answer_cand:
                    for instance in answer_cand['extractive']['single_answer']['instances']:
                        new_qa['answers'].append({'text': instance['text'] ,\
                                                  'answer_start':instance['start_byte'] + offsets[instance['doc_id']][instance['part']]})
                        # sanity check
                        #offest = instance['start_byte'] + offsets[instance['doc_id']][instance['part']]
                        #if instance['text'].lower() != squad_context[offest:offest+len(instance['text'])].lower():
                        #    print(instance['text'] + ' || ' + squad_context[offest:offest+len(instance['text'])])
                        #else:
                        #    print('OK!')
                if 'yesno' in answer_cand:
                    new_qa['answers'].append({'text': answer_cand['yesno']['single_answer'], \
                                              'answer_start': aug_offests[answer_cand['yesno']['single_answer']]})

        new_qas.append(new_qa)

    squad_example['qas'] = new_qas
    squad_example['context'] = squad_context

    return squad_example

def multiqa_to_squad(dataset_paths, dataset_weights=None, sample_size = -1):
    # take one or more multiqa files and convert it to a squad format file.
    # supporting multi-dataset training:
    datasets = []
    for ind, single_file_path in enumerate(dataset_paths):
        single_file_path_cached = cached_path(single_file_path)
        zip_handle = gzip.open(single_file_path_cached, 'rb')
        datasets.append({'single_file_path': single_file_path, \
                         'file_handle': zip_handle, \
                         'num_of_questions': 0, 'inst_remainder': [], \
                         'dataset_weight': 1 if dataset_weights is None else dataset_weights[ind]})
        datasets[ind]['header'] = json.loads(datasets[ind]['file_handle'].readline())['header']

    # We will have only one topic here..
    squad_data = {'data':[{'title':'','paragraphs':[]}]}
    is_done = [False for _ in datasets]
    while not all(is_done):
        for ind, dataset in enumerate(datasets):
            if is_done[ind]:
                continue

            for example in dataset['file_handle']:
                squad_example = multi_example_to_squad(json.loads(example))

                squad_data['data'][0]['paragraphs'].append(squad_example)
                dataset['num_of_questions'] += len(squad_example['qas'])

                # supporting sampling of first #dataset['num_to_sample'] examples
                if dataset['num_of_questions'] >= dataset['dataset_weight']:
                    break

            else:
                # No more lines to be read from file
                is_done[ind] = True

            # per dataset sampling
            if sample_size > -1 and dataset['num_of_questions'] >= sample_size:
                is_done[ind] = True

    for dataset in datasets:
        logger.info("Total number of processed questions for %s is %d", dataset['header']['dataset_name'], dataset['num_of_questions'])
        dataset['file_handle'].close()

    return squad_data

def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("--datasets", type=str, help="", default=None)
    parse.add_argument("--output_file", type=str, help="directory to which results JSONs of eval will written", default='results/eval/')
    args = parse.parse_args()

    squad_data = multiqa_to_squad(args.datasets.split(','))

    with open(args.output_file,'w') as f:
        json.dump(squad_data,f)


if __name__ == '__main__':
    main()





