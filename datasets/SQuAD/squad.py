import json
from datasets.multiqa_dataset import MultiQA_DataSet
from overrides import overrides
from allennlp.common.file_utils import cached_path
from common.uuid import gen_uuid
import tqdm
import logging
logger = logging.getLogger(__name__)


class SQuAD(MultiQA_DataSet):
    """

    """
    def __init__(self, preprocessor, split, dataset_version, dataset_flavor, dataset_specific_props, \
                 sample_size, max_contexts_in_file, custom_input_file):
        self._preprocessor = preprocessor
        self._split = split
        self._dataset_version = dataset_version
        self._dataset_flavor = dataset_flavor
        self._dataset_specific_props = dataset_specific_props
        self._sample_size = sample_size
        self._custom_input_file = custom_input_file
        self._max_contexts_in_file = max_contexts_in_file
        self.DATASET_NAME = 'SQuAD'
        self._output_file_count = 0
        self.done_processing = True

    @overrides
    def build_header(self, contexts):

        header = {
            "dataset_name": self.DATASET_NAME,
            "version": self._dataset_version,
            "flavor": self._dataset_flavor,
            "split": self._split,
            "dataset_url": "https://rajpurkar.github.io/SQuAD-explorer/",
            "license": "http://creativecommons.org/licenses/by-sa/4.0/legalcode",
            "data_source": "Wikipedia",
            "context_answer_detection_source": self.DATASET_NAME,
            "tokenization_source": "MultiQA",
            "full_schema": super().compute_schema(contexts),
            "text_type": "abstract",
            "number_of_qas": sum([len(context['qas']) for context in contexts]),
            "number_of_qas_with_gold_answers": sum([len(context['qas']) for context in contexts]),
            "number_of_contexts": len(contexts),
            "file_num":self._output_file_count,
            "next_file_exists": not self._done_processing,
            "readme": "",
            "multiqa_version": super().get_multiqa_version()
        }

        return header

    @overrides
    def build_contexts(self):
        single_file_path = cached_path("https://rajpurkar.github.io/SQuAD-explorer/dataset/" + \
                                       self._split + "-v" + self._dataset_version.replace('-','.') +".json")

        with open(single_file_path, 'r') as myfile:
            original_dataset = json.load(myfile)

        contexts = []
        total_qas_count = 0
        data = original_dataset['data']


        for topic in tqdm.tqdm(data, total=len(data), ncols=80):
            for paragraph in topic['paragraphs']:
                qas = []
                for qa in paragraph['qas']:
                    new_qa = {'qid':self.DATASET_NAME + '_q_' + qa['id'],
                                'question':qa['question']}
                    answer_candidates = []
                    if 'is_impossible' in qa and qa['is_impossible']:
                        new_qa['answers'] = {"open-ended": {'cannot_answer': 'yes'}}
                        new_qa['metadata'] = {'plausible_answers': qa['plausible_answers']}
                    else:
                        for answer_candidate in qa['answers']:
                            answer_candidates.append({"single_answer":{'extractive':{"answer": answer_candidate['text'],
                                "instances": [{'doc_id':0,
                                           'part':'text',
                                           'start_byte':answer_candidate['answer_start'],
                                           'text':answer_candidate['text']}]}}})
                        new_qa['answers'] = {"open-ended": {'annotators_answer_candidates': answer_candidates}}
                    qas.append(new_qa)

                contexts.append({"id": self.DATASET_NAME + '_'  + gen_uuid(),
                                 "context": {"documents": [{"text": paragraph['context'],
                                                            "title":topic['title']}]},
                                 "qas": qas})

                total_qas_count += len(qas)
                if (self._sample_size != None and total_qas_count > self._sample_size):
                    break

        logger.info('producing final context file')
        self._done_processing = True
        self._output_file_count = 1
        yield self._preprocessor.tokenize_and_detect_answers(contexts)