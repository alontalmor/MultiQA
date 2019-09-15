import json
from datasets.multiqa_dataset import MultiQA_DataSet
from overrides import overrides
from allennlp.common.file_utils import cached_path
from common.uuid import gen_uuid
import tqdm
import zipfile
import logging
logger = logging.getLogger(__name__)


class DROP(MultiQA_DataSet):
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
        self.DATASET_NAME = 'DROP'
        self._output_file_count = 0
        self.done_processing = True

    @overrides
    def build_header(self, contexts):
        header = {
            "dataset_name": self.DATASET_NAME,
            "version": self._dataset_version,
            "flavor": self._dataset_flavor,
            "split": self._split,
            "dataset_url": "https://allennlp.org/drop",
            "license": "https://creativecommons.org/licenses/by-sa/4.0/legalcode",
            "data_source": "Wikipedia",
            "context_answer_detection_source": self.DATASET_NAME,
            "tokenization_source": "MultiQA",
            "full_schema": super().compute_schema(contexts),
            "text_type": "abstract",
            "number_of_qas": sum([len(context['qas']) for context in contexts]),
            "number_of_qas_with_gold_answers": sum([len(context['qas']) for context in contexts]),
            "number_of_contexts": len(contexts),
            "file_num": self._output_file_count,
            "next_file_exists": not self._done_processing,
            "readme": "",
            "multiqa_version": super().get_multiqa_version()
        }

        return header

    @overrides
    def build_contexts(self):
        single_file_path = cached_path("https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip")
        with zipfile.ZipFile(single_file_path, 'r') as archive:
            data = json.loads(archive.read('drop_dataset/drop_dataset_' + self._split + '.json'))

        contexts = []
        total_qas_count = 0
        for id, context in tqdm.tqdm(data.items(), total=len(data), ncols=80):

            qas = []
            for qa in context['qa_pairs']:

                new_qa = {'qid': self.DATASET_NAME + '_q_' + qa['query_id'],
                          'question': qa['question']}
                answer_candidates = []

                # building a list of original answer candidates to iterate on, main answer will be first.
                org_answer_candidates = [qa['answer']]
                if "validated_answers" in qa:
                    org_answer_candidates += qa["validated_answers"]

                for answer_candidate in org_answer_candidates:
                    new_ans_cand = {}

                    if len(answer_candidate['spans']) > 0:
                        new_ans_cand['list'] = [{"extractive":{"answer": span}} for span in answer_candidate['spans']]

                    if answer_candidate['number'] != '':
                        new_ans_cand['single_answer'] = {'number':float(answer_candidate['number'])}

                    if 'day' in answer_candidate and (answer_candidate['date']['day'] != '' or \
                            answer_candidate['date']['month'] != '' or \
                            answer_candidate['date']['year'] != ''):
                        new_ans_cand['single_answer'] = {'date':answer_candidate['date']}

                    if new_ans_cand != {}:
                        answer_candidates.append(new_ans_cand)

                new_qa['answers'] = {'open-ended': {'annotators_answer_candidates': answer_candidates}}
                qas.append(new_qa)

            contexts.append({"id": self.DATASET_NAME + '_' + id,
                             "context": {"documents": [{"text": context['passage'],
                                                        "url": context['wiki_url']}]},
                             "qas": qas})

            total_qas_count += len(qas)
            if (self._sample_size != None and total_qas_count > self._sample_size):
                break

        logger.info('producing final context file')
        self._done_processing = True
        self._output_file_count = 1
        yield self._preprocessor.tokenize_and_detect_answers(contexts)