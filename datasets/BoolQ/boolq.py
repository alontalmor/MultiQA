import json
from datasets.multiqa_dataset import MultiQA_DataSet
from overrides import overrides
from allennlp.common.file_utils import cached_path
from common.uuid import gen_uuid
import tqdm
import logging
import gzip
logger = logging.getLogger(__name__)

class BoolQ(MultiQA_DataSet):
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
        self.DATASET_NAME = 'BoolQ'
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
            "file_num": self._output_file_count,
            "next_file_exists": not self._done_processing,
            "readme": "",
            "multiqa_version": super().get_multiqa_version()
        }

        return header

    @overrides
    def build_contexts(self):
        # This dataset is provided with google storage, https://storage.cloud.google.com/boolq/dev.jsonl which is a pain to work with
        # python, so i just added it to the datasets folder directly ...
        single_file_path = "datasets/BoolQ/" + self._split + ".jsonl.gz"

        data = []
        total_qas_count = 0
        with gzip.open(single_file_path, 'rb') as myfile:
            for example in myfile:
                data.append(json.loads(example))

        contexts = []
        qas_count = 0
        for example in tqdm.tqdm(data, total=len(data), ncols=80):
            q_uuid = gen_uuid()
            if example['answer'] == True:
                answers = {'open-ended': {'annotators_answer_candidates': [{'single_answer':{'yesno':'yes'}}]}}
            elif example['answer'] == False:
                answers = {'open-ended': {'annotators_answer_candidates': [{'single_answer':{'yesno':'no'}}]}}

            qas = [{"qid": self.DATASET_NAME + '_q_' + q_uuid,
                    "question": example['question'],
                    "answers": answers,
                    }]

            contexts.append({"id": self.DATASET_NAME + '_' + q_uuid,
                             "context": {"documents": [{"text": example['passage'], \
                                                        "title": example['title']}]},
                             "qas": qas})

            total_qas_count += len(qas)
            if (self._sample_size != None and total_qas_count > self._sample_size):
                break

        logger.info('producing final context file')
        self._done_processing = True
        self._output_file_count = 1
        yield self._preprocessor.tokenize_and_detect_answers(contexts)