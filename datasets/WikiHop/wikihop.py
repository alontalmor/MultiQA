import json
from datasets.multiqa_dataset import MultiQA_DataSet
from overrides import overrides
from allennlp.common.file_utils import cached_path
from common.uuid import gen_uuid
import tqdm
import os
from google_drive_downloader import GoogleDriveDownloader as gdd
import logging
logger = logging.getLogger(__name__)


class WikiHop(MultiQA_DataSet):
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
        self.DATASET_NAME = 'WikiHop'
        self._output_file_count = 0
        self.done_processing = True

    @overrides
    def build_header(self, contexts):
        header = {
            "dataset_name": self.DATASET_NAME,
            "version": self._dataset_version,
            "flavor": self._dataset_flavor,
            "split": self._split,
            "dataset_url": "http://qangaroo.cs.ucl.ac.uk/",
            "license": "https://creativecommons.org/licenses/by-sa/3.0/",
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
        if not os.path.exists('data/quangaroo_v1.1/wikihop/' + self._split + '.json'):
            gdd.download_file_from_google_drive(file_id='1ytVZ4AhubFDOEL7o7XrIRIyhU8g9wvKA',
                                            dest_path='data/qangroo.zip',
                                            unzip=True)
        with open('data/qangaroo_v1.1/wikihop/' + self._split + '.json','r') as f:
            data = json.load(f)

        total_qas_count = 0
        contexts = []
        for example in tqdm.tqdm(data, total=len(data), ncols=80):

            documents = []
            for para in example['supports']:
                documents.append({'text':para})

            answers = {"open-ended": {
                'annotators_answer_candidates': [
                    {'single_answer':
                         {'extractive': {'answer': example['answer']}}
                     }
                ]}}

            metadata = {'candidates':example['candidates']}
            if 'annotations' in example:
                metadata['annotations'] = example['annotations']

            qas = [{"qid": self.DATASET_NAME + '_q_' + example['id'],
                    "metadata":metadata,
                    "question": example['query'],
                    "answers": answers,
                    }]

            contexts.append({"id": self.DATASET_NAME + '_' + example['id'],
                             "context": {"documents": documents},
                             "qas": qas})

            total_qas_count += len(qas)
            if (self._sample_size != None and total_qas_count > self._sample_size):
                break

        logger.info('producing final context file')
        self._done_processing = True
        self._output_file_count = 1
        yield self._preprocessor.tokenize_and_detect_answers(contexts)
