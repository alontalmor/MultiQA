import json
from datasets.multiqa_dataset import MultiQA_DataSet
from overrides import overrides
from allennlp.common.file_utils import cached_path
from common.uuid import gen_uuid
import tqdm
import os
from google_drive_downloader import GoogleDriveDownloader as gdd


class WikiHop(MultiQA_DataSet):
    """

    """

    def __init__(self):
        self.DATASET_NAME = 'WikiHop'

    @overrides
    def build_header(self, contexts, split, preprocessor):
        header = {
            "dataset_name": self.DATASET_NAME,
            "split": split,
            "dataset_url": "http://qangaroo.cs.ucl.ac.uk/",
            "license": "https://creativecommons.org/licenses/by-sa/3.0/",
            "data_source": "wikipedia",
            "tokenization_source": "multiqa",
            "full_schema": super().compute_schema(contexts),
            "text_type": "abstract",
            "number_of_qas": sum([len(context['qas']) for context in contexts]),
            "number_of_contexts": len(contexts),
            "readme": "",
            "multiqa_version": super().get_multiqa_version()
        }

        return header

    @overrides
    def build_contexts(self, split, preprocessor , sample_size):
        if not os.path.exists('data/quangaroo_v1.1/wikihop/' + split + '.json'):
            gdd.download_file_from_google_drive(file_id='1ytVZ4AhubFDOEL7o7XrIRIyhU8g9wvKA',
                                            dest_path='data/qangroo.zip',
                                            unzip=True)
        with open('data/qangaroo_v1.1/wikihop/' + split + '.json','r') as f:
            data = json.load(f)

        contexts = []
        for example in tqdm.tqdm(data, total=len(data), ncols=80):

            documents = []
            for para in example['supports']:
                documents.append({'text':para})

            answers = {
                'multi-choice': {
                    "choices": [{"extractive": {"single_answer":{"answer": answer}}} for answer in example['candidates']],
                    "correct_answer_index": example['candidates'].index(example['answer'])
                }
            }

            qas = [{"qid": self.DATASET_NAME + '_q_' + example['id'],
                    "metadata":{'annotations':example['annotations']},
                    "question": example['query'],
                    "answers": answers,
                    }]

            contexts.append({"id": self.DATASET_NAME + '_' + example['id'],
                             "context": {"documents": documents},
                             "qas": qas})

        if sample_size != None:
            contexts = contexts[0:sample_size]
        contexts = preprocessor.tokenize_and_detect_answers(contexts)

        # detect answers

        # save dataset

        return contexts
