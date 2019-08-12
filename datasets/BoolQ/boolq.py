

import json
from datasets.multiqa_dataset import MultiQA_DataSet
from overrides import overrides
from allennlp.common.file_utils import cached_path
from common.uuid import gen_uuid
import tqdm


class BoolQ(MultiQA_DataSet):
    """

    """

    def __init__(self):
        self.DATASET_NAME = 'BoolQ'

    @overrides
    def build_header(self, preprocessor, contexts, split, dataset_version, dataset_flavor, dataset_specific_props):

        header = {
            "dataset_name": self.DATASET_NAME,
            "version": dataset_version,
            "flavor": dataset_flavor,
            "split": split,
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
            "readme": "",
            "multiqa_version": super().get_multiqa_version()
        }

        return header

    @overrides
    def build_contexts(self, preprocessor, split, sample_size, dataset_version, dataset_flavor, dataset_specific_props, input_file):
        single_file_path = "data/boolq/" + split + ".jsonl"

        data = []
        with open(single_file_path, 'r') as myfile:
            for example in myfile:
                data.append(json.loads(example))

        contexts = []
        qas_count = 0
        for example in tqdm.tqdm(data, total=len(data), ncols=80):
            q_uuid = gen_uuid()
            if example['answer'] == True:
                answers = {'open-ended': {'answer_candidates': [{'yesno':{'single_answer':'yes'}}]}}
            elif example['answer'] == False:
                answers = {'open-ended': {'answer_candidates': [{'yesno':{'single_answer':'no'}}]}}

            qas = [{"qid": self.DATASET_NAME + '_q_' + q_uuid,
                    "question": example['question'],
                    "answers": answers,
                    }]

            qas_count += len(qas)
            if sample_size != None and qas_count > sample_size:
                break

            contexts.append({"id": self.DATASET_NAME + '_' + q_uuid,
                             "context": {"documents": [{"text": example['passage'], \
                                                        "title": example['title']}]},
                             "qas": qas})


        contexts = preprocessor.tokenize_and_detect_answers(contexts)

        return contexts