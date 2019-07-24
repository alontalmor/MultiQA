import json
from datasets.multiqa_dataset import MultiQA_DataSet
from overrides import overrides
from allennlp.common.file_utils import cached_path
from common.uuid import gen_uuid
import tqdm
import zipfile


class SearchQA(MultiQA_DataSet):
    """

    """

    def __init__(self):
        self.DATASET_NAME = 'SearchQA'

    @overrides
    def build_header(self, contexts, split, preprocessor, dataset_version, dataset_flavor):
        header = {
            "dataset_name": self.DATASET_NAME,
            "split": split,
            "dataset_url": "https://github.com/nyu-dl/dl4ir-searchqA",
            "license": "",
            "data_source": "Web",
            "context_answer_detection_source": 'MultiQA',
            "tokenization_source": "MultiQA",
            "full_schema": super().compute_schema(contexts),
            "text_type": "abstract",
            "number_of_qas": sum([len(context['qas']) for context in contexts]),
            "number_of_contexts": len(contexts),
            "readme": "",
            "multiqa_version": super().get_multiqa_version()
        }

        return header

    @overrides
    def build_contexts(self, split, preprocessor, sample_size, dataset_version, dataset_flavor, input_file, build_properties):
        if split == 'train':
            single_file_path = cached_path("https://s3.amazonaws.com/multiqa/raw_datasets/SearchQA/train.zip")
        elif split == 'dev':
            single_file_path = cached_path("https://s3.amazonaws.com/multiqa/raw_datasets/SearchQA/val.zip")

        examples = []
        with zipfile.ZipFile(single_file_path, 'r') as myzip:
            for name in myzip.namelist():
                with myzip.open(name) as myfile:
                    examples.append(json.load(myfile))

        contexts = []
        for example in tqdm.tqdm(examples, total=len(examples), ncols=80):
            qas = []
            new_qa = {'qid':self.DATASET_NAME + '_q_' + str(example['id']),
                        'question':example['question']}
            new_qa['answers'] = new_qa['answers'] = {"open-ended":{
                        'answer_candidates': [
                            {'extractive':
                                {"single_answer": {'answer': example['answer']}}
                             }
                        ]}}
            qas.append(new_qa)

            documents = []
            for search_res in example['search_results']:
                if search_res['snippet'] is not None:
                    documents.append({'title': search_res['title'],'text': search_res['snippet'],'url':search_res['url']})

            contexts.append({"id": self.DATASET_NAME + '_'  + str(example['id']),
                             "context": {"documents": documents},
                             "qas": qas})


        if sample_size != None:
            contexts = contexts[0:sample_size]
        contexts = preprocessor.tokenize_and_detect_answers(contexts)

        return contexts