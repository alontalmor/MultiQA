import json
from datasets.multiqa_dataset import MultiQA_DataSet
from overrides import overrides
from allennlp.common.file_utils import cached_path
from common.uuid import gen_uuid
import tqdm
import zipfile


class ComplexWebQuestions(MultiQA_DataSet):
    """

    """

    def __init__(self):
        self.DATASET_NAME = 'ComplexWebQuestions'

    @overrides
    def build_header(self, preprocessor, contexts, split, dataset_version, dataset_flavor, dataset_specific_props):
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
    def build_contexts(self, preprocessor, split, sample_size, dataset_version, dataset_flavor, dataset_specific_props, input_file):
        single_file_path = cached_path("https://s3.amazonaws.com/multiqa/datasets/ComplexWebQuestions_RC_" + split + ".jsonl.zip")

        examples = []
        with zipfile.ZipFile(single_file_path, 'r') as myzip:
            with myzip.open(myzip.namelist()[0]) as myfile:
                header = json.loads(myfile.readline())['header']
                for line, example in enumerate(myfile):
                    # header
                    examples.append(json.loads(example))

        contexts = []
        qas_count = 0
        for example in tqdm.tqdm(examples, total=len(examples), ncols=80):
            qas = []
            new_qa = {'qid':self.DATASET_NAME + '_q_' + str(example['id']),
                        'question':example['qas'][0]['question']}
            new_qa['answers'] = new_qa['answers'] = {"open-ended": {
                'answer_candidates': [
                    {'extractive':
                         {"single_answer": {'answer': example['qas'][0]['answers'][0]['answer'], \
                                            'aliases': [alias['text'] for alias in example['qas'][0]['answers'][0]['aliases']]}}
                     }
                ]}}
            qas.append(new_qa)

            documents = []
            for search_res in example['documents']:
                if search_res['paragraphs'] is not None:
                    documents.append({'title': search_res['title'],'text': search_res['paragraphs'][0]})

            contexts.append({"id": self.DATASET_NAME + '_'  + str(example['id']),
                             "context": {"documents": documents},
                             "qas": qas})

            qas_count += len(qas)
            if sample_size != None and qas_count > sample_size:
                break


        contexts = preprocessor.tokenize_and_detect_answers(contexts)

        return contexts


