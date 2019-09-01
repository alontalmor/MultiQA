import json
from datasets.multiqa_dataset import MultiQA_DataSet
from overrides import overrides
from allennlp.common.file_utils import cached_path
from common.uuid import gen_uuid
import tqdm
import zipfile
import logging
logger = logging.getLogger(__name__)

class ComplexWebQuestions(MultiQA_DataSet):
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
        self.DATASET_NAME = 'ComplexWebQuestions'
        self._output_file_count = 0
        self.done_processing = True

    @overrides
    def build_header(self, contexts):
        header = {
            "dataset_name": self.DATASET_NAME,
            "split": self._split,
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
    def build_contexts(self):
        single_file_path = cached_path("https://s3.amazonaws.com/multiqa/datasets/ComplexWebQuestions_RC_" + self._split + ".jsonl.zip")

        examples = []
        with zipfile.ZipFile(single_file_path, 'r') as myzip:
            with myzip.open(myzip.namelist()[0]) as myfile:
                header = json.loads(myfile.readline())['header']
                for line, example in enumerate(myfile):
                    # header
                    examples.append(json.loads(example))

        contexts = []
        total_qas_count = 0
        for example in tqdm.tqdm(examples, total=len(examples), ncols=80):
            qas = []
            new_qa = {'qid':self.DATASET_NAME + '_q_' + str(example['id']),
                        'question':example['qas'][0]['question']}
            new_qa['answers'] = new_qa['answers'] = {"open-ended": {
                'annotators_answer_candidates': [
                    {'single_answer':
                         {'extractive': {'answer': example['qas'][0]['answers'][0]['answer'], \
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

            total_qas_count += len(qas)
            if (self._sample_size != None and total_qas_count > self._sample_size):
                break

        logger.info('producing final context file')
        self._done_processing = True
        self._output_file_count = 1
        yield self._preprocessor.tokenize_and_detect_answers(contexts)


