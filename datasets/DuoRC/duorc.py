import json
from datasets.multiqa_dataset import MultiQA_DataSet
from overrides import overrides
from allennlp.common.file_utils import cached_path
import tqdm
import gzip
import logging
logger = logging.getLogger(__name__)

class DuoRC(MultiQA_DataSet):
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
        self.DATASET_NAME = 'DuoRC'
        self._output_file_count = 0
        self.done_processing = True

    @overrides
    def build_header(self, contexts):
        header = {
            "dataset_name": self.DATASET_NAME,
            "split": self._split,
            "dataset_url": "https://datasets.maluuba.com/NewsQA",
            "license": "https://datasets.maluuba.com/terms-and-conditions",
            "data_source": "NewsWire",
            "context_answer_detection_source": self.DATASET_NAME,
            "tokenization_source": "MultiQA",
            "full_schema": super().compute_schema(contexts),
            "text_type": "abstract",
            "number_of_qas": sum([len(context['qas']) for context in contexts]),
            "number_of_contexts": len(contexts),
            "number_of_qas_with_gold_answers": sum([len(context['qas']) for context in contexts]),
            "file_num": self._output_file_count,
            "next_file_exists": not self._done_processing,
            "readme": "",
            "multiqa_version": super().get_multiqa_version()
        }

        return header

    @overrides
    def build_contexts(self):

        single_file_path = cached_path('https://s3.amazonaws.com/multiqa/raw_datasets/DuoRC/' + \
            self._dataset_flavor + 'RC_' + self._split + '.json.gz')

        with gzip.open(single_file_path, 'rb') as myfile:
            data = json.load(myfile)

        contexts = []
        total_qas_count = 0
        for plot in tqdm.tqdm(data, total=len(data), ncols=80):

            # we have only one question per context here:
            qas = []
            for qa_ind, qa in enumerate(plot['qa']):
                new_qa = {'qid': self.DATASET_NAME + '_q_' + qa['id'] ,
                          'question': qa['question']}

                if qa['no_answer']:
                    new_qa['answers'] = {"open-ended": {'cannot_answer': 'yes'}}
                else:
                    new_qa['answers'] = {"open-ended":{
                        'annotators_answer_candidates': [
                            {
                                # TODO some of the aliases are actually abstractive...
                                'single_answer':{
                                    "extractive": {"answer": qa['answers'][0],
                                                   "aliases": qa['answers'][1:]}
                                }
                            }]
                         }}
                qas.append(new_qa)

            contexts.append({"id": self.DATASET_NAME + '_'  + plot['id'],
                             "context": {"documents": [{"text": plot['plot'], "title": plot['title']}]},
                             "qas": qas})

            total_qas_count += len(qas)
            if (self._sample_size != None and total_qas_count > self._sample_size):
                break

        logger.info('producing final context file')
        self._done_processing = True
        self._output_file_count = 1
        yield self._preprocessor.tokenize_and_detect_answers(contexts)