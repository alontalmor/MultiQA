import json
from datasets.multiqa_dataset import MultiQA_DataSet
from overrides import overrides
from allennlp.common.file_utils import cached_path
import tqdm
import logging
logger = logging.getLogger(__name__)

class NewsQA(MultiQA_DataSet):
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
        self.DATASET_NAME = 'NewsQA'
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
            "file_num": self._output_file_count,
            "next_file_exists": not self._done_processing,
            "readme": "",
            "multiqa_version": super().get_multiqa_version()
        }

        return header

    @overrides
    def build_contexts(self):
        single_file_path = cached_path("https://s3.amazonaws.com/multiqa/raw_datasets/combined-newsqa-data-v1.json")

        with open(single_file_path, 'r') as myfile:
            original_dataset = json.load(myfile)

        contexts = []
        total_qas_count = 0
        data = original_dataset['data']


        for story in tqdm.tqdm(data, total=len(data), ncols=80):
            if story['type'] != self._split:
                continue

            # we have only one question per context here:
            qas = []
            for qa_ind, qa in enumerate(story['questions']):
                new_qa = {'qid': self.DATASET_NAME + '_q_' + story['storyId'] + '_' + str(qa_ind) ,
                          'question': qa['q']}

                if 'badQuestion' in qa['consensus']:
                    continue
                elif 'noAnswer' in qa['consensus']:
                    continue
                    #new_qa['answers'] = {"open-ended": {'cannot_answer': 'yes'}}
                else:
                    new_qa['answers'] = {"open-ended":{
                        'annotators_answer_candidates': [
                            {'single_answer':
                                {'extractive':
                                    {"answer": story['text'][qa['consensus']['s']:qa['consensus']['e']].strip(),
                                        "instances": [{
                                            'doc_id':0,
                                            'part':'text',
                                            'start_byte':qa['consensus']['s'],
                                            'text':story['text'][qa['consensus']['s']:qa['consensus']['e']].strip()}
                                    ]}
                                }
                            }]
                         }}
                qas.append(new_qa)

            contexts.append({"id": self.DATASET_NAME + '_'  + story['storyId'],
                             "context": {"documents": [{"text": story['text']}]},
                             "qas": qas})

            total_qas_count += len(qas)
            if (self._sample_size != None and total_qas_count > self._sample_size):
                break

        logger.info('producing final context file')
        self._done_processing = True
        self._output_file_count = 1
        yield self._preprocessor.tokenize_and_detect_answers(contexts)