import json
from datasets.multiqa_dataset import MultiQA_DataSet
from overrides import overrides
from allennlp.common.file_utils import cached_path
from common.uuid import gen_uuid
import tqdm


class NewsQA(MultiQA_DataSet):
    """

    """

    def __init__(self):
        self.DATASET_NAME = 'NewsQA'

    @overrides
    def build_header(self, contexts, split, preprocessor ):
        header = {
            "dataset_name": self.DATASET_NAME,
            "split": split,
            "dataset_url": "https://rajpurkar.github.io/SQuAD-explorer/",
            "license": "http://creativecommons.org/licenses/by-sa/4.0/legalcode",
            "data_source": "Wikipedia",
            "context_answer_detection_source": self.DATASET_NAME,
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
    def build_contexts(self, split, preprocessor, sample_size):
        single_file_path = cached_path("https://s3.amazonaws.com/multiqa/raw_datasets/combined-newsqa-data-v1.json")

        with open(single_file_path, 'r') as myfile:
            original_dataset = json.load(myfile)

        contexts = []
        data = original_dataset['data']


        for story in tqdm.tqdm(data, total=len(data), ncols=80):

            # we have only one question per context here:
            qas = []
            for qa_ind, qa in enumerate(story['questions']):
                new_qa = {'qid': self.DATASET_NAME + '_q_' + story['storyId'] + '_' + str(qa_ind) ,
                          'question': qa['q']}

                if 'badQuestion' in qa['consensus']:
                    continue
                elif 'noAnswer' in qa['consensus']:
                    new_qa['answers'] = {"open-ended": {'cannot_answer': 'yes'}}
                else:
                    new_qa['answers'] = {"open-ended":{
                        'answer_candidates': [
                            {'extractive':
                                {"single_answer":
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


        if sample_size != None:
            contexts = contexts[0:sample_size]
        contexts = preprocessor.tokenize_and_detect_answers(contexts)

        # detect answers

        # save dataset

        return contexts