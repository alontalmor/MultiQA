import json
from datasets.multiqa_dataset import MultiQA_DataSet
from overrides import overrides
from allennlp.common.file_utils import cached_path
from common.uuid import gen_uuid
import tqdm


class SQuAD(MultiQA_DataSet):
    """

    """

    def __init__(self):
        self.DATASET_NAME = 'SQuAD'

    @overrides
    def build_header(self, contexts, split, preprocessor):
        header = {}

        return header

    @overrides
    def build_contexts(self, split, preprocessor):
        #single_file_path = cached_path("https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/" + \
        #                               split + "-v1.1.json")
        single_file_path = "/Users/alontalmor/Documents/dev/datasets/Squad1.1/" + split + "-v1.1.json"

        with open(single_file_path, 'r') as myfile:
            original_dataset = json.load(myfile)

        contexts = []
        data = original_dataset['data']


        for topic in tqdm.tqdm(data, total=len(data), ncols=80):
            for paragraph in topic['paragraphs']:
                qas = []
                for qa in paragraph['qas']:
                    # TODO drop duplicates answer instances, divide to aliases
                    answers = {'extractive':{'single_answer': {"answer": qa['answers'][0]['text'],
                        "aliases":list(set([instance['text'] for instance in qa['answers']])),
                        "instances": [{'doc_id':0,
                                       'doc_part':'text',
                                       'start_byte':instance['answer_start'],
                                       'text':instance['text']} for instance in qa['answers']]}}}

                    qas.append({'qid':qa['id'],
                                'question':qa['question'],
                                'answers':answers})

                contexts.append({"id": self.DATASET_NAME + '_'  + gen_uuid(),
                                 "context": {"documents": [{"text": paragraph['context'],
                                                            "title":topic['title']}]},
                                 "qas": qas})


        # tokenize
        # TODO this is only for debugging :
        contexts = contexts[0:2]
        contexts = preprocessor.tokenize_and_detect_answers(contexts)

        # detect answers

        # save dataset

        return contexts