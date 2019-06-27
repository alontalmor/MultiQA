import json
import gzip
from datasets.multiqa_dataset import MultiQA_DataSet
from overrides import overrides
from allennlp.common.file_utils import cached_path
from common.uuid import gen_uuid
import tqdm


class NaturalQuestions(MultiQA_DataSet):
    """

    """

    def __init__(self):
        self.DATASET_NAME = 'NaturalQuestions'

    @overrides
    def build_header(self, contexts, split, preprocessor):
        header = {}

        return header

    @overrides
    def build_contexts(self, split, preprocessor):
        #if split == 'train':
        #    single_file_path = cached_path("http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json")
        #elif split == 'dev':
        #    single_file_path = cached_path("http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json")
        single_file_path = "/Users/alontalmor/Documents/dev/datasets/NaturalQuestions/natural_questions/v1.0/sample/nq-" \
                           + split + "-sample.jsonl.gz"

        with gzip.open(single_file_path, 'r') as f:
            data = []
            for example in f:
                data.append(json.loads(example))



        contexts = []
        for example in tqdm.tqdm(data, total=len(data), ncols=80):
            # we need to check how does the evaluation treat the five answers ...
            if len(example['annotations']) > 1:
                # If >= 2 of the annotators marked a non-null set of short answers, or a yes/no
                #   answer, then the short answers prediction must match any one of the non-null
                #   sets of short answers *or* the yes/no prediction must match one of the
                #   non-null yes/no answer labels.
                # from https://github.com/google-research-datasets/natural-questions/blob/master/nq_eval.py
            answers = {'extractive': {'single_answer': {"answer": "",
                        "aliases": list(set([instance['text'] for instance in qa['answers']])),
                        "instances": [{'doc_id': 0,
                                       'doc_part': 'text',
                                       'start_byte': instance['answer_start'],
                                       'text': instance['text']} for instance in qa['answers']]}}}

            contexts.append({"id": self.DATASET_NAME + '_' + example['example_id'],
                             "context": {"documents": [{"text": example['document_html'],
                                    "title": example['document_title'],
                                    "url": example['document_url'],
                                    "tokens": {"text": [(t["token"],t["start_byte"]) for t in example['document_tokens']]}}]},
                             "qas": [{'qid':example['example_id'],
                                      'question':example['question_text'],
                                      'question_tokens':example['question_tokens']
                                      'answers':}]})



        # tokenize
        # TODO this is only for debugging :
        #contexts = contexts[0:2]
        contexts = preprocessor.tokenize_and_detect_answers(contexts)

        # detect answers

        # save dataset

        return contexts