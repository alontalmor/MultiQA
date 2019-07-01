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
        header = {
            "dataset_name": self.DATASET_NAME,
            "split": split,
            "dataset_url": "https://ai.google.com/research/NaturalQuestions/",
            "license": "https://ai.google.com/research/NaturalQuestions/termsAndConditions",
            "data_source": "Wikipedia",
            "context_answer_detection_source": self.DATASET_NAME,
            "tokenization_source": self.DATASET_NAME,
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
        single_file_path = "/Users/alontalmor/Documents/dev/datasets/NaturalQuestions/natural_questions/v1.0/sample/nq-" \
                           + split + "-sample.jsonl.gz"

        with gzip.open(single_file_path, 'r') as f:
            data = []
            for example in f:
                data.append(json.loads(example))



        contexts = []
        for example in tqdm.tqdm(data, total=len(data), ncols=80):


            answer_candidates = []
            for answer_candidate in example['annotations']:
                new_ans_cand = {}
                # We do include the long answer as metadata.
                if len(answer_candidate['long_answer']) > 0 and answer_candidate['long_answer']['start_byte'] != -1:
                    new_ans_cand['metadata'] = {'long_answer':answer_candidate['long_answer']}

                if len(answer_candidate['short_answers']) > 0 and answer_candidate['short_answers'][0]['start_byte'] != -1:
                    new_ans_cand['extractive'] =  {"single_answer": {"answer": example['document_html'][
                                     answer_candidate['short_answers'][0]['start_byte']: \
                                     answer_candidate['short_answers'][0]['end_byte']],
                        "instances": [{'doc_id': 0,
                             'doc_part': 'text',
                             'start_byte': instance['start_byte'],
                             'token_inds': [instance['start_token'], instance['end_token']],
                             'text': example['document_html'][instance['start_byte']: instance['end_byte']]} \
                                      for instance in answer_candidate['short_answers']]}}

                if answer_candidate['yes_no_answer'] != 'NONE':
                    new_ans_cand['yesno'] = {"single_answer": 'yes' if answer_candidate['yes_no_answer'] == 'YES' else 'no'}

                if new_ans_cand != {}:
                    answer_candidates.append(new_ans_cand)

            # We define an evaluation measure based on the
            # 5 way annotations as follows. If at least 2 out of
            # 5 annotators have given a non-null long answer on
            # the example, then the system is required to output
            # a non-null answer that is seen at least once in the 5
            # annotations; conversely if fewer than 2 annotators
            # give a non-null long answer, the system is required
            # to return NULL as its output.
            # from https://ai.google/research/pubs/pub47761
            # we will ineicate cannot answer as well as add the candidates (for completeness)
            open_ended = {'answer_candidates': answer_candidates}
            if len(answer_candidates) < 2:
                open_ended['cannot_answer'] = 'yes'

            # TODO question_tokens should contain start bytles
            contexts.append({"id": self.DATASET_NAME + '_' + str(example['example_id']),
                             "context": {"documents": [{"text": example['document_html'],
                                    "title": example['document_title'],
                                    "url": example['document_url'],
                                    "metadata":{"tokens": {"text":{"is_html_token": \
                                                    [1 if t["html_token"] else 0 for t in example['document_tokens']]}}},
                                    "tokens": {"text": [(t["token"],t["start_byte"]) for t in example['document_tokens']]}}]},
                             "qas": [{'qid':self.DATASET_NAME + '_q_' + str(example['example_id']),
                                      'question':example['question_text'],
                                      'question_tokens':example['question_tokens'],
                                      'answers':{'open-ended': open_ended}}]})

        if sample_size != None:
            contexts = contexts[0:sample_size]
        contexts = preprocessor.tokenize_and_detect_answers(contexts)

        # detect answers

        # save dataset

        return contexts