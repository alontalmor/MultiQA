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
        self.output_file_count = 0
        self.DATASET_NAME = 'NaturalQuestions'

    @overrides
    def build_header(self, contexts):
        header = {
            "dataset_name": self.DATASET_NAME,
            "split": self._split,
            "dataset_url": "https://ai.google.com/research/NaturalQuestions/",
            "license": "https://ai.google.com/research/NaturalQuestions/termsAndConditions",
            "data_source": "Wikipedia",
            "context_answer_detection_source": self.DATASET_NAME,
            "tokenization_source": self.DATASET_NAME,
            "full_schema": super().compute_schema(contexts),
            "text_type": "full_html",
            "number_of_qas": sum([len(context['qas']) for context in contexts]),
            "number_of_contexts": len(contexts),
            "file_num": self.output_file_count,
            "next_file_exists": not self.done_processing,
            "readme": "",
            "multiqa_version": super().get_multiqa_version()
        }

        return header

    @overrides
    def build_contexts(self):
        single_file_path = "/Users/alontalmor/Documents/dev/datasets/NaturalQuestions/natural_questions/v1.0/sample/nq-" \
                           + self._split + "-sample.jsonl.gz"

        with gzip.open(single_file_path, 'r') as f:
            data = []
            for example in f:
                data.append(json.loads(example))



        contexts = []
        total_qas_count = 0
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
                             'part': 'text',
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
            open_ended = {'annotators_answer_candidates': answer_candidates}
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
            total_qas_count += 1

            if (self._sample_size != None and total_qas_count > self._sample_size):
                break

            # TODO make the size limit a param
            # if we reached the size of a dataset file, preprocess and upload the results...
            if (self._max_contexts_in_file is not None and len(contexts) >= self._max_contexts_in_file):
                self.done_processing = False
                yield self._preprocessor.tokenize_and_detect_answers(contexts)
                contexts = []
                self.output_file_count += 1

        self.done_processing = True
        yield self._preprocessor.tokenize_and_detect_answers(contexts)