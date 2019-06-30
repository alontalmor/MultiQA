import json
from datasets.multiqa_dataset import MultiQA_DataSet
from overrides import overrides
from allennlp.common.file_utils import cached_path
from common.uuid import gen_uuid
import tqdm
import zipfile


class DROP(MultiQA_DataSet):
    """

    """

    def __init__(self):
        self.DATASET_NAME = 'DROP'

    @overrides
    def build_header(self, contexts, split, preprocessor):
        header = {}

        return header

    @overrides
    def build_contexts(self, split, preprocessor, sample_size):
        single_file_path = cached_path("https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip")
        with zipfile.ZipFile(single_file_path, 'r') as archive:
            data = json.loads(archive.read('drop_dataset/drop_dataset_' + split + '.json'))

        contexts = []
        for id, context in tqdm.tqdm(data.items(), total=len(data), ncols=80):

            qas = []
            for qa in context['qa_pairs']:
                # From the DROP eval script:
                # https://github.com/allenai/allennlp/blob/master/allennlp/tools/drop_eval.py
                # predicted = predicted_answers[query_id]
                # candidate_answers = [qa_pair["answer"]]
                # if "validated_answers" in qa_pair and qa_pair["validated_answers"]:
                #    candidate_answers += qa_pair["validated_answers"]
                # for answer in candidate_answers:
                #    gold_answer, gold_type = answer_json_to_strings(answer)
                #    em_score, f1_score = get_metrics(predicted, gold_answer)
                #    if gold_answer[0].strip() != "":
                #        max_em_score = max(max_em_score, em_score)
                #        max_f1_score = max(max_f1_score, f1_score)
                #        if max_em_score == em_score or max_f1_score == f1_score:
                #            max_type = gold_type
                #



                new_qa = {'qid': self.DATASET_NAME + '_q_' + qa['query_id'],
                          'question': qa['question']}
                answer_candidates = []

                # building a list of original answer candidates to iterate on, main answer will be first.
                org_answer_candidates = [qa['answer']]
                if "validated_answers" in qa:
                    org_answer_candidates += qa["validated_answers"]

                for answer_candidate in org_answer_candidates:
                    new_ans_cand = {}

                    if len(answer_candidate['spans']) > 0:
                        new_ans_cand['extractive'] = {"list":[{"answer": span} for span in answer_candidate['spans']]}

                    if answer_candidate['number'] != '':
                        new_ans_cand['number'] = {"single_answer": answer_candidate['number']}

                    if answer_candidate['date']['day'] != '':
                        new_ans_cand['date'] = {"single_answer": answer_candidate['date']}

                    if new_ans_cand != {}:
                        answer_candidates.append(new_ans_cand)

                new_qa['answers'] = {'open-ended': {'answer_candidates': answer_candidates}}
                qas.append(new_qa)

            contexts.append({"id": self.DATASET_NAME + '_' + id,
                             "context": {"documents": [{"text": context['passage'],
                                                        "url": context['wiki_url']}]},
                             "qas": qas})

        if sample_size != None:
            contexts = contexts[0:sample_size]
        contexts = preprocessor.tokenize_and_detect_answers(contexts)

        # detect answers

        # save dataset

        return contexts