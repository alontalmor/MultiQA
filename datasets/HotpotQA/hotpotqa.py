import json
from datasets.multiqa_dataset import MultiQA_DataSet
from overrides import overrides
from allennlp.common.file_utils import cached_path
import tqdm


class HotpotQA(MultiQA_DataSet):
    """

    """

    def __init__(self):
        self.DATASET_NAME = 'HotpotQA'

    @overrides
    def build_header(self, contexts, split, preprocessor, dataset_version, dataset_flavor):
        header = {
            "dataset_name": self.DATASET_NAME,
            "split": split,
            "dataset_url": "https://hotpotqa.github.io/",
            "license": "http://creativecommons.org/licenses/by-sa/4.0/legalcode",
            "data_source": "Wikipedia",
            "context_answer_detection_source": "MultiQA",
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
    def format_predictions(self, predictions):
        return {"answer": predictions, "sp": {}}

    @overrides
    def build_contexts(self, split, preprocessor, sample_size, dataset_version, dataset_flavor, input_file):
        if input_file is not None:
            single_file_path = cached_path(input_file)
        else:
            if split == 'train':
                single_file_path = cached_path("http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json")
            elif split == 'dev':
                single_file_path = cached_path("http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json")

        with open(single_file_path, 'r') as myfile:
            data = json.load(myfile)

        contexts = []
        for example in tqdm.tqdm(data, total=len(data), ncols=80):

            # choosing only the gold paragraphs
            #gold_paragraphs = []
            #for supp_fact_title in set([supp_fact[0] for supp_fact in example['supporting_facts']]):
            #    for context in example['context']:
            #        # finding the gold context
            #        if context[0] == supp_fact_title:
            #            gold_paragraphs.append(context)

            documents = []
            supporting_context = []
            for doc_id, para in enumerate(example['context']):

                # calcing the sentence_start_bytes for the supporting facts in hotpotqa
                offset = 0
                sentence_start_bytes = [0]
                for sentence in para[1]:
                    offset += len(sentence) + 1
                    sentence_start_bytes.append(offset)
                sentence_start_bytes = sentence_start_bytes[:-1]

                # choosing only the gold paragraphs
                for supp_fact in example['supporting_facts']:
                    # finding the gold context
                    if para[0] == supp_fact[0] and len(sentence_start_bytes) > supp_fact[1]:
                        supporting_context.append({'doc_id':doc_id,
                                                   'part':'text',
                                                   'start_byte': sentence_start_bytes[supp_fact[1]],
                                                   'text':para[1][supp_fact[1]]})

                # joining all sentences into one
                documents.append({'text':' '.join(para[1]) + ' ',
                                 'title': para[0],
                                 'metadata': {"text": {"sentence_start_bytes": sentence_start_bytes}}})

            if example['answer'].lower() == 'yes':
                answers = {'open-ended': {'answer_candidates': [{'yesno':{'single_answer':'yes'}}]}}
            elif example['answer'].lower() == 'no':
                answers = {'open-ended': {'answer_candidates': [{'yesno':{'single_answer':'no'}}]}}
            else:
                answers = {'open-ended': {'answer_candidates': [{'extractive': {'single_answer': {'answer': example['answer']}}}]}}


            qas = [{"qid": self.DATASET_NAME + '_q_' + example['_id'],
                    "metadata":{'type':example['type'],'level':example['level']},
                    "supporting_context": supporting_context,
                    "question": example['question'],
                    "answers": answers,
                    }]

            contexts.append({"id": self.DATASET_NAME + '_' + example['_id'],
                             "context": {"documents": documents},
                             "qas": qas})

        if sample_size != None:
            contexts = contexts[0:sample_size]

        if split == 'train':
            ans_in_supp_context = True
        else:
            ans_in_supp_context = False
        contexts = preprocessor.tokenize_and_detect_answers(contexts, search_answer_within_supp_context=ans_in_supp_context)

        return contexts