import json
from datasets.multiqa_dataset import MultiQA_DataSet
from overrides import overrides
from allennlp.common.file_utils import cached_path
from common.uuid import gen_uuid
import tqdm
import tarfile


class TriviaQA(MultiQA_DataSet):
    """

    """

    def __init__(self):
        self.DATASET_NAME = 'TriviaQA'

    @overrides
    def build_header(self, contexts, split, preprocessor, dataset_version, dataset_flavor):
        header = {
            "dataset_name": self.DATASET_NAME,
            "version": '',
            "flavor": dataset_flavor,
            "split": split,
            "dataset_url": "https://nlp.cs.washington.edu/triviaqa/",
            "license": "",
            "data_source": "Web" if dataset_flavor!="wiki" else "Wikipedia",
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
    def build_contexts(self, split, preprocessor, sample_size, dataset_version, dataset_flavor, input_file, build_properties):
        if not input_file:
            if dataset_flavor == "unfiltered":
                single_file_path = cached_path("https://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered.tar.gz")
            else:
                single_file_path = cached_path("https://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz")
        else:
            single_file_path = cached_path(input_file)

        tar = tarfile.open(single_file_path, "r:gz")
        for member in tar.getmembers():
            if member.name.find(dataset_flavor) > -1 and member.name.find(split) > -1:
                f = tar.extractfile(member)
                if f is not None:
                    examples = json.load(f)['Data']
                break

        contexts = []
        for example in tqdm.tqdm(examples, total=len(examples), ncols=80):


            qas = []
            new_qa = {'qid':self.DATASET_NAME + '_q_' + str(example['QuestionId']),
                        'question':example['Question']}
            new_qa['answers'] = new_qa['answers'] = {"open-ended":{
                'answer_candidates': [
                    {'extractive':
                        {"single_answer": { 'answer': example['Answer']['Value'], \
                                            'aliases': example['Answer']['NormalizedAliases']}}
                     }
                ]}}
            qas.append(new_qa)

            documents = []
            for search_res in example['SearchResults']:
                documents.append({'title': search_res['Title'],'text': search_res['Description'],'url':search_res['Url']})

            contexts.append({"id": self.DATASET_NAME + '_'  + str(example['QuestionId']),
                             "context": {"documents": documents},
                             "qas": qas})


        if sample_size != None:
            contexts = contexts[0:sample_size]
        contexts = preprocessor.tokenize_and_detect_answers(contexts)

        return contexts