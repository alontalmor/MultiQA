import json
from datasets.multiqa_dataset import MultiQA_DataSet
from overrides import overrides
from allennlp.common.file_utils import cached_path
from common.uuid import gen_uuid
import tqdm
import os
import tarfile
import logging
logger = logging.getLogger(__name__)

class TriviaQA(MultiQA_DataSet):
    """

    """

    def __init__(self):
        self.DATASET_NAME = 'TriviaQA'

    @overrides
    def build_header(self, preprocessor, contexts, split, dataset_version, dataset_flavor, dataset_specific_props):
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
    def build_contexts(self, preprocessor, split, sample_size, dataset_version, dataset_flavor, dataset_specific_props, input_file):

        if dataset_flavor == "unfiltered":
            data_path = "data/triviaqa-unfiltered"
        else:
            data_path = "data/triviaqa-rc"
        evidence_path = "data/triviaqa-rc/evidence"
        dataset_url = "https://nlp.cs.washington.edu/triviaqa/" + data_path + ".tar.gz"

        if not input_file:
            # Checking if the data is already extracted to the data directory
            if not os.path.exists(data_path):
                logger.info('Getting the dataset from cache, this may take some time first time the dataset needs to be downloaded')
                single_file_path = cached_path(dataset_url)
                logger.info('Getting the data file from the dataset tar, this may also take some time...')
                dataset_tar = tarfile.open(single_file_path, "r:gz")
                dataset_tar.extractall(path=data_path)
                dataset_tar.close()

            if not os.path.exists(evidence_path):
                logger.info('Getting the dataset from cache, this may take some time first time the dataset needs to be downloaded')
                single_file_path = cached_path("https://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz")
                logger.info('Getting the data file from the dataset tar, this may also take some time...')
                dataset_tar = tarfile.open(single_file_path, "r:gz")
                dataset_tar.extractall(path="data/triviaqa-rc")
                dataset_tar.close()
        else:
            single_file_path = cached_path(input_file)



        if dataset_flavor == "unfiltered":
            with open(data_path + '/triviaqa-unfiltered/unfiltered-web-' + split + '.json') as f:
                examples = json.load(f)['Data']
        elif dataset_flavor == "wiki":
            with open(data_path + '/qa/wikipedia-' + split + '.json') as f:
                examples = json.load(f)['Data']
        elif dataset_flavor == "web":
            with open(data_path + '/qa/web-' + split + '.json') as f:
                examples = json.load(f)['Data']
        else:
            logger.error('The dataset_flavor ' + dataset_flavor + ' is not supported, please use unfiltered,wiki,web')
            return []

        logger.info('Iterating over the examples')
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

            # SearchResults
            if 'SearchResults' in example:
                for search_res in example['SearchResults']:
                    document = {'rank': search_res['Rank'], 'title': search_res['Title'], \
                                'url':search_res['Url'],'snippet': search_res['Description']}
                    if 'Filename' in search_res:
                        with open(evidence_path + '/web/' + search_res['Filename']) as f:
                            text = f.read()
                        document.update({'text': text })
                    documents.append(document)

            # EntityPages
            for res in example['EntityPages']:
                if 'Filename' in res:
                    with open(evidence_path + '/wikipedia/' + res['Filename']) as f:
                        text = f.read()
                    documents.append({'title': res['Title'], \
                                  'text': text})


            contexts.append({"id": self.DATASET_NAME + '_'  + str(example['QuestionId']),
                             "context": {"documents": documents},
                             "qas": qas})


        if sample_size != None:
            contexts = contexts[0:sample_size]
        contexts = preprocessor.tokenize_and_detect_answers(contexts)

        return contexts