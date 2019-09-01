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
        self.DATASET_NAME = 'TriviaQA'
        self.output_file_count = 0
        self.done_processing = True

    @overrides
    def build_header(self, contexts):
        header = {
            "dataset_name": self.DATASET_NAME,
            "version": '',
            "flavor": self._dataset_flavor,
            "split": self._split,
            "dataset_url": "https://nlp.cs.washington.edu/triviaqa/",
            "license": "",
            "data_source": "Web" if self._dataset_flavor!="wiki" else "Wikipedia",
            "context_answer_detection_source": 'MultiQA',
            "tokenization_source": "MultiQA",
            "full_schema": super().compute_schema(contexts),
            "text_type": "abstract",
            "number_of_qas": sum([len(context['qas']) for context in contexts]),
            "number_of_contexts": len(contexts),
            "file_num":self.output_file_count,
            "next_file_exists": not self.done_processing,
            "readme": "",
            "multiqa_version": super().get_multiqa_version()
        }

        return header

    @overrides
    def build_contexts(self):

        if self._dataset_flavor == "unfiltered":
            data_path = "data/triviaqa-unfiltered"
        else:
            data_path = "data/triviaqa-rc"
        evidence_path = "data/triviaqa-rc/evidence"
        dataset_url = "https://nlp.cs.washington.edu/triviaqa/" + data_path + ".tar.gz"

        if not self._custom_input_file:
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
            single_file_path = cached_path(self._custom_input_file)



        if self._dataset_flavor == "unfiltered":
            with open(data_path + '/triviaqa-unfiltered/unfiltered-web-' + self._split + '.json') as f:
                examples = json.load(f)['Data']
        elif self._dataset_flavor == "wiki":
            with open(data_path + '/qa/wikipedia-' + self._split + '.json') as f:
                examples = json.load(f)['Data']
        elif self._dataset_flavor == "web":
            with open(data_path + '/qa/web-' + self._split + '.json') as f:
                examples = json.load(f)['Data']
        else:
            logger.error('The dataset_flavor ' + self._dataset_flavor + ' is not supported, please use unfiltered,wiki,web')
            return []

        logger.info('Iterating over the examples')
        contexts = []
        total_qas_count = 0
        # done processing is used in the next file exists
        self.done_processing = False
        for example in tqdm.tqdm(examples, total=len(examples), ncols=80):

            qas = []
            new_qa = {'qid':self.DATASET_NAME + '_q_' + str(example['QuestionId']),
                        'question':example['Question']}
            new_qa['answers'] = new_qa['answers'] = {"open-ended":{
                'annotators_answer_candidates': [
                    {'single_answer':
                        {'extractive': { 'answer': example['Answer']['Value'], \
                                            'aliases': example['Answer']['NormalizedAliases']}}
                     }
                ]}}
            qas.append(new_qa)

            documents = []

            # SearchResults
            if 'SearchResults' in example:
                for search_res in example['SearchResults']:
                    document = {'metadata':{'rank': search_res['Rank']}, 'title': search_res['Title'], \
                                'url':search_res['Url'],'snippet': search_res['Description']}
                    if 'Filename' in search_res:
                        with open(evidence_path + '/web/' + search_res['Filename']) as f:
                            text = f.read()
                        document.update({'text': text })

                    # In the web version each context documents is treated as a separate example
                    if self._dataset_flavor == "web":
                        contexts.append({"id": self.DATASET_NAME + '_' + str(example['QuestionId']) + '_' + search_res['Title'],
                                         "context": {"documents": [document]},
                                         "qas": qas})
                        total_qas_count += len(qas)
                    else:
                        documents.append(document)

            # EntityPages
            for res in example['EntityPages']:
                if 'Filename' in res:
                    with open(evidence_path + '/wikipedia/' + res['Filename']) as f:
                        text = f.read()

                    # In the web version each context documents is treated as a separate example
                    if self._dataset_flavor == "web":
                        contexts.append({"id": self.DATASET_NAME + '_' + str(example['QuestionId']) + '_' + res['Title'],
                                         "context": {"documents": [{'title': res['Title'], \
                                                'text': text}]},
                                         "qas": qas})
                        total_qas_count += len(qas)
                    else:
                        documents.append({'title': res['Title'], \
                                  'text': text})

            # In versions that are not web, all evidence belongs to a single question.
            if self._dataset_flavor != "web":
                contexts.append({"id": self.DATASET_NAME + '_'  + str(example['QuestionId']),
                                 "context": {"documents": documents},
                                 "qas": qas})
                total_qas_count += len(qas)


            if (self._sample_size != None and total_qas_count > self._sample_size):
                break

            # if we reached the size of a dataset file, preprocess and upload the results...
            if (self._max_contexts_in_file is not None and len(contexts) >= self._max_contexts_in_file):
                logger.info('producing one context file')
                yield self._preprocessor.tokenize_and_detect_answers(contexts)
                contexts = []
                self.output_file_count += 1

        logger.info('producing final context file')
        self.done_processing = True
        yield self._preprocessor.tokenize_and_detect_answers(contexts)
