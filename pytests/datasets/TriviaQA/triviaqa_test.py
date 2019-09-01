# pylint: disable=no-self-use,invalid-name
import pytest
import json
import tqdm
from jsonschema import validate

from datasets.TriviaQA.triviaqa import TriviaQA
from common.preprocess import MultiQAPreProcess


class TestTriviaQADataset:
    @pytest.mark.parametrize("dataset_flavor,max_contexts_in_file", [("wiki",5), ("unfiltered",5), ("web",2)])
    @pytest.mark.parametrize("split", ("train","dev"))
    def test_build_contexts(self, dataset_flavor, max_contexts_in_file, split):
        # TODO do we want more than 1?
        N_PROCESSES = 1
        sample_size = 10
        # max_contexts_in_file = 5
        dataset_version = None
        dataset_specific_props = None
        custom_input_file = None
        preprocessor = MultiQAPreProcess(N_PROCESSES)
        dataset = TriviaQA(preprocessor, split, dataset_version, dataset_flavor, dataset_specific_props, \
                 sample_size, max_contexts_in_file, custom_input_file)

        # loading multiqa schema
        with open("datasets/multiqa_jsonschema.json") as f:
            multiqa_schema = json.load(f)

        for contexts in dataset.build_contexts():
            header = dataset.build_header(contexts)

            # validating each context
            for context in tqdm.tqdm(contexts, total=len(contexts), desc="validating all contexts"):
                assert validate(instance=context, schema=multiqa_schema) is None


