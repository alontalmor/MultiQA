# pylint: disable=no-self-use,invalid-name
import pytest
import json
import tqdm
from jsonschema import validate

from datasets.SearchQA.searchqa import SearchQA
from common.preprocess import MultiQAPreProcess


class TestSearchQADataset:
    # TODO testing only dev for now
    @pytest.mark.parametrize("split", {"dev"} )
    def test_build_contexts(self, split):
        N_PROCESSES = 1
        sample_size = 10
        max_contexts_in_file = 5
        dataset_version = None
        dataset_flavor = None
        dataset_specific_props = None
        custom_input_file = None
        preprocessor = MultiQAPreProcess(N_PROCESSES)
        dataset = SearchQA(preprocessor, split, dataset_version, dataset_flavor, dataset_specific_props, \
                 sample_size, max_contexts_in_file, custom_input_file)

        # loading multiqa schema
        with open("datasets/multiqa_jsonschema.json") as f:
            multiqa_schema = json.load(f)

        for contexts in dataset.build_contexts():
            header = dataset.build_header(contexts)

            # validating each context
            for context in tqdm.tqdm(contexts, total=len(contexts), desc="validating all contexts"):
                assert validate(instance=context, schema=multiqa_schema) is None