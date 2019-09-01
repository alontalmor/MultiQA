# pylint: disable=no-self-use,invalid-name
import pytest
import json
import tqdm
from jsonschema import validate

from datasets.HotpotQA.hotpotqa import HotpotQA
from common.preprocess import MultiQAPreProcess


class TestHotpotQADataset:
    # TODO Only distractor setting supported for now
    #@pytest.mark.parametrize("dataset_flavor", ("distractor", "full-wiki"))
    @pytest.mark.parametrize("split,dataset_specific_props", [("dev",["use_all_answers_in_training","original_context_order"]), ("train",[])])
    def test_build_contexts(self, split,dataset_specific_props):
        N_PROCESSES = 1
        sample_size = 10
        max_contexts_in_file = 5
        dataset_version = None
        dataset_flavor = None
        #dataset_specific_props = ''
        custom_input_file = None
        preprocessor = MultiQAPreProcess(N_PROCESSES)
        dataset = HotpotQA(preprocessor, split, dataset_version, dataset_flavor, dataset_specific_props, \
                 sample_size, max_contexts_in_file, custom_input_file)

        # loading multiqa schema
        with open("datasets/multiqa_jsonschema.json") as f:
            multiqa_schema = json.load(f)

        for contexts in dataset.build_contexts():
            header = dataset.build_header(contexts)

            # validating each context
            for context in tqdm.tqdm(contexts, total=len(contexts), desc="validating all contexts"):
                assert validate(instance=context, schema=multiqa_schema) is None



