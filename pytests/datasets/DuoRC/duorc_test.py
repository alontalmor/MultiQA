# pylint: disable=no-self-use,invalid-name
import pytest
import json
import tqdm
from jsonschema import validate
from datasets.DuoRC.duorc import DuoRC
from common.preprocess import MultiQAPreProcess


class TestDuoRCDataset:
    @pytest.mark.parametrize("split", ("dev","train"))
    @pytest.mark.parametrize("dataset_flavor", ("Paraphrase", "Self"))
    def test_build_contexts(self, split,dataset_flavor):
        N_PROCESSES = 1
        sample_size = 10
        max_contexts_in_file = 5
        dataset_version = None
        dataset_specific_props = None
        custom_input_file = None
        preprocessor = MultiQAPreProcess(N_PROCESSES)
        dataset = DuoRC(preprocessor, split, dataset_version, dataset_flavor, dataset_specific_props, \
                 sample_size, max_contexts_in_file, custom_input_file)

        # loading multiqa schema
        with open("datasets/multiqa_jsonschema.json") as f:
            multiqa_schema = json.load(f)

        for contexts in dataset.build_contexts():
            header = dataset.build_header(contexts)

            # validating each context
            for context in tqdm.tqdm(contexts, total=len(contexts), desc="validating all contexts"):
                assert validate(instance=context, schema=multiqa_schema) is None