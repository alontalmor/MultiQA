# pylint: disable=no-self-use,invalid-name
import pytest

from datasets.BoolQ.boolq import BoolQ
from common.preprocess import MultiQAPreProcess


class TestBoolQDataset:
    @pytest.mark.parametrize("split", ("dev","train"))
    def test_build_contexts(self, split):
        N_PROCESSES = 1
        sample_size = 10
        max_contexts_in_file = 5
        dataset_version = None
        dataset_flavor = None
        dataset_specific_props = None
        custom_input_file = None
        preprocessor = MultiQAPreProcess(N_PROCESSES)
        dataset = BoolQ(preprocessor, split, dataset_version, dataset_flavor, dataset_specific_props, \
                 sample_size, max_contexts_in_file, custom_input_file)

        for contexts in dataset.build_contexts():
            header = dataset.build_header(contexts)