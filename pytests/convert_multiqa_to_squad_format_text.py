# pylint: disable=no-self-use,invalid-name
import pytest

from convert_multiqa_to_squad_format import multiqa_to_squad
from common.preprocess import MultiQAPreProcess

class TestConvertMultiQAToSQuADFormat:
    @pytest.mark.parametrize("dataset", ("SQuAD1-1","SQuAD2-0"))
    def test_build_contexts(self, dataset):

        assert multiqa_to_squad(["https://multiqa.s3.amazonaws.com/data/" + dataset + "_dev.jsonl.gz"])