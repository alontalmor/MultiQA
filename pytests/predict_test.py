import pytest
import argparse
from datasets.SQuAD.squad import SQuAD
from datasets.HotpotQA.hotpotqa import HotpotQA
from predict import predict

class TestPredict:
    @pytest.mark.parametrize(
        "dataset_name,dataset",
        [
            ("SQuAD", "SQuAD1-1"),
            ("HotpotQA", "HotpotQA"),
        ],
    )
    def test_build_challenge(self, dataset_name, dataset):
        parse = argparse.ArgumentParser("")
        parse.add_argument("--model")
        parse.add_argument("--dataset")
        parse.add_argument("--dataset_name")
        parse.add_argument("--prediction_filepath", type=str, default=None)
        parse.add_argument("--cuda_device", type=int, default=-1)
        parse.add_argument("--sample_size", type=int, default=-1)
        model = 'https://multiqa.s3.amazonaws.com/models/BERTBase/' + dataset + '.tar.gz'
        dataset = 'https://multiqa.s3.amazonaws.com/data/' + dataset + '_dev.jsonl.gz'
        # In the test no output file will be produced, change -out to create an output
        args = parse.parse_args(["--model", model,"--dataset", dataset, "--dataset_name", dataset_name, "--sample_size", "5"])

        predict(args)
