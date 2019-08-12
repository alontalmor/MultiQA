from typing import Dict, Any
import argparse
import logging
import json
import os
import _jsonnet
from allennlp.common.file_utils import cached_path
from allennlp.commands.train import train_model_from_file
from allennlp.common.util import import_submodules
from allennlp.common.util import prepare_environment
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.models.archival import load_archive
from allennlp.training.util import evaluate
from allennlp.common import Params

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("command", type=str, help="one of the following options train, evaluate, generalize")
    parse.add_argument("--datasets", type=str, help="", default=None)
    parse.add_argument("--model", type=str, help="", default=None)
    parse.add_argument("--serialization_dir", type=str, help="the directory storing the intermediate files and output", default=None)
    parse.add_argument("--cuda_device", type=str, help="Cuda device ID", default="-1")
    parse.add_argument("--split", type=str, help="dev / test", default="dev")
    parse.add_argument("--bert_type", type=str, help="Base / Large /", default="Base")
    parse.add_argument("--config", type=str, help="dev / test", default=None)
    parse.add_argument("--output_path", type=str, help="directory to which results JSONs of eval will written", default='results/eval/')
    parse.add_argument("--models_dir", type=str, help="directory containing the models used for eval , (please add '/' at the end)", default=None)
    parse.add_argument("--data_dir", type=str, help="directory containing the multiqa format datasets , (please add '/' at the end and make sure to have a headers directory with all headers under your specified path)", default='https://multiqa.s3.amazonaws.com/data/')
    parse.add_argument("--t_total", type=str, help="used for training, see BERT's learning rate schedule for details", default=None)
    args = parse.parse_args()

    import_submodules("models")

    # TODO add best config for specific datasets as default, not one general default...
    if args.config is None:
        config = 'models/MultiQA_BERT' + args.bert_type + '.jsonnet'
    else:
        config = args.config
    config_params = Params(json.loads(_jsonnet.evaluate_file(config)))

    if args.command == 'train':
        # building the default dataset urls
        train_datasets = [args.data_dir + dataset + '_train.jsonl.gz' for dataset in args.datasets.split(',')]
        val_datasets = [args.data_dir + dataset + '_' + args.split + '.jsonl.gz' for dataset in args.datasets.split(',')]

        # calculating the t_total
        if args.t_total == None:
            logging.info('getting headers of the chosen dataset in order to compute learning rate schedule t_total')
            total_number_of_examples = 0
            for header_url in [args.data_dir + 'headers/' + dataset + '_train.json' for dataset in
                              args.datasets.split(',')]:
                with open(cached_path(header_url),'r') as f:
                    header = json.load(f)
                    total_number_of_examples += header['number_of_qas']
            t_total = int(total_number_of_examples / float(config_params['iterator']['batch_size']) \
                    * float(config_params['trainer']['num_epochs'])) \
                    / len(args.cuda_device.split(','))

        if args.serialization_dir is None:
            serialization_dir = 'models/' + args.datasets.replace(',','_')
        else:
            serialization_dir = args.serialization_dir

        overrides = {
            'train_data_path': ','.join(train_datasets),
            'validation_data_path': ','.join(val_datasets),
            'trainer': {
                'cuda_device': args.cuda_device,
                'optimizer': {'t_total': t_total}
            }
        }

        overrides_str = str(overrides).replace('True', 'true').replace('False', 'false')
        train_model_from_file(config, serialization_dir, overrides_str, True, False, True, "", "")
    elif args.command == 'evaluate':

        if args.models_dir is None:
            model_path = 'https://multiqa.s3.amazonaws.com/models/BERT' + args.bert_type + '/' + args.model + '.tar.gz'
        else:
            model_path = args.models_dir + args.model + '.tar.gz'
        model_cached_path = cached_path(model_path)


        overrides_str = ''
        # Load from archive
        archive = load_archive(model_cached_path, int(args.cuda_device), overrides_str, '')
        prepare_environment(config_params)
        model = archive.model
        model.eval()

        # Load the evaluation data
        validation_dataset_reader_params = config_params.get('validation_dataset_reader', None)
        dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)

        # running over all validation datasets specified
        val_dataset_names = args.datasets.split(',')
        val_datasets = [args.data_dir + dataset + '_' + args.split + '.jsonl.gz' for dataset in val_dataset_names]

        for val_dataset_path,val_dataset_name in zip(val_datasets,val_dataset_names):
            # This is a bit strange but there is a lot of config "popping" going on implicitly in allennlp
            # so we need to have the full config reloaded every iteration...
            config_params = Params(json.loads(_jsonnet.evaluate_file(config)))

            logger.info("Reading evaluation data from %s", val_dataset_path)
            instances = dataset_reader.read(val_dataset_path)

            # loading iterator
            iterator_params = config_params.get("validation_iterator", None)
            iterator = DataIterator.from_params(iterator_params)
            iterator.index_with(model.vocab)

            metrics = evaluate(model, instances, iterator, int(args.cuda_device), '')

            logger.info("Finished evaluating " + val_dataset_name)
            logger.info("Metrics:")
            for key, metric in metrics.items():
                logger.info("%s: %s", key, metric)

            if not os.path.exists(args.output_path):
                os.makedirs(args.output_path)
            output_path = args.output_path + args.model + '_BERT' + args.bert_type + '_eval-on_' \
                          + val_dataset_name + '_' + args.split + '.json'
            with open(output_path, "w") as file:
                json.dump(metrics, file, indent=4)
        return metrics

    elif args.command == 'generalize':
        logging.error('The command %s is not yet supported' % (args.command))
    else:
        logging.error('The command %s is not supported' % (args.command))

if __name__ == '__main__':
    main()