import argparse
import json
import _jsonnet
from allennlp.common.file_utils import cached_path
from allennlp.commands.train import train_model_from_file
from allennlp.common.util import import_submodules
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)

def main():
    parse = argparse.ArgumentParser("")
    parse.add_argument("command", type=str, help="one of the following options train, evaluate, generalize")
    parse.add_argument("--datasets", type=str, help="", default=None)
    parse.add_argument("--model", type=str, help="", default=None)
    parse.add_argument("--cuda_device", type=str, help="Cuda device ID", default="-1")
    parse.add_argument("--split", type=str, help="dev / test", default="dev")
    parse.add_argument("--config", type=str, help="dev / test", default=None)
    parse.add_argument("--t_total", type=str, help="used for training, see BERT's learning rate schedule for details", default=None)
    args = parse.parse_args()

    import_submodules("models")

    if args.command == 'train':
        # TODO add best config for specific datasets
        if args.config is None:
            param_path = 'models/MultiQA_BERTBase.jsonnet'
        config_dict = json.loads(_jsonnet.evaluate_file(param_path))

        # building the default dataset urls
        train_datasets = ['https://multiqa.s3.amazonaws.com/data/' + dataset + '_train.jsonl.gz' for dataset in args.datasets.split(',')]
        val_datasets = ['https://multiqa.s3.amazonaws.com/data/' + dataset + '_' + args.split + '.jsonl.gz' for dataset in args.datasets.split(',')]

        # calculating the t_total
        if args.t_total == None:
            logging.info('getting headers of the chosen dataset in order to compute learning rate schedule t_total')
            total_number_of_examples = 0
            for header_url in ['https://multiqa.s3.amazonaws.com/data/headers/' + dataset + '_train.json' for dataset in
                              args.datasets.split(',')]:
                with open(cached_path(header_url),'r') as f:
                    header = json.load(f)
                    total_number_of_examples += header['number_of_qas']
            t_total = int(total_number_of_examples / float(config_dict['iterator']['batch_size']) \
                    * float(config_dict['trainer']['num_epochs'])) \
                    / len(args.cuda_device.split(','))

        serialization_dir = 'models/' + args.datasets.replace(',','_')
        overrides = {
            'train_data_path': ','.join(train_datasets),
            'validation_data_path': ','.join(val_datasets),
            'trainer': {
                'cuda_device': args.cuda_device,
                'optimizer': {'t_total': t_total}
            }
        }

        overrides_str = str(overrides).replace('True', 'true').replace('False', 'false')
        train_model_from_file(param_path, serialization_dir, overrides_str, True, False, True, "", "")
    else:
        logging.error('The command %s is not supported' % (args.command))

if __name__ == '__main__':
    main()