


## Training

The AllenNLP train command is used for training. The training and validation files should be provided as an override to the base config. 

 `python -m allennlp.run train models/MultiQA_BERTBase.jsonnet -s [SERIALIZATION_DIR] -o "{'train_data_path': 'https://multiqa.s3.amazonaws.com/data/[TRAINING_SET1],https://multiqa.s3.amazonaws.com/data/[TRAINING_SET2(optional)]', 'validation_data_path': 'https://multiqa.s3.amazonaws.com/data[DEV_SET]', 'trainer': {'cuda_device': [CUDE DEVICEID or -1 for CPU], 'num_epochs': [NUM_OF_EPOCHES, usually 2], 'optimizer': {'type': 'bert_adam', 'lr': 3e-05, 'warmup': 0.1, 't_total': [T_TOTAL = #TRAINING_EXAMPLES / BATCH_SIZE(default=8) * NUM_OF_EPOCHES}}}" --include-package models`.
 
 ### Example - BERTBase
 
 single dataset training:
 
 `python -m allennlp.run train models/MultiQA_BERTBase.jsonnet -s Results/SQuAD/Train -o "{'train_data_path': 'https://multiqa.s3.amazonaws.com/data/SQuAD1-1_train.jsonl.gz', 'validation_data_path': 'https://multiqa.s3.amazonaws.com/data/SQuAD1-1_dev.jsonl.gz', 'trainer': {'cuda_device': -1, 'optimizer': {'t_total': 29000}}}" --include-package models `

 ##Training on all multiple sets (MultiQA)
 
 `python -m allennlp.run train models/MultiQA_BERTBase.jsonnet -s ../Models/MultiTrain -o "{'dataset_reader': {'sample_size': 75000}, 'validation_dataset_reader': {'sample_size': 1000}, 'train_data_path': 'https://multiqa.s3.amazonaws.com/data/SQuAD.jsonl.gz,https://multiqa.s3.amazonaws.com/data/NewsQA.jsonl.gz,https://multiqa.s3.amazonaws.com/data/HotpotQA.jsonl.gz,https://multiqa.s3.amazonaws.com/data/SearchQA.jsonl.gz,https://multiqa.s3.amazonaws.com/data/TriviaQA-web.jsonl.gz,https://multiqa.s3.amazonaws.com/data/NaturalQuestionsShort.jsonl.gz', 'validation_data_path': 'https://multiqa.s3.amazonaws.com/data/SQuAD.jsonl.gz,https://multiqa.s3.amazonaws.com/data/NewsQA.jsonl.gz,https://multiqa.s3.amazonaws.com/data/HotpotQA.jsonl.gz,https://multiqa.s3.amazonaws.com/data/SearchQA.jsonl.gz,https://multiqa.s3.amazonaws.com/data/TriviaQA-web.jsonl.gz,https://multiqa.s3.amazonaws.com/data/NaturalQuestionsShort.jsonl.gz', 'trainer': {'cuda_device': [2,3,4,5], 'num_epochs': '2', 'optimizer': {'type': 'bert_adam', 'lr': 3e-05, 'warmup': 0.1, 't_total': '120000'}}}" --include-package mrqa_allennlp`
 