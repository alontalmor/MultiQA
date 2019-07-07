{
    "dataset_reader": {
        "type": "multiqa_reader",
        "is_training": true,
        "lazy": true,
        "sample_size": -1,
        "token_indexers": {
            "bert": {
                  "type": "bert-pretrained",
                  "pretrained_model": "bert-base-uncased",
                  "do_lowercase": true,
                  "use_starting_offsets": true
              }
        }
    },
    "validation_dataset_reader": {
        "type": "multiqa_reader",
        "lazy": true,
        "sample_size": -1,
        "token_indexers": {
            "bert": {
                  "type": "bert-pretrained",
                  "pretrained_model": "bert-base-uncased",
                  "do_lowercase": true,
                  "use_starting_offsets": true
              }
        }
    },
    "iterator": {
        "type": "mrqa_iterator",
        "batch_size": 6,
        "max_instances_in_memory": 1000
    },
    "model": {
        "type": "multiqa_bert",
        "initializer": [],
        "text_field_embedder": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": {
                "bert": ["bert", "bert-offsets"]
            },
            "token_embedders": {
                "bert": {
                    "type": "bert-pretrained",
                    "pretrained_model": "bert-base-uncased",
                    "requires_grad":true
                }
            }
        }
    },
    "train_data_path": "s3://multiqa/data/SQuAD_train.jsonl.gz",
    "validation_data_path": "s3://multiqa/data/SQuAD_dev.jsonl.gz",
    "trainer": {
        "cuda_device": -1,
        "num_epochs": 2,
        "optimizer": {
            "type": "bert_adam",
            "lr": 0.00003,
            "warmup":0.1,
            "t_total": 29100
        },
        "patience": 10,
        "validation_metric": "+f1"
    },
    "validation_iterator": {
        "type": "mrqa_iterator",
        "batch_size": 6
    }
}

