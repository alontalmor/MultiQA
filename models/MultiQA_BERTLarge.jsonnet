{
    "random_seed": 0,
    "dataset_reader": {
        "MAX_WORDPIECES": 384,
        "type": "multiqa_reader",
        "is_training": true,
        "lazy": true,
        "sample_size": -1,
        "support_yesno":false,
        "support_cannotanswer":false,
        "token_indexers": {
            "bert": {
                  "type": "bert-pretrained",
                  "pretrained_model": "bert-large-uncased",
                  "do_lowercase": true,
                  "use_starting_offsets": true
              }
        }
    },
    "validation_dataset_reader": {
        "MAX_WORDPIECES": 384,
        "support_yesno":false,
        "support_cannotanswer":false,
        "type": "multiqa_reader",
        "lazy": true,
        "sample_size": -1,
        "token_indexers": {
            "bert": {
                  "type": "bert-pretrained",
                  "pretrained_model": "bert-large-uncased",
                  "do_lowercase": true,
                  "use_starting_offsets": true
              }
        }
    },
    "iterator": {
        "type": "basic",
        "batch_size": 2,
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
                    "pretrained_model": "bert-large-uncased",
                    "requires_grad":true
                }
            }
        }
    },
    "train_data_path": "https://multiqa.s3.amazonaws.com/data/SQuAD1-1_train.jsonl.gz",
    "validation_data_path": "https://multiqa.s3.amazonaws.com/data/SQuAD1-1_dev.jsonl.gz",
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
        "type": "basic",
        "batch_size": 20
    }
}

