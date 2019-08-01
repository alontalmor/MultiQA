# MultiQA

MultiQA is a project whose goal is to facilitate training and evaluating reading
comprehension models over arbitrary sets of datasets.
All datasets are in a single format, and it is accompanied by
an AllenNLP `DatasetReader` and model that enable easy training and evaluation
on multiple subsets of datasets.

This repository contains code for our paper [MultiQA: An Empirical Investigation of Generalization and Transfer in Reading Comprehension](https://arxiv.org/abs/1905.13453).

This work was performed at AllenAI institute of Artificial Intelligence.   

This project is constantly being improved, contributions, comments and suggestions are welcome!


## Datasets

Link to the single format dataset are provided in the Train, Dev and Test columns.
The BERT-Base column contains evaluation results (EM/F1) as well as a link to the trained model. 
The MultiQA-5Base column contain the link to the model (in the header) and evalution results for this model. This model is BERT-Base that has been trained on 5 datasets. 

| Dataset | Train | Dev | BERT-Base | MultiQA-5Base [(model)](https://multiqa.s3.amazonaws.com/models_new/BERTBase/SQuAD1-1_HotpotQA_NewsQA_TriviaQA_unfiltered_SearchQA__full.tar.gz)|
| :----- | :-----:|  :------------------: | :------------------: |  :------------------: |
| SQuAD-1.1 | [data](https://multiqa.s3.amazonaws.com/data/SQuAD1-1_train.jsonl.gz) | [data](https://multiqa.s3.amazonaws.com/data/SQuAD1-1_dev.jsonl.gz) | 80.1 / 87.5 [(model)](https://multiqa.s3.amazonaws.com/models_new/BERTBase/SQuAD1-1.tar.gz) | 81.7 / 88.8 |
| NewsQA | [data](https://multiqa.s3.amazonaws.com/data/NewsQA_train.jsonl.gz) | [data](https://multiqa.s3.amazonaws.com/data/NewsQA_dev.jsonl.gz) | 47.5 / 62.9 [(model)](https://multiqa.s3.amazonaws.com/models_new/BERTBase/NewsQA.tar.gz) | 48.3 / 64.7 |
| HotpotQA | [data](https://multiqa.s3.amazonaws.com/data/HotpotQA_train.jsonl.gz) | [data](https://multiqa.s3.amazonaws.com/data/HotpotQA_dev.jsonl.gz) | 50.1 / 63.2 [(model)](https://multiqa.s3.amazonaws.com/models_new/BERTBase/HotpotQA.tar.gz) | - |
| TriviaQA-unfiltered | [data](https://multiqa.s3.amazonaws.com/data/TriviaQA_unfiltered_train.jsonl.gz) | [data](https://multiqa.s3.amazonaws.com/data/TriviaQA_unfiltered_dev.jsonl.gz) | 59.4 / 65.2 [(model)](https://multiqa.s3.amazonaws.com/models_new/BERTBase/TriviaQA_unfiltered.tar.gz) | 59.0 / 64.7 |
| SearchQA | [data](https://multiqa.s3.amazonaws.com/data/SearchQA_train.jsonl.gz) | [data](https://multiqa.s3.amazonaws.com/data/SearchQA_dev.jsonl.gz) | 58.7 / 65.2 [(model)](https://multiqa.s3.amazonaws.com/models_new/BERTBase/SearchQA.tar.gz) | 58.8 / 65.3 |
| NaturalQuestions | Comming soon | Comming soon | Comming soon | Comming soon |

Datasets will be addeed weekly, so please stay tuned!
 
 
## Setup

### Setting up a virtual environment

1.  First, clone the repository:

    ```
    git clone https://github.com/alontalmor/MultiQA.git
    ```

2.  Change your directory to where you cloned the files:

    ```
    cd MultiQA
    ```

3.  Create a virtual environment with Python 3.6 or above:

    ```
    virtualenv venv --python=python3.7 (or python3.7 -m venv venv)
    ```

4.  Activate the virtual environment. You will need to activate the venv environment in each terminal in which you want to use WebAsKB.

    ```
    source venv/bin/activate (or source venv/bin/activate.csh)
    ```
5.  Install the required dependencies:

    ```
    pip3 install -r requirements.txt
    ```

### Data

The allennlp caching infra is used, so be sure to have enough disk space, and control the cache directory using ALLENNLP_CACHE_ROOT env variable.

## Build Dataset
    
   This will take a dataset from it's original URL and output the same dataset in the MultiQA format.  
   
  `python build_dataset.py --dataset_name HotpotQA --split train --output_file path/to/output.jsonl.gz --n_processes 10`


## Predict 

first argument is the allennlp model, second is the preprocessed evalutaion file ( path/to/output.jsonl.gz in preprocess), then the dataset name (in order to create the official predictions format) 

 `python predict.py --model https://multiqa.s3.amazonaws.com/models_new/BERTBase/SQuAD1-1.tar.gz  --dataset https://multiqa.s3.amazonaws.com/data/SQuAD1-1_dev.jsonl.gz --dataset_name SQuAD`
 
 To predict only a the first N examples use `--sample_size N`
 
 To add a GPU device simply append: `--cuda_device 0`
 
 By default the output will be saved at results/DATASET_NAME/...  You may also change the output filename and path using `--prediction_filepath path/to/my/output`
 
 
## Multiqa Data Format
see [Readme](https://github.com/alontalmor/multiqa/blob/master/datasets/README.md) in the datasets folder.

## Training using AlleNLP
see [Readme](https://github.com/alontalmor/multiqa/blob/master/models/README.md) in the models folder.

## Other

Allennlp caching infra is used, so make sure to have enough disk space, and control the cache directory using `ALLENNLP_CACHE_ROOT` env variable.





