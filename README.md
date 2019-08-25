# MultiQA

MultiQA is a project whose goal is to facilitate training and evaluating reading
comprehension models over arbitrary sets of datasets.
All datasets are in a single format, and it is accompanied by
an AllenNLP `DatasetReader` and `model` that enable easy training and evaluation
on multiple subsets of datasets.

This repository contains the code for our paper [MultiQA: An Empirical Investigation of Generalization and Transfer in Reading Comprehension](https://arxiv.org/abs/1905.13453).

This work was performed at The Allen Institute of Artificial Intelligence.   

This project is constantly being improved. Contributions, comments and suggestions are welcome!

## News

| Date | Message
| :----- | :-----
| Aug 24, 2019 | New! convert multiqa format to SQuAD2.0 format + Pytorch-Transformers models support 
| Aug 14, 2019 | BoolQ and ComplexWebQuestions data added. 
| Aug 12, 2019 | multiqa.py added enabling easy multiple dataset training and evaluation. 
| Aug 07, 2019 | TriviaQA-Wikipedia BERT-Base Model is now available, improved results will be soon to follow. 
| Aug 03, 2019 | BERT-Large Models are now available! 


## Datasets

Link to the single format dataset are provided in the Train, Dev and Test columns.
The BERT-Base column contains evaluation results (EM/F1) as well as a link to the trained model. 
The MultiQA-5Base column contain the link to the model (in the header) and evalution results for this model. This model is BERT-Base that has been trained on 5 datasets. 

| Dataset | Train | Dev | BERT-Base | MultiQA-5Base [(model)](https://multiqa.s3.amazonaws.com/models/BERTBase/SQuAD1-1_HotpotQA_NewsQA_TriviaQA_unfiltered_SearchQA__full.tar.gz) | BERT-Large | 
| :----- | :-----:|  :------------------: | :------------------: | :------------------: |  :------------------: | 
| SQuAD-1.1 | [data](https://multiqa.s3.amazonaws.com/data/SQuAD1-1_train.jsonl.gz) | [data](https://multiqa.s3.amazonaws.com/data/SQuAD1-1_dev.jsonl.gz) | 80.1 / 87.5 [(model)](https://multiqa.s3.amazonaws.com/models/BERTBase/SQuAD1-1.tar.gz) | 81.7 / 88.8 | 83.3 / 90.3 [(model)](https://multiqa.s3.amazonaws.com/models/BERTLarge/SQuAD1-1.tar.gz)  | 
| NewsQA | [data](https://multiqa.s3.amazonaws.com/data/NewsQA_train.jsonl.gz) | [data](https://multiqa.s3.amazonaws.com/data/NewsQA_dev.jsonl.gz) | 47.5 / 62.9 [(model)](https://multiqa.s3.amazonaws.com/models/BERTBase/NewsQA.tar.gz) | 48.3 / 64.7 | 50.3 / 66.0 [(model)](https://multiqa.s3.amazonaws.com/models/BERTLarge/NewsQA.tar.gz)  |  
| HotpotQA | [data](https://multiqa.s3.amazonaws.com/data/HotpotQA_train.jsonl.gz) | [data](https://multiqa.s3.amazonaws.com/data/HotpotQA_dev.jsonl.gz) | 50.1 / 63.2 [(model)](https://multiqa.s3.amazonaws.com/models/BERTBase/HotpotQA.tar.gz) | - | 54.0 / 67.0 [(model)](https://multiqa.s3.amazonaws.com/models/BERTLarge/HotpotQA.tar.gz) |  
| TriviaQA-unfiltered | [data](https://multiqa.s3.amazonaws.com/data/TriviaQA_unfiltered_train.jsonl.gz) | [data](https://multiqa.s3.amazonaws.com/data/TriviaQA_unfiltered_dev.jsonl.gz) | 59.4 / 65.2 [(model)](https://multiqa.s3.amazonaws.com/models/BERTBase/TriviaQA_unfiltered.tar.gz) | 59.0 / 64.7 | 60.7 / 66.5 [(model)](https://multiqa.s3.amazonaws.com/models/BERTLarge/TriviaQA_unfiltered.tar.gz)  |  
| TriviaQA-wiki | [data](https://multiqa.s3.amazonaws.com/data/TriviaQA_wiki_train.jsonl.gz) | [data](https://multiqa.s3.amazonaws.com/data/TriviaQA_wiki_dev.jsonl.gz) | 57.5 / 62.3 [(model)](https://multiqa.s3.amazonaws.com/models/BERTBase/TriviaQA_wiki.tar.gz) | -  | -  |  
| SearchQA | [data](https://multiqa.s3.amazonaws.com/data/SearchQA_train.jsonl.gz) | [data](https://multiqa.s3.amazonaws.com/data/SearchQA_dev.jsonl.gz) | 58.7 / 65.2 [(model)](https://multiqa.s3.amazonaws.com/models/BERTBase/SearchQA.tar.gz) | 58.8 / 65.3 | 60.5 / 67.3 [(model)](https://multiqa.s3.amazonaws.com/models/BERTLarge/SearchQA.tar.gz)  |  
| BoolQ | [data](https://multiqa.s3.amazonaws.com/data/BoolQ_jsonl.gz) | [data](https://multiqa.s3.amazonaws.com/data/BoolQ_dev.jsonl.gz) | |  |   |  
| ComplexWebQuestions | [data](https://multiqa.s3.amazonaws.com/data/ComplexWebQuestions_jsonl.gz) | [data](https://multiqa.s3.amazonaws.com/data/ComplexWebQuestions_dev.jsonl.gz) | |  |   |  
| Natural Questions | Coming soon | Coming soon | Coming soon | Coming soon | Coming soon |  

Datasets will be addeed weekly, so please stay tuned!

### multiqa commands

In order to simply train BERT on multiple datasets please use:
```
python multiqa.py train --datasets SQuAD1-1,NewsQA,SearchQA --cuda_device 0,1,2,3
python multiqa.py evaluate --model SQuAD1-1 --datasets SQuAD1-1,NewsQA,SearchQA --cuda_device 0
```
By default the output will be stored in models/datatset1_dataset2_... to change this please change  `--serialization_dir`

Type `python multiqa.py` for additional options.

Note, this version uses the default multiqa format datasets stored in s3, to use your own dataset please see [Readme](https://github.com/alontalmor/multiqa/blob/master/models/README.md) for using allennlp core commands.
 
### MultiQA format to SQuAD2.0 format
If you prefer using SQuAD2.0 format, or run the Pytorch-Trasformers models, please use:
```
python convert_multiqa_to_squad_format.py --datasets https://multiqa.s3.amazonaws.com/data/HotpotQA_dev.jsonl.gz --output_file data/squad_format/HotpotQA_dev.json
```

 
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
    virtualenv venv --python=python3.7 (or python3.7 -m venv venv or conda create -n multiqa python=3.7)
    ```

4.  Activate the virtual environment. You will need to activate the venv environment in each terminal in which you want to use WebAsKB.

    ```
    source venv/bin/activate (or source venv/bin/activate.csh or conda activate multiqa)
    ```
5.  Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```


### Data

The allennlp caching infra is used, so be sure to have enough disk space, and control the cache directory using ALLENNLP_CACHE_ROOT env variable.

## Build Dataset
    
   This will take a dataset from it's original URL and output the same dataset in the MultiQA format.  
   
  `python build_dataset.py --dataset_name HotpotQA --split train --output_file path/to/output.jsonl.gz --n_processes 10`


## Predict 

first argument is the allennlp model, second is the preprocessed evalutaion file ( path/to/output.jsonl.gz in preprocess), then the dataset name (in order to create the official predictions format) 

 `python predict.py --model https://multiqa.s3.amazonaws.com/models/BERTBase/SQuAD1-1.tar.gz  --dataset https://multiqa.s3.amazonaws.com/data/SQuAD1-1_dev.jsonl.gz --dataset_name SQuAD`
 
 To predict only a the first N examples use `--sample_size N`
 
 To add a GPU device simply append: `--cuda_device 0`
 
 By default the output will be saved at results/DATASET_NAME/...  You may also change the output filename and path using `--prediction_filepath path/to/my/output`
 
 
## Multiqa Data Format
see [Readme](https://github.com/alontalmor/multiqa/blob/master/datasets/README.md) in the datasets folder.

## Training using AlleNLP
see [Readme](https://github.com/alontalmor/multiqa/blob/master/models/README.md) in the models folder.

## Training using Pytorch-Trasformers
see [Readme](https://github.com/alontalmor/multiqa/blob/master/models/README.md) in the models folder.

## Other

Allennlp caching infra is used, so make sure to have enough disk space, and control the cache directory using `ALLENNLP_CACHE_ROOT` env variable.





