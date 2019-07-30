# MultiQA

MultiQA is a project whose goal is to facilitate training and evaluating reading
comprehension models over arbitrary sets of datasets.
All datasets are in a single format, and it is accompanied by
an AllenNLP `DatasetReader` and model that enable easy training and evaluation
on multiple subsets of datasets.


// As part of the paper - put link

// Done at AI2?  

//// This project is constantly being improved, contributions and comments and suggestions are welcome!

 

## Datasets

Link to the single format dataset are provided in the Train, Dev and Test columns.
The BERT-Base column contains evaluation results (EM/F1) as well as a link to the trained model. 
The MultiQA (BERT-Base) column contain the link to the model (in the header) and evalution results for this model. 

| Dataset | Train | Dev | BERT-Base | MultiQA BERT-Base [(model)](https://multiqa.s3.amazonaws.com/models_new/BERTBase/SQuAD1-1_HotpotQA_NewsQA_TriviaQA_unfiltered_SearchQA__full.tar.gz)|
| :----- | :-----:|  :------------------: | :------------------: |  :------------------: |
| SQuAD-1.1 | [data](https://multiqa.s3.amazonaws.com/data/SQuAD1-1_train.jsonl.gz) | [data](https://multiqa.s3.amazonaws.com/data/SQuAD1-1_dev.jsonl.gz) | 80.1 / 87.5 [(model)](https://multiqa.s3.amazonaws.com/models_new/BERTBase/SQuAD1-1.tar.gz) | 81.7 / 88.8 |
| NewsQA | [data](https://multiqa.s3.amazonaws.com/data/NewsQA_train.jsonl.gz) | [data](https://multiqa.s3.amazonaws.com/data/NewsQA_dev.jsonl.gz) | 47.5 / 62.9 [(model)](https://multiqa.s3.amazonaws.com/models_new/BERTBase/NewsQA.tar.gz) | 48.3 / 64.7 |
| HotpotQA | [data](https://multiqa.s3.amazonaws.com/data/HotpotQA_train.jsonl.gz) | [data](https://multiqa.s3.amazonaws.com/data/HotpotQA_dev.jsonl.gz) | 50.1 / 63.2 [(model)](https://multiqa.s3.amazonaws.com/models_new/BERTBase/HotpotQA.tar.gz) | - |
| TriviaQA-unfiltered | [data](https://multiqa.s3.amazonaws.com/data/TriviaQA_unfiltered_train.jsonl.gz) | [data](https://multiqa.s3.amazonaws.com/data/TriviaQA_unfiltered_dev.jsonl.gz) | 59.4 / 65.2 [(model)](https://multiqa.s3.amazonaws.com/models_new/BERTBase/TriviaQA_unfiltered.tar.gz) | 59.0 / 64.7 |
| SearchQA | [data](https://multiqa.s3.amazonaws.com/data/SearchQA_train.jsonl.gz) | [data](https://multiqa.s3.amazonaws.com/data/SearchQA_dev.jsonl.gz) | 58.7 / 65.2 [(model)](https://multiqa.s3.amazonaws.com/models_new/BERTBase/SearchQA.tar.gz) | 58.8 / 65.3 |

 
 
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
    virtualenv venv --python=python3.7
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

## Preprocess
    
   before you run: 
   
  `python build_dataset.py HotpotQA dev path/to/output.jsonl.gz --n_processes 10`


## Predict

first argument is the allennlp model, second is the preprocessed evalutaion file ( path/to/output.jsonl.gz in preprocess), then the dataset name (in order to create the official predictions format)

 `python predict.py https://multiqa.s3.amazonaws.com/models_new/BERTbase/HotpotQA.tar.gz https://multiqa.s3.amazonaws.com/data/HotpotQA_dev.jsonl.gz HotpotQA`
 
 To add a GPU device simply append: `--cuda_device 0`
 
 You may also change the output filename and path using `--prediction_filepath path/to/my/output`
 
 ###  Official Evaluation script 
 An example of running the HotpotQA official evaluation script 
 
 `python datasets/HotpotQA/eval_script.py datasets/HotpotQA/BERTbase_HotpotQA__on__HotpotQA_dev.json datasets/HotpotQA/hotpot_dev_distractor_v1.json`
 
## Multiqa Data Format
see Readme in the datasets folder.





