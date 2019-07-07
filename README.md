# MultiQA

MultiQA is a project whose goal is to facilitate training and evaluating reading
comprehension models over arbitrary sets of datasets.
All datasets are in a single format, and it is accompanied by
an AllenNLP `DatasetReader` and model that enable easy training and evaluation
on multiple subsets of datasets.
 
## Setup

### Setting up a virtual environment

1.  First, clone the repository:

    ```
    git clone -b staging --single-branch https://github.com/alontalmor/MultiQA.git
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
 
 ### HotpotQA Official Evaluation script 
 `python datasets/HotpotQA/eval_script.py datasets/HotpotQA/BERTbase_HotpotQA__on__HotpotQA_dev.json datasets/HotpotQA/hotpot_dev_distractor_v1.json`
 
## Multiqa Data Format
see Readme in the datasets folder.
 



