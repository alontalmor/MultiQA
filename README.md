# MultiQA

MultiQA is an set of datasets currently confined to the task of machine reading comprehenstion, that are preprocessed in a single format.
It is accompanied by an AllenNLP dataset reader and model that enable train and/or evaluation on multiple subsets of the datasets. 




# Data Description
We will have table of many datasets statistics here .. 

## Data Format
Each example in the MultiQA format contains a `context`, that may contain multiple documents with diverse types of text, and a list of one or more question on the context (`qas`).

### Context

Each context contains a list of one or more documents with different possible types of text. A document `title` if such exists, a `snippet` for context produced by search engines. A `text` for various types of body text such abstract, partial or full body text or full html. And finally the `source_utl` of the document. 

```json
"title": "the document title",
"snippet":"a snippet, mostly applicable for search engine results", 
"text":"full context text as provided by the dataset",
"source_url":"",
```

Each document may also contain a metadata field for datasets with annotated field on the context. Such as `sentence_start_bytes`  that serparating the `text` field into sentences for supporting context in datasets such as HotpotQA.

```json
"metadata": {
    "text": {
        "sentence_start_bytes": [0,90,128]
    }
}
```

#### Tokens
Each document and text field is tokenized separatly. The tokens are stored in the `tokens` for every document in the list of documents.

```json
"tokens": {
    "text": [["Scott",0],["Derrickson",6],["(",17],["born",18],["July",23],["16",28] ..],
    "title": [["Scott",0],["Derrickson",6]]
}
```

### Question List `qas`

Each context contains a list of one or more question and answers. 

A natural language `question` text is acompanied by a list of `question_tokens` as well as a set of one or more `answers` for various tasks.  Dataset specific annotations or question properties may be added to `metadata`. 
```json
"qid": "56ddde6b9a695914005b9629#0",
"question": "e.g. Who is president?",
"question_tokens": "e.g. Who is president?",
"metadata": {"...":"..."}
"answers": {"...":"..."}
```

#### Metadata

```json
"metadata":{
    "supporting_context": [{
        "doc_ind": 1,
        "part": "text",
        "start_byte": 0,
        "text": "Scott Derrickson (born July 16, 1966) ..."},
    {
        "doc_ind": 4,
        "part": "text",
        "start_byte": 0,
        "text": "Edward Davis Wood Jr. (October 10, 1924 \u2013 December 10, 1978) was an American filmmaker, ..."}
    ]}
}
```

#### Answers 

Answers may be of multiple types depending on the dataset tasks.

Answers can be of `multi-choice` type in which a set of `choices` is provided, or `open-ended`. The `open-ended` option may contain `cannot_answer` if applicable for the question, and/or `answer_candidates` that are a list of annotations provided by different annotators (e.g. SQuAD, NaturalQuestions and DROP). If a main answer exists it will be the first in the `answer_candidates` list (e.g. DROP). 
Answers can be of `multi-choice` type in which a set of `choices` is provided, or `open-ended`. The `open-ended` option may contain `cannot_answer` if applicable for the question, and/or `answer_candidates` that are a list of annotations provided by different annotators (e.g. SQuAD, NaturalQuestions and DROP. Note that in all these datasets answer should much ONLY ONE of the `answer_candidates`). If a main answer exists it will be the first in the `answer_candidates` list (e.g. DROP).

In the `multi-choice` options a set of `answer_candidate` `choices` is provided, as well as a `correct_answer_index` if only one answer is correct OR `multi_correct_answer_indexes` (e.g. MultiRC)

 ```json
"answers": {
    "open-ended": {
        "cannot_answer": "yes/no",
        "answer_candidates":["answer_candidate1", "answer_candidate2"]
    },
    "multi-choice": {
        "correct_answer_index": 0,
        "multi_correct_answer_indexes":[1,2],
        "choices":["answer_candidate1", "answer_candidate2", "answer_candidate3" ]
    }
}
```


#### answer candidate

Each answer candidate may contain more than one answer type. Each answer type (e.g. `yesno`, `extractive`)  may contain one of [`single_answer`, `list`,`set`] this allows very expressive answers such as:
`Yes, [span1, span2 ... ]` or a sorted set of instructions provided as generated text.

Fields that are not applicable for a certain dataset will not be shown in it's file.

In some datasets such as DROP the `extractive` spans are always provided as a list to indicate that in this dataset the model can always expected to produce a list of spans (as opposed to SQuAD in which only a `single_answer` span is required)

Note that in all observerd datasets, model is required to match only one of the strongly typed fields in the `answer_candidate`


 ```json
"yesno": {
    "single_answer":"yes",
  },
  "generated_text": {
    "list":["sentense0","sentence1"],
  },
  "date": {
    "single_answer": {
        "day": "23",
        "month": "April",
        "year": "1734"
    }
  },
  "number": {
    "set":[0, 7, 8.4],
  },
  "extractive": {
    "single_answer":{
      "answer": "the main answer",
      "aliases": ["Donald Trump","Trump"],
      "instances":[
        {
            "doc_ind":0,
            "part":"abstract",
            "start_byte":15,
            "start_end_tokens":[2,3],
            "text": "Donald Trump",
        },
        {
            "doc_ind":0,
            "part":"abstract",
            "start_byte":15,
            "start_end_tokens":[2,3],
            "text": "Trump",
        }
      ]
    }
  }
```


