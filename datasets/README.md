
# Data Description
We will have table of many datasets statistics here .. 

## Header

The first line in a multiqa file is a header, containing various information on dataset at hand.
```json
"header": {
    "data_source": "wikipedia",
    "dataset_name": "SQuAD",
    "dataset_url": "https://rajpurkar.github.io/SQuAD-explorer/",
    "license": "http://creativecommons.org/licenses/by-sa/4.0/legalcode",
    "multiqa_version": "0.00",
    "number_of_contexts": 19035,
    "number_of_qas": 130319,
    "readme": "",
    "schema": {},
    "split": "train",
    "text_type": "abstract",
    "tokenization_source": "multiqa"
}
```

## A single line

After the header, each line in the MultiQA format contains one context + questions. The `context` field may contain one or more documents, and a list of one or more questions about the context (`qas`).

```json
"id": "HotpotQA_5a85ea095542994775f606a8",
"context": {
  "documents":["document1","document2","document3"],
},
"qas":["question1","question2"], 
```

### The `Context`

Each context contains a LIST of one or more documents with different possible types of text. A document `title` if such exists, a `text` for various types of internal document texts such abstract (e.g. HotpotQA), partial or full body text (e.g. TriviaQA) or full html (e.g. NaturalQuestions). And finally the `source_url` of the document. 

```json
"documents":[
    {
        "title": "the document title",
        "text": "abstract / paragraph / full_html",
        "snippet": "search engine snippet for this document",
        "source_url":"http:// ... ",
        "metadata": {},
        "tokens": {}
    }
]
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
Each document and text field is tokenized separately. The tokens are stored in the `tokens` field for each document.

```json
"tokens": {
    "text": [["Scott",0],["Derrickson",6],["(",17],["born",18],["July",23],["16",28]],
    "title": [["Scott",0],["Derrickson",6]]
}
```

### Question List `qas`

Each context contains a list of one or more question and answers. 

A natural language `question` text is accompanied by a list of `question_tokens` as well as a set of one or more `answers` for various tasks.  Dataset specific annotations or question properties may be added to `metadata`. 
```json
"question": {
    "qid": "DROP_q_1e50dd00-e837-4ecc-8265-83365d286aa4",
    "question": "How many years was the Mon kingdom",
    "question_tokens": [["How",0],["many",4],["years",9],["was",15],["the",19],["Mon",23],["kingdom",27]],
    "supporting_context": {},
    "metadata": {},
    "answers": {}
}
```

#### supporting_context

Datasets such as HotpotQA and MultiRC require `supporting_context` to be provided by the model.
  
```json
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
```

#### Answers 

Answers can be of `multi-choice` type for which a set of `choices` is provided, or `open-ended`. The `open-ended` option may contain `cannot_answer` if applicable for the question, and/or `answer_candidates` that are a list of annotations provided by different annotators (e.g. SQuAD, NaturalQuestions and DROP. Note that in all these datasets answer should much ONLY ONE of the `answer_candidates`). If a main answer exists it will be the first in the `answer_candidates` list (e.g. DROP).

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

Each answer candidate may contain more than one answer type. Each answer type (e.g. `yesno`, `extractive`) can be one of the following:
 * `single_answer` : only one answer in needed here.
 * `list` : a list of answers (order is not important)
 * `set`: a set or sorted list of answers in which order is important.
  
 this allows expressive answers such as:
`Yes, [span1, span2 ... ]` or a sorted set of instructions provided as generated text, etc ... 

Fields that are not applicable for a certain datasets will not be shown in it's file.

In some datasets such as DROP the `extractive` spans are always provided as a list to indicate that in this dataset the model is always expected to produce a `list` of spans (as opposed to SQuAD in which only a `single_answer` span is required)

Note that in all observed datasets, a model is required to match only one of the strongly typed fields in the `answer_candidate`


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
            "doc_id":0,
            "part":"text",
            "start_byte":15,
            "start_end_tokens":[2,3],
            "text": "Donald Trump",
        }
      ]
    }
  }
```

Extractive answers contain a main answer, and a list of aliases when applicable (e.g. TriviaQA, ComplexWebQuestions). In addition for each span a list of `instances` is provided pointing the document index (`doc_ind`) and the document `part` from which it was extracted, as well as the `start_byte`, `start_end_tokens` and the span `text`.  `Instances` provided as part of the a dataset will be used, if none are provided instances (or fields of `instances` such as `start_end_tokens` that were not provided) are extracted in pre-processing.


