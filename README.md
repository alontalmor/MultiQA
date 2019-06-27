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
"metadata": {...}
"answers": {...}
```




In this example, you can see that the second long answer candidate is contained
within the first. We do not disallow nested long answer candidates, we just ask
annotators to find the *smallest candidate containing all of the information
required to infer the answer to the question*. However, we do observe that 95%
of all long answers (including all paragraph answers) are not nested below any
other candidates.
Since we believe that some users may want to start by only considering
non-overlapping candidates, we include a boolean flag `top_level` that
identifies whether a candidate is nested below another (`top_level = False`) or
not (`top_level = True`). Please be aware that this flag is only included for
convenience and it is not related to the task definition in any way.
For more information about the distribution of long answer types, please
see the data statistics section below.

### Annotations
The NQ training data has a single annotation with each example and the evaluation
data has five. Each annotation defines a "long_answer" span, a list of
`short_answers`, and a `yes_no_answer`.  If the annotator has marked a long
answer, then the long answer dictionary identifies this long answer using byte
offsets, token offsets, and an index into the list of long answer candidates. If
the annotator has marked that no long answer is available, all of the fields in
the long answer dictionary are set to -1.

```json
"annotations": [{
  "long_answer": { "start_byte": 32, "end_byte": 106, "start_token": 5, "end_token": 22, "candidate_index": 0 },
  "short_answers": [
    {"start_byte": 73, "end_byte": 78, "start_token": 15, "end_token": 16},
    {"start_byte": 87, "end_byte": 92, "start_token": 18, "end_token": 19}
  ],
  "yes_no_answer": "NONE"
}]
```

Each of the short answers is also identified using both byte offsets and token
indices. There is no limit to the number of short answers. There is also often
no short answer, since some questions such as "describe google's founding" do
not have a succinct extractive answer. When this is the case, the long answer is
given but the "short_answers" list is empty.

Finally, if no short answer is given, it is possible that there is a
`yes_no_answer` for questions such as "did larry co-found google". The values
for this field `YES`, or `NO` if a yes/no answer is given. The default value is
`NONE` when no yes/no answer is given. For statistics on long answers, short
answers, and yes/no answers, please see the data statistics section below.

### Data Statistics
The NQ training data contains 307,373 examples. 152,148 have a long answer
and 110,724 have a short answer. Short answers can be sets of spans in the document
(106,926), or yes or no (3,798). Long answers are HTML bounding boxes, and the
distribution of NQ long answer types is as follows:

| HTML tags | Percent of long answers |
|-----------|-------------------------|
| `<P>`     | 72.9%                   |
| `<Table>` | 19.0%                   |
| `<Tr>`    | 1.5%                    |
| `<Ul>`, `<Ol>`, `<Dl>` | 3.2%       |
| `<Li>`, `<Dd>`, `<Dt>` | 3.4%       |

While we allow any paragraph, table, or list element to be a long answer,
we find that 95% of the long answers are not contained by any other
long answer candidate. We mark these `top level` candidates in the data,
as described above.

Short answers may contain more than one span, if the question is asking
for a list of answers (e.g. who made it to stage 3 in american ninja warrior season 9).
However, almost all short answers (90%) only contain a single span of text.
All short answers are contained by the long answer given in the same annotation.

# Prediction Format
Please see the [evaluation script](nq_eval.py) for a description of the prediction
format that your model should output.

# Contact us
If you have a technical question regarding the dataset, code or publication, please
create an issue in this repository. This is the fastest way to reach us.

If you would like to share feedback or report concerns, please email us at <natural-questions@google.com>.

# Dataset Metadata
The following table is necessary for this dataset to be indexed by search
engines such as <a href="https://g.co/datasetsearch">Google Dataset Search</a>.
<div itemscope itemtype="http://schema.org/Dataset">
<table>
  <tr>
    <th>property</th>
    <th>value</th>
  </tr>
  <tr>
    <td>name</td>
    <td><code itemprop="name">Natural Questions</code></td>
  </tr>
  <tr>
    <td>alternateName</td>
    <td><code itemprop="alternateName">natural-questions</code></td>
  </tr>
  <tr>
    <td>url</td>
    <td><code itemprop="url">https://github.com/google-research-datasets/natural-questions</code></td>
  </tr>
  <tr>
    <td>sameAs</td>
    <td><code itemprop="sameAs">https://ai.google.com/research/NaturalQuestions</code></td>
  </tr>
  <tr>
    <td>description</td>
    <td><code itemprop="description">Natural Questions (NQ) contains real user questions issued to Google search, and
answers found from Wikipedia by annotators.\n
NQ is designed for the training and evaluation of automatic question answering systems.\n
\n
NQ contains 307,372 training examples, 7,830 examples for development, and we withold a further 7,842 examples for testing.\n
\n
Each example contains a single question, a tokenized representation of the question, a timestamped Wikipedia URL, and the HTML representation of that Wikipedia page.\n
\n
```json\n
"question_text": "who founded google",\n
"question_tokens": ["who", "founded", "google"],\n
"document_url": "http://www.wikipedia.org/Google",\n
"document_html": "<html><body>Google<p>Google was founded in 1998 by ..."\n
```\n</code></td>
  </tr>
  <tr>
    <td>provider</td>
    <td>
      <div itemscope itemtype="http://schema.org/Organization" itemprop="provider">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">Google</code></td>
          </tr>
          <tr>
            <td>sameAs</td>
            <td><code itemprop="sameAs">https://en.wikipedia.org/wiki/Google</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
  <tr>
    <td>license</td>
    <td>
      <div itemscope itemtype="http://schema.org/CreativeWork" itemprop="license">
        <table>
          <tr>
            <th>property</th>
            <th>value</th>
          </tr>
          <tr>
            <td>name</td>
            <td><code itemprop="name">CC BY-SA 3.0</code></td>
          </tr>
          <tr>
            <td>url</td>
            <td><code itemprop="url">https://creativecommons.org/licenses/by-sa/3.0/</code></td>
          </tr>
        </table>
      </div>
    </td>
  </tr>
  <tr>
    <td>citation</td>
    <td><code itemprop="citation">Kwiatkowski, Tom, Jennimaria Palomaki, Olivia Rhinehart, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein et al. "Natural questions: a benchmark for question answering research." (2019). https://ai.google/research/pubs/pub47761</code></td>
  </tr>
</table>
</div>

