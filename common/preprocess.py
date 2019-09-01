import argparse
import json
import logging
from typing import Any, Dict, List, Tuple
import zipfile,re, copy, random, math
import sys, os
import boto3
from typing import TypeVar,Iterable
from multiprocessing import Pool



T = TypeVar('T')

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(os.path.join(__file__, os.pardir), os.pardir))))

from allennlp.common.tqdm import Tqdm
from allennlp.common.file_utils import cached_path
from allennlp.common.util import add_noise_to_dict_values

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.reading_comprehension import util
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from nltk.corpus import stopwords
import string

def split(lst: List[T], n_groups) -> List[List[T]]:
    """ partition `lst` into `n_groups` that are as evenly sized as possible  """
    per_group = len(lst) // n_groups
    remainder = len(lst) % n_groups
    groups = []
    ix = 0
    for _ in range(n_groups):
        group_size = per_group
        if remainder > 0:
            remainder -= 1
            group_size += 1
        groups.append(lst[ix:ix + group_size])
        ix += group_size
    return groups

def flatten_iterable(listoflists: Iterable[Iterable[T]]) -> List[T]:
    return [item for sublist in listoflists for item in sublist]

def group(lst: List[T], max_group_size) -> List[List[T]]:
    """ partition `lst` into that the mininal number of groups that as evenly sized
    as possible  and are at most `max_group_size` in size """
    if max_group_size is None:
        return [lst]
    n_groups = (len(lst)+max_group_size-1) // max_group_size
    per_group = len(lst) // n_groups
    remainder = len(lst) % n_groups
    groups = []
    ix = 0
    for _ in range(n_groups):
        group_size = per_group
        if remainder > 0:
            remainder -= 1
            group_size += 1
        groups.append(lst[ix:ix + group_size])
        ix += group_size
    return groups

class SpaceTokenizer():
    def __init__(self):
        pass

    def is_whitespace(self, c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def tokenize(self, text):
        doc_tokens = []
        start_offsets = []
        prev_is_whitespace = True
        for i, c in enumerate(text):
            if self.is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                    start_offsets.append(i)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
        return [Token(t,s) for t,s in zip(doc_tokens,start_offsets)]


class MultiQAPreProcess:

    def __init__(self,n_processes):
        self._n_processes = n_processes
        self._tokenizer = WordTokenizer()

        # We also support a simple space tokenizer, although we observe that WordTokenizer is much better.
        #self._tokenizer = SpaceTokenizer()

        self._STRIP_CHARS = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~‘’´`_'

        self._context_parts = ['title','snippet','text']
        self._context_seps = [' [TLE] ', ' [DOC]', ' [DOC] ']

    def char_span_to_token_span(self, instance, tokens):
        """
        Converts a character span from a passage into the corresponding token span in the tokenized
        version of the passage.  If you pass in a character span that does not correspond to complete
        tokens in the tokenized version, we'll do our best, but the behavior is officially undefined.
        We return an error flag in this case, and have some debug logging so you can figure out the
        cause of this issue (in SQuAD, these are mostly either tokenization problems or annotation
        problems; there's a fair amount of both).

        The basic outline of this method is to find the token span that has the same offsets as the
        input character span.  If the tokenizer tokenized the passage correctly and has matching
        offsets, this is easy.  We try to be a little smart about cases where they don't match exactly,
        but mostly just find the closest thing we can.

        The returned ``(begin, end)`` indices are `inclusive` for both ``begin`` and ``end``.
        So, for example, ``(2, 2)`` is the one word span beginning at token index 2, ``(3, 4)`` is the
        two-word span beginning at token index 3, and so on.

        Returns
        -------
        token_span : ``Tuple[int, int]``
            `Inclusive` span start and end token indices that match as closely as possible to the input
            character spans.
        error : ``bool``
            Whether the token spans match the input character spans exactly.  If this is ``False``, it
            means there was an error in either the tokenization or the annotated character span.
        """
        # We have token offsets into the passage from the tokenizer; we _should_ be able to just find
        # the tokens that have the same offsets as our span.

        token_offsets = [(token[1], token[1] + len(token[0])) for token in tokens]
        character_span = (instance['start_byte'],instance['start_byte'] + len(instance['text']))

        error = False
        start_index = 0
        while start_index < len(token_offsets) and token_offsets[start_index][0] < character_span[0]:
            start_index += 1
        # start_index should now be pointing at the span start index.
        if token_offsets[start_index][0] > character_span[0]:
            # In this case, a tokenization or labeling issue made us go too far - the character span
            # we're looking for actually starts in the previous token.  We'll back up one.
            logger.debug("Bad labelling or tokenization - start offset doesn't match")
            start_index -= 1
        if token_offsets[start_index][0] != character_span[0]:
            error = True
        end_index = start_index
        while end_index < len(token_offsets) and token_offsets[end_index][1] < character_span[1]:
            end_index += 1
        if end_index == start_index and token_offsets[end_index][1] > character_span[1]:
            # Looks like there was a token that should have been split, like "1854-1855", where the
            # answer is "1854".  We can't do much in this case, except keep the answer as the whole
            # token.
            logger.debug("Bad tokenization - end offset doesn't match")
        elif token_offsets[end_index][1] > character_span[1]:
            # This is a case where the given answer span is more than one token, and the last token is
            # cut off for some reason, like "split with Luckett and Rober", when the original passage
            # said "split with Luckett and Roberson".  In this case, we'll just keep the end index
            # where it is, and assume the intent was to mark the whole token.
            logger.debug("Bad labelling or tokenization - end offset doesn't match")
        if token_offsets[end_index][1] != character_span[1]:
            error = True

        # inclusive start_index and inclusive end_index
        # Only add answers where the text can be exactly recovered from the
        # document. If this CAN'T happen it's likely due to weird Unicode
        # stuff so we will just skip the example.
        #
        # Note that this means for training mode, every example is NOT
        # guaranteed to be preserved.
        #actual_text = " ".join([t[0] for t in tokens[start_index:(end_index + 1)]])
        #cleaned_answer_text = " ".join([t.text for t in self._tokenizer.tokenize(instance['text'])])
        #if actual_text.find(cleaned_answer_text) == -1:
        #    logger.warning("Could not find answer: '%s' vs. '%s'",
        #                   actual_text, cleaned_answer_text)
        #    assert()
        #else:
        instance['token_inds'] = (start_index, end_index)

    def find_all_answer_spans(self, answer, context):
        """Find all exact matches of `answer` in `context`.
        - Matches are case, article, and punctuation insensitive.
        - Matching follows SQuAD eval protocol.
        - The context and answer are assumed to be tokenized
          (using either tokens or word pieces).
        Returns [start, end] (inclusive) token span.
        """
        # Lower-case and strip all tokens.

        # tokenizing answer
        tokenized_answer = self._tokenizer.tokenize(answer)

        words = [t[0].lower().strip(self._STRIP_CHARS) for t in context]
        answer = [t[0].lower().strip(self._STRIP_CHARS) for t in tokenized_answer]

        # Strip answer empty tokens + articles
        answer = [t for t in answer if t not in {'', 'a', 'an', 'the'}]
        if len(answer) == 0:
            return []

        # Find all possible starts (matches first answer token).
        occurences = []
        word_starts = [i for i, w in enumerate(words) if answer[0] == w]
        n_tokens = len(answer)

        # Advance forward until we find all the words, skipping over articles
        for start in word_starts:
            end = start + 1
            ans_token = 1
            while ans_token < n_tokens and end < len(words):
                next = words[end]
                if answer[ans_token] == next:
                    ans_token += 1
                    end += 1
                elif next in {'', 'a', 'an', 'the'}:
                    end += 1
                else:
                    break
            if n_tokens == ans_token:
                # inclusive start_index and inclusive end_index
                occurences.append((start, end - 1))

        return list(set(occurences))

    def tokenize_context(self, document):
        if 'tokens' not in document:
            document['tokens'] = {}

        for part, SEP in zip(self._context_parts , self._context_seps):
            if part in document and part not in document['tokens']:
                part_tokens = self._tokenizer.tokenize(document[part])
                # seems Spacy class is pretty heavy in memory, lets move to a simple representation for now..
                document['tokens'][part] = [(t.text, t.idx) for t in part_tokens]

    def preprocess_context(self, context, search_answer_within_supp_context):



        # tokenizing contexts:
        for document in context['context']['documents']:
            self.tokenize_context(document)

        # find answers and find answer tokens start/end
        for qa in context['qas']:
            # tokenizing question
            if 'question_tokens' not in qa:
                qa['question_tokens'] = [(t.text, t.idx) for t in self._tokenizer.tokenize(qa['question'])]

            answer_cand_list = []
            if 'open-ended' in qa['answers']:
                if 'cannot_answer' in qa['answers']['open-ended'] and qa['answers']['open-ended']['cannot_answer'] == 'yes':
                    pass
                else:
                    for answer_cand in qa['answers']['open-ended']['annotators_answer_candidates']:
                        if 'extractive' in answer_cand:
                            if "single_answer" in answer_cand['extractive']:
                                answer_cand_list.append(answer_cand['extractive']['single_answer'])
                            if "list" in answer_cand['extractive']:
                                answer_cand_list += [answer for answer in answer_cand['extractive']['list']]

            elif 'multi-choice' in qa['answers']:
                for answer_cand in qa['answers']['multi-choice']['choices']:
                    if 'extractive' in answer_cand:
                        if "single_answer" in answer_cand['extractive']:
                            answer_cand_list.append(answer_cand['extractive']['single_answer'])
                        if "list" in answer_cand['extractive']:
                            answer_cand_list += [answer for answer in answer_cand['extractive']['list']]


            for single_item in answer_cand_list:
                if 'instances' in single_item:
                    for instance in single_item['instances']:
                        try:
                            self.char_span_to_token_span(instance, document['tokens'][instance['part']])
                        except:
                            single_item['instances'].remove(instance)
                            print('error in char_span_to_token_span, remove instance')
                else:
                    single_item['instances'] = []
                    aliases = [single_item['answer']]
                    if 'aliases' in single_item:
                        aliases += single_item['aliases']
                    for alias in aliases:
                        for doc_id, document in enumerate(context['context']['documents']):
                            for part in document['tokens'].keys():
                                occurences = self.find_all_answer_spans(alias, document['tokens'][part])

                                for occurence in occurences:
                                    start_byte = document['tokens'][part][occurence[0]][1]
                                    if search_answer_within_supp_context:
                                        keep_occurance = False
                                        for supp_context in qa['supporting_context']:
                                            if supp_context['doc_id'] == doc_id and supp_context['part'] == part and \
                                                start_byte >= supp_context['start_byte'] and \
                                                start_byte <= supp_context['start_byte'] + len(supp_context['text']):
                                                    keep_occurance = True
                                    else:
                                        keep_occurance = True

                                    if keep_occurance:
                                        instance = {
                                             'doc_id': doc_id,
                                             'part': part,
                                             'start_byte': start_byte,
                                             'text': alias,
                                             'token_inds':occurence}
                                        single_item['instances'].append(instance)

    def preprocess_multiple_contexts(self, contexts, search_answer_within_supp_context):
        for context in contexts:
            self.preprocess_context(context, search_answer_within_supp_context)
        return contexts

    def _preprocess_t(self, arg):
        return self.preprocess_multiple_contexts(*arg[0:2])

    def tokenize_and_detect_answers(self, contexts, shuffle=True, search_answer_within_supp_context=False):
        if shuffle:
            random.seed(0)
            random.shuffle(contexts)

        if self._n_processes == 1:
            for context in Tqdm.tqdm(contexts, ncols=80):
                self.preprocess_context(context, search_answer_within_supp_context)
        else:
            # multi process (creates chunks of 200 contexts each )
            preprocessed_instances = []
            with Pool(self._n_processes) as pool:
                chunks = split(contexts, self._n_processes)
                chunks = flatten_iterable(group(c, 200) for c in chunks)
                pbar = Tqdm.tqdm(total=len(chunks), ncols=80, smoothing=0.0)
                for preproc_inst in pool.imap_unordered(self._preprocess_t,[[c, search_answer_within_supp_context] for c in chunks]):
                    preprocessed_instances += preproc_inst
                    pbar.update(1)
                pbar.close()
            contexts = preprocessed_instances

        return contexts








