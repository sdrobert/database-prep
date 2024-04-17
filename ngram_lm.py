#!/usr/bin/env python

# Caching is based off the Python module `shelve`; DbCountDict.update on Counter.update.
# Both are PSF-licensed.
#
#   https://docs.python.org/3/license.html

# Copyright 2023 Sean Robertson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
from typing import (
    Generator,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    TypeVar,
)
import warnings
import sys
import re
import struct
import logging
import gzip

from collections import OrderedDict, Counter
from collections.abc import Iterable, MutableMapping, Mapping
from itertools import product
from typing_extensions import Final
from tempfile import TemporaryDirectory
from pathlib import Path
from io import TextIOWrapper

import numpy as np

try:
    from diskcache import Cache
except:
    Cache = None

try:
    from sortedcontainers import SortedList as sorted
except ImportError:
    pass

__all__ = [
    "BackoffNGramLM",
    "main",
    "count_dicts_to_prob_list_absolute_discounting",
    "count_dicts_to_prob_list_add_k",
    "count_dicts_to_prob_list_katz_backoff",
    "count_dicts_to_prob_list_kneser_ney",
    "count_dicts_to_prob_list_mle",
    "count_dicts_to_prob_list_simple_good_turing",
    "sents_to_count_dicts",
    "text_to_sents",
    "write_arpa",
    "DbCountDict",
    "open_count_dict",
]

DEFT_SENT_END_EXPR: Final[re.Pattern] = re.compile(r"[.?!]+")
DEFT_WORD_DELIM_EXPR: Final[re.Pattern] = re.compile(r"\W+")
DEFT_EPS_LPROB: Final[float] = -99.999
DEFT_ADD_K_K: Final[float] = 0.5
DEFAULT_KATZ_THRESH: Final[int] = 7
COUNTFILE_FMT_PREFIX: Final[str] = "counts.{order}"
LPROBFILE_FMT_PREFIX: Final[str] = "lprobs.{order}"
COMPLETE_SUFFIX: Final[str] = ".complete"
DEFT_MAX_CACHE_LEN: Final[int] = 1_000_000
DEFT_MAX_CACHE_AGE_SECONDS: Final[float] = 10.0
SENTS_PER_INFO: Final[int] = 1_000

FALLBACK_DELTAS: Final[Tuple[float, float, float]] = (0.5, 1.0, 1.5)

Ngram = Union[str, Tuple[str, ...]]
CountDict = MutableMapping[Ngram, int]
CountDicts = List[CountDict]
LProb = Union[float, Tuple[float, float]]
ProbDict = MutableMapping[Ngram, LProb]
ProbDicts = List[ProbDict]

warnings.simplefilter("error", RuntimeWarning)


class BackoffNGramLM(object):
    """A backoff NGram language model, stored as a trie

    This class is intended for two things: one, to prune backoff language models, and
    two, to calculate the perplexity of a language model on a corpus. It is very
    inefficient.

    Parameters
    ----------
    prob_dicts
        See :mod:`pydrobert.torch.util.parse_arpa_lm`
    sos
        The start-of-sequence symbol. When calculating the probability of a
        sequence, :math:`P(sos) = 1` when `sos` starts the sequence. Defaults
        to ``'<S>'`` if that symbol is in the vocabulary, otherwise
        ``'<s>'``
    eos
        The end-of-sequence symbol. This symbol is expected to terminate each
        sequence when calculating sequence or corpus perplexity. Defaults to
        ``</S>`` if that symbol is in the vocabulary, otherwise ``</s>``
    unk
        The out-of-vocabulary symbol. If a unigram probability does not exist
        for a token, the token is replaced with this symbol. Defaults to
        ``'<UNK>'`` if that symbol is in the vocabulary, otherwise
        ``'<unk>'``
    destructive
    """

    def __init__(
        self,
        prob_dicts: ProbDicts,
        sos: Optional[str] = None,
        eos: Optional[str] = None,
        unk: Optional[str] = None,
        destructive: bool = False,
    ):
        self.trie = self.TrieNode(0.0, 0.0)
        self.vocab = set()
        max_order = len(prob_dicts)
        if not max_order or not len(prob_dicts[0]):
            raise ValueError("prob_dicts must contain (all) unigrams")
        for order, prob_dict in enumerate(prob_dicts):
            is_first = not order
            is_last = order == max_order - 1
            if not destructive:
                prob_dict = prob_dict.copy()
            while prob_dict:
                context, value = prob_dict.popitem()
                if is_first:
                    self.vocab.add(context)
                    context = (context,)
                if is_last:
                    lprob, bo = value, 0.0
                else:
                    lprob, bo = value
                self.trie.add_child(context, lprob, bo)
        if sos is None:
            if "<S>" in self.vocab:
                sos = "<S>"
            elif "<s>" in self.vocab:
                sos = "<s>"
            else:
                warnings.warn(
                    "start-of-sequence symbol could not be found. Will not include "
                    "in computations"
                )
        elif sos not in self.vocab:
            raise ValueError(
                f"start-of-sequence symbol '{sos}' does not have unigram entry "
            )
        self.sos = self.trie.sos = sos
        if eos is None:
            if "</S>" in self.vocab:
                eos = "</S>"
            elif "</s>" in self.vocab:
                eos = "</s>"
            else:
                warnings.warn(
                    "end-of-sequence symbol could not be found. Will not include "
                    "in computations"
                )
        elif eos not in self.vocab:
            raise ValueError(
                f"end-of-sequence symbol '{eos}' does not have a unigram entry"
            )
        self.eos = self.trie.eos = eos
        if unk is None:
            if "<UNK>" in self.vocab:
                unk = "<UNK>"
            elif "<unk>" in self.vocab:
                unk = "<unk>"
            else:
                warnings.warn(
                    "out-of-vocabulary symbol could not be found. OOVs will raise an "
                    "error"
                )
        elif unk not in self.vocab:
            raise ValueError(
                f"out-of-vocabulary symbol '{unk}' does not have a unigram entry"
            )
        self.unk = unk
        assert self.trie.depth == len(prob_dicts)

    class TrieNode(object):
        def __init__(self, lprob, bo):
            self.lprob = lprob
            self.bo = bo
            self.children = OrderedDict()
            self.depth = 0
            self.sos = None
            self.eos = None

        def add_child(self, context, lprob, bo):
            assert len(context)
            next_, rest = context[0], context[1:]
            child = self.children.setdefault(next_, type(self)(None, 0.0))
            if rest:
                child.add_child(rest, lprob, bo)
            else:
                child.lprob = lprob
                child.bo = bo
            self.depth = max(self.depth, child.depth + 1)

        def conditional(self, context):
            assert context and self.depth
            context = context[-self.depth :]
            cond = 0.0
            while True:
                assert len(context)
                cur_node = self
                idx = 0
                while idx < len(context):
                    token = context[idx]
                    next_node = cur_node.children.get(token, None)
                    if next_node is None:
                        if idx == len(context) - 1:
                            cond += cur_node.bo
                        break
                    else:
                        cur_node = next_node
                    idx += 1
                if idx == len(context):
                    return cond + cur_node.lprob
                assert len(context) > 1  # all unigrams should exist
                context = context[1:]
            # should never get here

        def log_prob(self, context, _srilm_hacks=False):
            joint = 0.0
            for prefix in range(2 if context[0] == self.sos else 1, len(context) + 1):
                joint += self.conditional(context[:prefix])
            if _srilm_hacks and context[0] == self.sos:
                # this is a really silly thing that SRI does - it estimates
                # the initial SOS probability with an EOS probability. Why?
                # The unigram probability of an SOS is 0. However, we assume
                # the sentence-initial SOS exists prior to the generation task,
                # and isn't a "real" part of the vocabulary
                joint += self.conditional((self.eos,))
            return joint

        def _gather_nodes_by_depth(self, order):
            nodes = [(tuple(), self)]
            nodes_by_depth = []
            for _ in range(order):
                last, nodes = nodes, []
                nodes_by_depth.append(nodes)
                for ctx, parent in last:
                    nodes.extend((ctx + (k,), v) for (k, v) in parent.children.items())
            return nodes_by_depth

        def _gather_nodes_at_depth(self, order):
            nodes = [(tuple(), self)]
            for _ in range(order):
                last, nodes = nodes, []
                for ctx, parent in last:
                    nodes.extend((ctx + (k,), v) for (k, v) in parent.children.items())
            return nodes

        def _renormalize_backoffs_for_order(self, order):
            nodes = self._gather_nodes_at_depth(order)
            base_10 = np.log(10)
            for h, node in nodes:
                if not len(node.children):
                    node.bo = 0.0
                    continue
                num = 0.0
                denom = 0.0
                for w, child in node.children.items():
                    assert child.lprob is not None
                    num -= 10.0**child.lprob
                    denom -= 10.0 ** self.conditional(h[1:] + (w,))
                # these values may be ridiculously close to 1, but still valid.
                if num < -1.0:
                    raise ValueError(
                        "Too much probability mass {} on children of n-gram {}"
                        "".format(-num, h)
                    )
                elif denom <= -1.0:
                    # We'll never back off. By convention, this is 0. (Pr(1.))
                    new_bo = 0.0
                elif num == -1.0:
                    if node.bo > -10:
                        warnings.warn(
                            "Found a non-negligible backoff {} for n-gram {} "
                            "when no backoff mass should exist".format(node.bo, h)
                        )
                    continue
                else:
                    new_bo = (np.log1p(num) - np.log1p(denom)) / base_10
                node.bo = new_bo

        def recalculate_depth(self):
            max_depth = 0
            stack = [(max_depth, self)]
            while stack:
                depth, node = stack.pop()
                max_depth = max(max_depth, depth)
                stack.extend((depth + 1, c) for c in node.children.values())
            self.depth = max_depth

        def renormalize_backoffs(self):
            for order in range(1, self.depth):  # final order has no backoffs
                self._renormalize_backoffs_for_order(order)

        def relative_entropy_pruning(self, threshold, eps=1e-8, _srilm_hacks=False):
            nodes_by_depth = self._gather_nodes_by_depth(self.depth - 1)
            base_10 = np.log(10)
            while nodes_by_depth:
                nodes = nodes_by_depth.pop()  # highest order first
                for h, node in nodes:
                    num = 0.0
                    denom = 0.0
                    logP_w_given_hprimes = []  # log P(w | h')
                    P_h = 10 ** self.log_prob(h, _srilm_hacks=_srilm_hacks)
                    for w, child in node.children.items():
                        assert child.lprob is not None
                        num -= 10.0**child.lprob
                        logP_w_given_hprime = self.conditional(h[1:] + (w,))
                        logP_w_given_hprimes.append(logP_w_given_hprime)
                        denom -= 10.0**logP_w_given_hprime
                    if num + 1 < eps or denom + 1 < eps:
                        warnings.warn(
                            "Malformed backoff weight for context {}. Leaving "
                            "as is".format(h)
                        )
                        continue
                    # alpha = (1 + num) / (1 + denom)
                    log_alpha = (np.log1p(num) - np.log1p(denom)) / base_10
                    if abs(log_alpha - node.bo) > 1e-2:
                        warnings.warn(
                            "Calculated backoff ({}) differs from stored "
                            "backoff ({}) for context {}"
                            "".format(log_alpha, node.bo, h)
                        )
                    if _srilm_hacks:
                        # technically these should match when well-formed, but
                        # re-calculating alpha allows us to re-normalize an ill-formed
                        # language model
                        log_alpha = node.bo
                    for idx, w in enumerate(tuple(node.children)):
                        child = node.children[w]
                        if child.bo:
                            continue  # don't prune children with backoffs
                        logP_w_given_h = child.lprob
                        P_w_given_h = 10**logP_w_given_h
                        logP_w_given_hprime = logP_w_given_hprimes[idx]
                        P_w_given_hprime = 10**logP_w_given_hprime
                        new_num = num + P_w_given_h
                        new_denom = denom + P_w_given_hprime
                        log_alphaprime = np.log1p(new_num)
                        log_alphaprime -= np.log1p(new_denom)
                        log_alphaprime /= base_10
                        log_delta_prob = logP_w_given_hprime + log_alphaprime
                        log_delta_prob -= logP_w_given_h
                        KL = -P_h * (
                            P_w_given_h * log_delta_prob
                            + (log_alphaprime - log_alpha) * (1.0 + num)
                        )
                        delta_perplexity = 10.0**KL - 1
                        if delta_perplexity < threshold:
                            node.children.pop(w)
                    # we don't have to set backoff properly (we'll renormalize at end).
                    # We just have to signal whether we can be pruned to our parents (do
                    # *we* have children?)
                    node.bo = float("nan") if len(node.children) else None
            # recalculate depth in case it's changed
            self.depth = -1
            cur_nodes = (self,)
            while cur_nodes:
                self.depth += 1
                next_nodes = []
                for parent in cur_nodes:
                    next_nodes.extend(parent.children.values())
                cur_nodes = next_nodes
            assert self.depth >= 1
            self.renormalize_backoffs()

        def to_prob_list(self):
            nodes_by_depth = self._gather_nodes_by_depth(self.depth)
            prob_list = []
            for order, nodes in enumerate(nodes_by_depth):
                is_first = not order
                is_last = order == self.depth - 1
                dict_ = dict()
                for context, node in nodes:
                    if is_first:
                        context = context[0]
                    if is_last:
                        assert not node.bo
                        value = node.lprob
                    else:
                        value = (node.lprob, node.bo)
                    dict_[context] = value
                prob_list.append(dict_)
            return prob_list

        def prune_by_threshold(self, lprob):
            for order in range(self.depth - 1, 0, -1):
                for _, parent in self._gather_nodes_at_depth(order):
                    for w in set(parent.children):
                        child = parent.children[w]
                        if not child.children and child.lprob <= lprob:
                            del parent.children[w]
            self.renormalize_backoffs()
            self.recalculate_depth()

        def prune_by_name(self, to_prune, eps_lprob):
            to_prune = set(to_prune)
            # we'll prune by threshold in a second pass, so no need to worry about
            # parent-child stuff
            extra_mass = -float("inf")
            remainder = set()
            stack = [((w,), c) for w, c in self.children.items()]
            while stack:
                ctx, node = stack.pop()
                stack.extend((ctx + (w,), c) for w, c in node.children.items())
                if len(ctx) == 1:
                    ctx = ctx[0]
                    if ctx in to_prune:
                        extra_mass = _log10sumexp(extra_mass, node.lprob)
                        node.lprob = eps_lprob
                    elif node.lprob > eps_lprob:
                        remainder.add(ctx)
                elif ctx in to_prune:
                    node.lprob = eps_lprob
            # we never *actually* remove unigrams - we set their probablities to roughly
            # zero and redistribute the collected mass across the remainder
            if not remainder:
                raise ValueError("No unigrams are left unpruned!")
            extra_mass -= np.log10(len(remainder))
            for w in remainder:
                child = self.children[w]
                child.lprob = _log10sumexp(child.lprob, extra_mass)
            self.prune_by_threshold(eps_lprob)

    def conditional(self, context):
        r"""Return the log probability of the last word in the context

        `context` is a non-empty sequence of tokens ``[w_1, w_2, ..., w_N]``. This
        method determines

        .. math::

            \log Pr(w_N | w_{N-1}, w_{N-2}, ... w_{N-C})

        Where ``C`` is this model's maximum n-gram size. If an exact entry cannot be
        found, the model backs off to a shorter context.

        Parameters
        ----------
        context : sequence

        Returns
        -------
        cond : float or :obj:`None`
        """
        if self.unk is None:
            context = tuple(context)
        else:
            context = tuple(t if t in self.vocab else self.unk for t in context)
        if not len(context):
            raise ValueError("context must have at least one token")
        return self.trie.conditional(context)

    def log_prob(self, context):
        r"""Return the log probability of the whole context

        `context` is a non-empty sequence of tokens ``[w_1, w_2, ..., w_N]``. This
        method determines

        .. math::

            \log Pr(w_1, w_2, ..., w_{N})

        Which it decomposes according to the markov assumption (see :func:`conditional`)

        Parameters
        ----------
        context : sequence

        Returns
        -------
        joint : float
        """
        if self.unk is None:
            context = tuple(context)
        else:
            context = tuple(t if t in self.vocab else self.unk for t in context)
        if not len(context):
            raise ValueError("context must have at least one token")
        return self.trie.log_prob(context)

    def to_prob_list(self):
        return self.trie.to_prob_list()

    def renormalize_backoffs(self):
        r"""Ensure backoffs induce a valid probability distribution

        Backoff models follow the same recursive formula for determining the probability
        of the next token:

        .. math::

            Pr(w_n|w_1, \ldots w_{n-1}) = \begin{cases}
                Entry(w_1, \ldots, w_n) &
                                    \text{if }Entry(\ldots)\text{ exists}\\
                Backoff(w_1, \ldots, w_{n-1})P(w_n|w_{n-1}, \ldots, w_2) &
                                    \text{otherwise}
            \end{cases}

        Calling this method renormalizes :math:`Backoff(\ldots)` such that,
        where possible, :math:`\sum_w Pr(w|\ldots) = 1`
        """
        return self.trie.renormalize_backoffs()

    def relative_entropy_pruning(self, threshold, _srilm_hacks=False):
        r"""Prune n-grams with negligible impact on model perplexity

        This method iterates through n-grams, highest order first, looking to absorb
        their explicit probabilities into a backoff. The language model defines a
        distribution over sequences, :math:`s \sim p(\cdot|\theta)`. Assuming this is
        the true distribution of sequences, we can define an approximation of
        :math:`p(\cdot)`, :math:`q(\cdot)`, as one that replaces one explicit n-gram
        probability with a backoff. [stolcke2000]_ defines the relative change in model
        perplexity as:

        .. math::

            \Delta PP = e^{D_{KL}(p\|q)} - 1

        Where :math:`D_{KL}` is the KL-divergence between the two distributions. This
        method will prune an n-gram whenever the change in model perplexity is
        negligible (below `threshold`). More details can be found in [stolcke2000]_.

        Parameters
        ----------
        threshold : float

        References
        ----------
        .. [stolcke2000] A. Stolcke "Entropy-based pruning of Backoff Language Models,"
           ArXiv ePrint, 2000
        """
        return self.trie.relative_entropy_pruning(threshold, _srilm_hacks=_srilm_hacks)

    def sequence_perplexity(self, sequence, include_delimiters=True):
        r"""Return the perplexity of the sequence using this language model

        Given a `sequence` of tokens ``[w_1, w_2, ..., w_N]``, the perplexity of the
        sequence is

        .. math::

            Pr(sequence)^{-1/N} = Pr(w_1, w_2, ..., w_N)^{-1/N}

        Parameters
        ----------
        sequence : sequence
        include_delimiters : bool, optional
            If :obj:`True`, the sequence will be prepended with the
            start-of-sequence symbol and appended with an end-of-sequence
            symbol, assuming they do not already exist as prefix and suffix of
            `sequence`

        Notes
        -----
        If the first token in `sequence` is the start-of-sequence token (or
        it is added using `include_delimiters`), it will not be included in
        the count ``N`` because ``Pr(sos) = 1`` always. An end-of-sequence
        token is always included in ``N``.
        """
        sequence = list(sequence)
        if include_delimiters:
            if self.sos is not None and (not len(sequence) or sequence[0] != self.sos):
                sequence.insert(0, self.sos)
            if self.eos is not None and sequence[-1] != self.eos:
                sequence.append(self.eos)
        if not len(sequence):
            raise ValueError(
                "sequence cannot be empty when include_delimiters is False"
            )
        N = len(sequence)
        if sequence[0] == self.sos:
            N -= 1
        return 10.0 ** (-self.log_prob(sequence) / N)

    def corpus_perplexity(self, corpus, include_delimiters=True) -> float:
        r"""Calculate the perplexity of an entire corpus using this model

        A `corpus` is a sequence of sequences ``[s_1, s_2, ..., s_S]``. Each
        sequence ``s_i`` is a sequence of tokens ``[w_1, w_2, ..., w_N_i]``.
        Assuming sentences are independent,

        .. math::

            Pr(corpus) = Pr(s_1, s_2, ..., s_S) = Pr(s_1)Pr(s_2)...Pr(s_S)

        We calculate the corpus perplexity as the inverse corpus probablity
        normalized by the total number of tokens in the corpus. Letting
        :math:`M = \sum_i^S N_i`, the corpus perplexity is

        .. math::

            Pr(corpus)^{-1/M}

        Parameters
        ----------
        corpus : sequence
        include_delimiters : bool, optional
            Whether to add start- and end-of-sequence delimiters to each sequence if
            necessary and available. See :func:`sequence_complexity` for more info
        """
        joint = 0.0
        M = 0
        for sequence in corpus:
            sequence = list(sequence)
            if include_delimiters:
                if self.sos is not None and (
                    not len(sequence) or sequence[0] != self.sos
                ):
                    sequence.insert(0, self.sos)
                if self.eos is not None and sequence[-1] != self.eos:
                    sequence.append(self.eos)
            if not len(sequence):
                warnings.warn("skipping empty sequence")
                continue
            N = len(sequence)
            if sequence[0] == self.sos:
                N -= 1
            M += N
            joint += self.log_prob(sequence)
        return 10.0 ** (-joint / M)

    def prune_by_threshold(self, lprob):
        """Prune n-grams with a log-probability <= a threshold

        This method prunes n-grams with a conditional log-probability less than or equal
        to some fixed threshold. The reclaimed probability mass is sent to the
        (n-1)-gram's backoff.

        This method never prunes unigrams. Further, it cannot prune n-grams which are a
        prefix of some higher-order n-gram that has a conditional probability above that
        threshold, since the higher-order n-gram may have need of the lower-order's
        backoff.

        Parameters
        ----------
        lprob : float
            The base-10 log probability of conditionals, below or at which the n-gram
            will be pruned.
        """
        self.trie.prune_by_threshold(lprob)

    def prune_by_name(self, to_prune, eps_lprob=DEFT_EPS_LPROB):
        """Prune n-grams by name

        This method prunes n-grams of arbitrary order by name. For n-grams of order > 1,
        the reclaimed probability mass is allotted to the appropriate backoff. For
        unigrams, the reclaimed probability mass is distributed uniformly across the
        remaining unigrams.

        This method prunes nodes by setting their probabilities a small log-probability
        (`eps_lprob`), then calling :func:`prune_by_threshold` with that small
        log-probability. This ensures we do not remove the backoff of higher-order
        n-grams (instead setting the probability of "pruned" nodes very low), and gets
        rid of lower-order nodes that were previously "pruned" but had to exist for
        their backoff when their backoff is now no longer needed.

        Unigrams are never fully pruned - their log probabilities are set to
        `eps_lprob`.

        Parameters
        ----------
        to_prune : set
            A set of all n-grams of all orders to prune.
        eps_lprob : float, optional
            A base 10 log probability considered negligible
        """
        self.trie.prune_by_name(to_prune, eps_lprob)


class _NoCacheDict(MutableMapping):
    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, key):
        raise KeyError

    def __setitem__(self, key, value) -> None:
        pass

    def __len__(self) -> int:
        return 0

    def __delitem__(self, key) -> None:
        pass

    def __iter__(self) -> Iterator:
        return iter(tuple())


V = TypeVar("V")


class DbNgramDict(MutableMapping[Ngram, V]):
    VALUE_FMT: str = NotImplemented

    store: MutableMapping[bytes, bytes]
    cache: MutableMapping[Ngram, V]
    ngram_encoding: str
    separator: str
    _is_diskcache: bool

    @classmethod
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)

        if cls.VALUE_FMT is NotImplemented:
            raise NotImplementedError(f"{cls} needs to define VALUE_FMT")

    def __init__(
        self,
        store: MutableMapping,
        ngram_encoding: str = "utf-8",
        separator: str = "\n",
        max_cache_len: Optional[int] = DEFT_MAX_CACHE_LEN,
        max_cache_age_seconds: Optional[float] = DEFT_MAX_CACHE_AGE_SECONDS,
    ) -> None:
        if not len(separator):
            raise ValueError("separator cannot be empty")
        super().__init__()
        self.store = store
        self._is_diskcache = Cache is not None and isinstance(store, Cache)
        if max_cache_age_seconds * max_cache_len > 0:
            try:
                from expiringdict import ExpiringDict

                self.cache = ExpiringDict(max_cache_len, max_cache_age_seconds)
            except ImportError:
                warnings.warn("expiringdict not installed. not caching counts")
                self.cache = _NoCacheDict()
        else:
            self.cache = _NoCacheDict()
        self.ngram_encoding, self.separator = ngram_encoding, separator

    def encode_ngram(self, ngram: Ngram) -> bytes:
        if not isinstance(ngram, str):
            ngram = self.separator.join(ngram)
        return ngram.encode(self.ngram_encoding)

    def decode_ngram(self, ngram_b: bytes) -> Ngram:
        key = ngram_b.decode(self.ngram_encoding)
        assert isinstance(key, str)
        if self.separator in key:
            key = tuple(key.split(self.separator))
        return key

    def encode_value(self, val: V) -> bytes:
        return struct.pack(self.VALUE_FMT, val)

    def decode_value(self, val_b: bytes) -> V:
        return struct.unpack(self.VALUE_FMT, val_b)[0]

    def encode_pair(self, ngram: Ngram, val: V) -> Tuple[bytes, bytes]:
        return self.encode_ngram(ngram), self.encode_value(val)

    def decode_pair(self, ngram_b: bytes, val_b: bytes) -> Tuple[Ngram, V]:
        return self.decode_ngram(ngram_b), self.decode_value(val_b)

    def __iter__(self):
        if self._is_diskcache:
            for ngram_b in self.store:
                yield self.decode_ngram(ngram_b)
        else:
            for ngram_b in self.store.keys():
                yield self.decode_ngram(ngram_b)

    def __len__(self) -> int:
        return len(self.store)

    def __contains__(self, ngram: Ngram) -> bool:
        return (ngram in self.cache) or (self.encode_ngram(ngram) in self.store)

    def get(self, ngram: Ngram, default=None):
        try:
            return self.cache[ngram]
        except KeyError:
            ngram_b = self.encode_ngram(ngram)
            try:
                val = self.cache[ngram] = self.decode_value(self.store[ngram_b])
                return val
            except KeyError:
                return default

    def __getitem__(self, ngram: Ngram) -> V:
        try:
            return self.cache[ngram]
        except KeyError:
            pass
        ngram_b = self.encode_ngram(ngram)
        val = self.cache[ngram_b] = self.decode_value(self.store[ngram_b])
        return val

    def __setitem__(self, ngram: Ngram, val: V):
        ngram_b, val_b = self.encode_pair(ngram, val)
        self.cache[ngram] = val
        self.store[ngram_b] = val_b

    def __delitem__(self, ngram: Ngram) -> None:
        ngram_b = self.encode_ngram(ngram)
        del self.store[ngram_b]
        try:
            del self.cache[ngram]
        except KeyError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        try:
            if self.store is not None and hasattr(self.store, "close"):
                self.store.close()
        finally:
            self.store = None


class DbCountDict(DbNgramDict[int]):
    """Disk-backed dictionary of n-gram counts

    Update shares semantics with :func:`collections.Counter.update`, i.e.

    >>> count_dict.update(ngrams)

    adds 1 to the value of every n-gram in `ngrams`, whereas

    >>> count_dict.update(other_count_dict)

    adds the counts in `other_count_dict` to `count_dict`
    """

    VALUE_FMT: Final[str] = "Q"

    def encode_value(self, val: int) -> Union[int, bytes]:
        if self._is_diskcache:
            return val
        else:
            return super().encode_value(val)

    def decode_value(self, val_b: Union[int, bytes]) -> int:
        if isinstance(val_b, int):
            return val_b
        else:
            return super().decode_value(val_b)

    def update(self, iterable=None, /, **kwds):
        if iterable is not None:
            if isinstance(iterable, Mapping):
                if not self:
                    super().update(iterable)
                elif self._is_diskcache:
                    assert isinstance(self.store, Cache)
                    with self.store.transact():
                        for ngram, count in iterable.items():
                            ngram_b = self.encode_ngram(ngram)
                            self.cache[ngram] = self.store.incr(ngram_b, count)
                else:
                    for ngram, count in iterable.items():
                        ngram_b = self.encode_ngram(ngram)
                        old_count = self.cache.get(ngram, None)
                        if old_count is None:
                            old_count_b = self.store.get(ngram_b, None)
                            if old_count_b is None:
                                old_count = 0
                            else:
                                old_count = self.decode_value(old_count_b)
                        count += old_count
                        self.cache[ngram] = count
                        self.store[ngram_b] = self.encode_value(count)
            else:
                counter = Counter(iterable)
                kwds.update(counter)

        if kwds:
            self.update(kwds)


class DbProbDict(DbNgramDict[LProb]):
    VALUE_FMT: Final[str] = "d"

    def encode_value(self, val: LProb) -> bytes:
        if isinstance(val, float) or isinstance(val, int):
            return super().encode_value(float(val))
        else:
            return struct.pack(self.VALUE_FMT * 2, float(val[0]), float(val[1]))

    def decode_value(self, val_b: bytes) -> LProb:
        val = tuple(x[0] for x in struct.iter_unpack(self.VALUE_FMT, val_b))
        if len(val) == 1:
            return val[0]
        else:
            return val


def get_store(
    filename: Optional[os.PathLike] = None, protocol: str = "c"
) -> MutableMapping[str, str]:
    if filename is None:
        filename = TemporaryDirectory()
    filename_ = os.fspath(filename)
    try:
        import diskcache

        os.makedirs(filename_, exist_ok=True)
        store = diskcache.Cache(filename_)
        if protocol == "n":
            store.clear()
    except ImportError:
        import dbm

        store = dbm.open(filename_, protocol)

    return store


def open_count_dict(
    filename: Optional[os.PathLike] = None,
    protocol: str = "c",
    ngram_encoding: str = "utf-8",
    separator: str = "\n",
    max_cache_len: Optional[int] = DEFT_MAX_CACHE_LEN,
    max_cache_age_seconds: Optional[float] = DEFT_MAX_CACHE_AGE_SECONDS,
) -> DbCountDict:
    store = get_store(filename, protocol)

    return DbCountDict(
        store,
        ngram_encoding,
        separator,
        max_cache_len,
        max_cache_age_seconds,
    )


def open_prob_dict(
    filename: Optional[os.PathLike] = None,
    protocol: str = "c",
    ngram_encoding: str = "utf-8",
    separator: str = "\n",
    max_cache_len: Optional[int] = DEFT_MAX_CACHE_LEN,
    max_cache_age_seconds: Optional[float] = DEFT_MAX_CACHE_AGE_SECONDS,
) -> DbProbDict:
    store = get_store(filename, protocol)

    return DbProbDict(
        store,
        ngram_encoding,
        separator,
        max_cache_len,
        max_cache_age_seconds,
    )


def write_arpa(prob_dicts: ProbDicts, out=sys.stdout):
    """Convert an lists of n-gram probabilities to arpa format

    The inverse operation of :func:`pydrobert.torch.util.parse_arpa_lm`

    Parameters
    ----------
    prob_dicts : list of dict
    out : file or str, optional
        Path or file object to output to
    """
    if isinstance(out, str):
        if out.endswith(".gz"):
            with gzip.open(out, "wt") as f:
                return write_arpa(prob_dicts, f)
        else:
            with open(out, "w") as f:
                return write_arpa(prob_dicts, f)
    entries_by_order = []
    for idx, dict_ in enumerate(prob_dicts):
        entries = sorted((k, v) if idx else ((k,), v) for (k, v) in dict_.items())
        entries_by_order.append(entries)
    out.write("\\data\\\n")
    for idx in range(len(entries_by_order)):
        out.write("ngram {}={}\n".format(idx + 1, len(entries_by_order[idx])))
    out.write("\n")
    for idx, entries in enumerate(entries_by_order):
        out.write("\\{}-grams:\n".format(idx + 1))
        if idx == len(entries_by_order) - 1:
            for entry in entries:
                out.write("{:f}\t{}\n".format(entry[1], " ".join(entry[0])))
        else:
            for entry in entries:
                x = f"{entry[1][0]:f}\t{' '.join(entry[0])}\t{entry[1][1]:f}\n"
                assert isinstance(x, str)
                out.write(x)
        out.write("\n")
    out.write("\\end\\\n")


def count_dicts_to_prob_list_mle(count_dicts, eps_lprob=DEFT_EPS_LPROB):
    r"""Determine probabilities based on MLE of observed n-gram counts

    For a given n-gram :math:`p, w`, where :math:`p` is a prefix, :math:`w` is the next
    word, the maximum likelihood estimate of the last token given the prefix is:

    .. math::

        Pr(w | p) = C(p, w) / (\sum_w' C(p, w'))

    Where :math:`C(x)` Is the count of the sequence :math:`x`. Many counts will be zero,
    especially for large n-grams or rare words, making this a not terribly generalizable
    solution.

    Parameters
    ----------
    count_dicts : sequence
        A list of dictionaries. ``count_dicts[0]`` should correspond to unigram counts
        in a corpus, ``count_dicts[1]`` to bi-grams, etc. Keys are tuples of tokens
        (n-grams) of the appropriate length, with the exception of unigrams, whose keys
        are the tokens themselves. Values are the counts of those n-grams in the corpus.
    eps_lprob : float, optional
        A very negative value substituted as "negligible probability"

    Returns
    -------
    prob_list : sequence
        Corresponding n-gram conditional probabilities. See
        :mod:`pydrobert.torch.util.parse_arpa_lm`

    Examples
    --------
    >>> from collections import Counter
    >>> text = 'a man a plan a canal panama'
    >>> count_dicts = [
    >>>     Counter(
    >>>         tuple(text[offs:offs + order]) if order > 1
    >>>         else text[offs:offs + order]
    >>>         for offs in range(len(text) - order + 1)
    >>>     )
    >>>     for order in range(1, 4)
    >>> ]
    >>> count_dicts[0]['<unk>'] = 0  # add oov to vocabulary
    >>> count_dicts[0]['a']
    10
    >>> sum(count_dicts[0].values())
    27
    >>> count_dicts[1][('a', ' ')]
    3
    >>> sum(v for (k, v) in count_dicts[1].items() if k[0] == 'a')
    9
    >>> prob_list = count_dicts_to_prob_list_mle(count_dicts)
    >>> prob_list[0]['a']   # (log10(10 / 27), eps_lprob)
    (-0.43136376415898736, -99.99)
    >>> '<unk>' in prob_list[0]  # no probability mass gets removed
    False
    >>> prob_list[1][('a', ' ')]  # (log10(3 / 9), eps_lprob)
    (-0.47712125471966244, -99.99)

    Notes
    -----
    To be compatible with back-off models, MLE estimates assign a negligible backoff
    probability (`eps_lprob`) to n-grams where necessary. This means the probability
    mass might not exactly sum to one.
    """
    return count_dicts_to_prob_list_add_k(count_dicts, eps_lprob=eps_lprob, k=0.0)


def _get_cond_mle(order, counts, vocab, k):
    n_counts = dict()  # C(p, w) + k
    d_counts = dict()  # \sum_w' C(p, w') + k|V|
    for ngram in product(vocab, repeat=order + 1):
        c = counts.get(ngram if order else ngram[0], 0) + k
        if not c:
            continue
        n_counts[ngram] = c
        d_counts[ngram[:-1]] = d_counts.get(ngram[:-1], 0) + c
    return dict(
        (ng, np.log10(num) - np.log10(d_counts[ng[:-1]]))
        for ng, num in n_counts.items()
    )


def count_dicts_to_prob_list_add_k(
    count_dicts: CountDicts, eps_lprob=DEFT_EPS_LPROB, k=DEFT_ADD_K_K
):
    r"""MLE probabilities with constant discount factor added to counts

    Similar to :func:`count_dicts_to_prob_list_mle`, but with a constant added to each
    count to smooth out probabilities:

    .. math::

        Pr(w|p) = (C(p,w) + k)/(\sum_w' C(p, w') + k|V|)

    Where :math:`p` is a prefix, :math:`w` is the next word, and :math:`V` is the
    vocabulary set. The initial vocabulary set is determined from the unique unigrams
    :math:`V = U`. The bigram vocabulary set is the Cartesian product :math:`V = U
    \times U`, trigrams :math:`V = U \times U \times U`, and so on.

    Parameters
    ----------
    count_dicts : sequence
        A list of dictionaries. ``count_dicts[0]`` should correspond to unigram counts
        in a corpus, ``count_dicts[1]`` to bi-grams, etc. Keys are tuples of tokens
        (n-grams) of the appropriate length, with the exception of unigrams, whose keys
        are the tokens themselves. Values are the counts of those n-grams in the corpus.
    eps_lprob : float, optional
        A very negative value substituted as "negligible probability"

    Returns
    -------
    prob_list : sequence
        Corresponding n-gram conditional probabilities. See
        :mod:`pydrobert.torch.util.parse_arpa_lm`

    Examples
    --------
    >>> from collections import Counter
    >>> text = 'a man a plan a canal panama'
    >>> count_dicts = [
    >>>     Counter(
    >>>         tuple(text[offs:offs + order]) if order > 1
    >>>         else text[offs:offs + order]
    >>>         for offs in range(len(text) - order + 1)
    >>>     )
    >>>     for order in range(1, 4)
    >>> ]
    >>> count_dicts[0]['<unk>'] = 0  # add oov to vocabulary
    >>> count_dicts[0]['a']
    10
    >>> sum(count_dicts[0].values())
    27
    >>> ('a', '<unk>') not in count_dicts[1]
    True
    >>> sum(v for (k, v) in count_dicts[1].items() if k[0] == 'a')
    9
    >>> prob_list = count_dicts_to_prob_list_add_k(count_dicts, k=1)
    >>> prob_list[0]['a']   # (log10((10 + 1) / (27 + 8)), eps_lprob)
    (-0.5026753591920505, -99.999)
    >>> # Pr('a' | '<unk>') = (C('<unk>', 'a') + k) / (C('<unk>', .) + k|V|)
    >>> #                   = 1 / 8
    >>> prob_list[1][('<unk>', 'a')]  # (log10(1 / 8), eps_lprob)
    (-0.9030899869919435, -99.999)
    >>> # Pr('<unk>' | 'a') = (C('a', '<unk>') + k) / (C('a', .) + k|V|)
    >>> #                   = 1 / (9 + 8)
    >>> prob_list[1][('a', '<unk>')]  # (log10(1 / 17), eps_lprob)
    (-1.2304489213782739, -99.999)
    """
    max_order = len(count_dicts) - 1
    if not len(count_dicts):
        raise ValueError("At least unigram counts must exist")
    vocab = set(count_dicts[0])
    prob_list = []
    for order, counts in enumerate(count_dicts):
        probs = _get_cond_mle(order, counts, vocab, k)
        if not order:
            for v in vocab:
                probs.setdefault((v,), eps_lprob)
        if order != max_order:
            probs = dict((ngram, (prob, eps_lprob)) for (ngram, prob) in probs.items())
        prob_list.append(probs)
    prob_list[0] = dict((ngram[0], p) for (ngram, p) in prob_list[0].items())
    return prob_list


def _log10sumexp(*args):
    if len(args) > 1:
        return _log10sumexp(args)
    args = np.array(args, dtype=float, copy=False)
    x = args[0]
    if np.any(np.isnan(x)):
        return np.nan
    if np.any(np.isposinf(x)):
        return np.inf
    x = x[np.isfinite(x)]
    if not len(x):
        return 0.0
    max_ = np.max(x)
    return np.log10((10 ** (x - max_)).sum()) + max_


def _simple_good_turing_counts(counts, eps_lprob):
    # this follows GT smoothing w/o tears section 6 pretty closely. You might
    # not know what's happening otherwise
    N_r = Counter(counts.values())
    max_r = max(N_r.keys())
    N_r = np.array(tuple(N_r.get(i, 0) for i in range(max_r + 2)))
    N_r[0] = 0
    r = np.arange(max_r + 2)
    N = (N_r * r).sum()
    log_N = np.log10(N)
    nonzeros = np.where(N_r != 0)[0]

    # find S(r) = a r^b
    Z_rp1 = 2.0 * N_r[1:-1]
    j = r[1:-1]
    diff = nonzeros - j[..., None]
    i = j - np.where(-diff < 1, max_r, -diff).min(1)
    i[0] = 0
    k = j + np.where(diff < 1, max_r, diff).min(1)
    k[-1] = 2 * j[-1] - i[-1]
    Z_rp1 /= k - i
    y = np.log10(Z_rp1[nonzeros - 1])  # Z_rp1 does not include r=0
    x = np.log10(r[nonzeros])
    # regress on y = bx + a
    mu_x, mu_y = x.mean(), y.mean()
    num = ((x - mu_x) * (y - mu_y)).sum()
    denom = ((x - mu_x) ** 2).sum()
    b = num / denom if denom else 0.0
    a = mu_y - b * mu_x
    log_Srp1 = a + b * np.log10(r[1:])

    # determine direct estimates of r* (x) as well as regressed estimates of
    # r* (y). Use x until absolute difference between x and y is statistically
    # significant (> 2 std dev of gauss defined by x)
    log_r_star = np.empty(max_r + 1, dtype=float)
    log_Nr = log_r_star[0] = np.log10(N_r[1]) if N_r[1] else eps_lprob + log_N
    switched = False
    C, ln_10 = np.log10(1.69), np.log(10)
    for r_ in range(1, max_r + 1):
        switched |= not N_r[r_]
        log_rp1 = np.log10(r_ + 1)
        log_y = log_rp1 + log_Srp1[r_] - log_Srp1[r_ - 1]
        if not switched:
            if N_r[r_ + 1]:
                log_Nrp1 = np.log10(N_r[r_ + 1])
            else:
                log_Nrp1 = eps_lprob + log_N + log_Nr
            log_x = log_rp1 + log_Nrp1 - log_Nr
            if log_y > log_x:
                log_abs_diff = log_y + np.log1p(-np.exp(log_x - log_y))
            elif log_x < log_y:
                log_abs_diff = log_x + np.log1p(-np.exp(log_y - log_x))
            else:
                log_abs_diff = -float("inf")
            log_z = C + log_rp1 - log_Nr + 0.5 * log_Nrp1
            log_z += 0.5 * np.log1p(N_r[r_ + 1] / N_r[r_]) / ln_10
            if log_abs_diff <= log_z:
                switched = True
            else:
                log_r_star[r_] = log_x
            log_Nr = log_Nrp1
        if switched:
            log_r_star[r_] = log_y

    # G&S tell us to renormalize the prob mass among the nonzero r terms. i.e.
    # p[0] = r_star[0] / N
    # p[i] = (1 - p[0]) r_star[i] / N'
    # where N' = \sum_i>0 N_r[i] r_star[i]
    # we convert back to counts so that our conditional MLEs are accurate
    max_log_r_star = np.max(log_r_star[1:][nonzeros[:-1] - 1])
    log_Np = np.log10((N_r[1:-1] * 10 ** (log_r_star[1:] - max_log_r_star)).sum())
    log_Np += max_log_r_star
    log_p_0 = log_r_star[0] - log_N
    log_r_star[1:] += -log_Np + np.log10(1 - 10**log_p_0) + log_N

    return log_r_star


def count_dicts_to_prob_list_simple_good_turing(
    count_dicts: CountDicts, eps_lprob=DEFT_EPS_LPROB
):
    r"""Determine probabilities based on n-gram counts using simple good-turing

    Simple Good-Turing smoothing discounts counts of n-grams according to the following
    scheme:

    .. math::

        r^* = (r + 1) N_{r + 1} / N_r

    Where :math:`r` is the original count of the n-gram in question, :math:`r^*` the
    discounted, and :math:`N_r` is the count of the number of times any n-gram had a
    count `r`.

    When :math:`N_r` becomes sparse, it is replaced with a log-linear regression of
    :math:`N_r` values, :math:`S(r) = a + b \log r`. :math:`r^*` for :math:`r > 0` are
    renormalized so that :math:`\sum_r N_r r^* = \sum_r N_r r`.

    We assume a closed vocabulary and that, for any order n-gram, :math:`N_0` is the
    size of the set of n-grams with frequency zero. This method differs from traditional
    Good-Turing, which assumes one unseen "event" (i.e. n-gram) per level. See below
    notes for more details.

    If, for a given order of n-gram, none of the terms have frequency zero, this
    function will warn and use MLEs.

    Parameters
    ----------
    count_dicts : sequence
        A list of dictionaries. ``count_dicts[0]`` should correspond to unigram counts
        in a corpus, ``count_dicts[1]`` to bi-grams, etc. Keys are tuples of tokens
        (n-grams) of the appropriate length, with the exception of unigrams, whose keys
        are the tokens themselves. Values are the counts of those n-grams in the corpus.
    eps_lprob : float, optional
        A very negative value substituted as "negligible probability."

    Returns
    -------
    prob_list : sequence
        Corresponding n-gram conditional probabilities. See
        :mod:`pydrobert.torch.util.parse_arpa_lm`

    Notes
    -----
    The traditional definition of Good-Turing is somewhat vague about how to assign
    probability mass among unseen events. By setting :math:`r^* = N_1 / N` for :math:`r
    = 0`, it's implicitly stating that :math:`N_0 = 1`, that is, there's only one
    possible unseen event. This is consistent with introducing a special token, e.g.
    ``"<unk>"``, that does not occur in the corpus. It also collapses unseen n-grams
    into one event.

    We cannot bootstrap the backoff penalty to be the probability of the unseen term
    because the backoff will be combined with a lower-order estimate, and Good-Turing
    uses a fixed unseen probability.

    As our solution, we assume the vocabulary is closed. Any term that appears zero
    times is added to :math:`N_0`. If all terms appear, then :math:`N_0 = 0` and we
    revert to the MLE. While you can simulate the traditional Good-Turing at the
    unigram-level by introducing ``"<unk>"`` with count 0, this will not hold for
    higher-order n-grams.

    Warnings
    --------
    This function manually defines all n-grams of the target order given a vocabulary.
    This means that higher-order n-grams will be very large.

    Examples
    --------
    >>> from collections import Counter
    >>> text = 'a man a plan a canal panama'
    >>> count_dicts = [
    >>>     Counter(
    >>>         tuple(text[offs:offs + order]) if order > 1
    >>>         else text[offs:offs + order]
    >>>         for offs in range(len(text) - order + 1)
    >>>     )
    >>>     for order in range(1, 4)
    >>> ]
    >>> count_dicts[0]['<unk>'] = 0  # add oov to vocabulary
    >>> sum(count_dicts[0].values())
    27
    >>> Counter(count_dicts[0].values())
    Counter({2: 3, 10: 1, 6: 1, 4: 1, 1: 1, 0: 1})
    >>> # N_1 = 1, N_2 = 3, N_3 = 1
    >>> prob_list = count_dicts_to_prob_list_simple_good_turing(count_dicts)
    >>> # Pr('<unk>') = Pr(r=0) = N_1 / N_0 / N = 1 / 27
    >>> prob_list[0]['<unk>']   # (log10(1 / 27), eps_lprob)
    (-1.4313637641589874, -99.999)
    >>> # Pr('a'|'<unk>') = Cstar('<unk>', 'a') / (Cstar('unk', .))
    >>> #                 = rstar[0] / (|V| * rstar[0]) = 1 / 8
    >>> prob_list[1][('<unk>', 'a')]  # (log10(1 / 8), eps_lprob)
    (-0.9030899869919435, -99.999)

    References
    ----------
    .. [gale1995] W. A. Gale and G. Sampson, "Good-Turing frequency estimation without
       tears," Journal of Quantitative Linguistics, vol. 2, no. 3, pp. 217-237, Jan.
       1995.
    """
    if len(count_dicts) < 1:
        raise ValueError("At least unigram counts must exist")
    max_order = len(count_dicts) - 1
    vocab = set(count_dicts[0])
    prob_list = []
    for order, counts in enumerate(count_dicts):
        N_0_vocab = set()
        log_r_stars = _simple_good_turing_counts(counts, eps_lprob)
        n_counts = dict()
        d_counts = dict()
        for ngram in product(vocab, repeat=order + 1):
            r = counts.get(ngram if order else ngram[0], 0)
            if r:
                c = 10.0 ** log_r_stars[r]
                n_counts[ngram] = c
                d_counts[ngram[:-1]] = d_counts.get(ngram[:-1], 0) + c
            else:
                N_0_vocab.add(ngram)
        N_0 = len(N_0_vocab)
        if N_0:
            c = (10 ** log_r_stars[0]) / N_0
            for ngram in N_0_vocab:
                n_counts[ngram] = c
                d_counts[ngram[:-1]] = d_counts.get(ngram[:-1], 0) + c
            probs = dict(
                (ng, np.log10(n_counts[ng]) - np.log10(d_counts[ng[:-1]]))
                for ng in n_counts
            )
        else:
            warnings.warn(
                "No {}-grams were missing. Using MLE instead" "".format(order + 1)
            )
            probs = _get_cond_mle(order, counts, vocab, 0)
        if order != max_order:
            probs = dict((ngram, (prob, eps_lprob)) for (ngram, prob) in probs.items())
        prob_list.append(probs)
    prob_list[0] = dict((ngram[0], p) for (ngram, p) in prob_list[0].items())
    return prob_list


def _get_katz_discounted_counts(counts, k):
    N_r = Counter(counts.values())
    max_r = max(N_r.keys())
    N_r = np.array(tuple(N_r.get(i, 0) for i in range(max_r + 2)))
    N_r[0] = 1
    r = np.arange(max_r + 2)
    N = (N_r * r).sum()
    log_N = np.log10(N)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_Nr = np.log10(N_r)
        log_rp1 = np.log10(r + 1)
        log_r_star = log_rp1[:-1] + log_Nr[1:] - log_Nr[:-1]
    if k + 1 < len(N_r):
        log_d_rp1 = np.zeros(max_r, dtype=float)
        log_num_minu = log_r_star[1 : k + 1] - log_rp1[:k]
        log_subtra = np.log10(k + 1) + log_Nr[k + 1] - log_Nr[1]
        if log_subtra >= 0:
            raise ValueError("Your corpus is too small for this")
        # np.log10((10 ** (x - max_)).sum()) + max_
        log_num = log_num_minu + np.log1p(
            -(10 ** (log_subtra - log_num_minu))
        ) / np.log(10)
        log_denom = np.log1p(-(10**log_subtra)) / np.log(10)
        log_d_rp1[:k] = log_num - log_denom
    else:
        log_d_rp1 = log_r_star[1:] - log_rp1[:-1]
    log_r_star = np.empty(max_r + 1, dtype=float)
    log_r_star[0] = log_Nr[1]
    log_r_star[1:] = log_d_rp1 + log_rp1[:-2]
    assert np.isclose(_log10sumexp(log_r_star + log_Nr[:-1]), log_N)
    return log_r_star


def count_dicts_to_prob_list_katz_backoff(
    count_dicts: CountDicts,
    thresh=DEFAULT_KATZ_THRESH,
    eps_lprob=DEFT_EPS_LPROB,
    _cmu_hacks=False,
):
    r"""Determine probabilities based on Katz's backoff algorithm

    Kat'z backoff algorithm determines the conditional probability of the last token in
    n-gram :math:`w = (w_1, w_2, ..., w_n)` as

    .. math::

        Pr_{BO}(w_n|w_{n-1}, w_{n-2} ..., w_1) = \begin{cases}
            d_w Pr_{MLE}(w_n|w_{n-1}, w_{n-1}, ..., w_1) & \text{if }C(w) > 0
            \alpha(w_1, ..., w_{n-1}) Pr_{BO}(w_n|w_{n-1}, ..., w_2)&
                                                                    \text{else}
        \end{cases}

    Where :math:`Pr_{MLE}` is the maximum likelihood estimate (based on frequencies),
    :math:`d_w` is some discount factor (based on Good-Turing for low-frequency
    n-grams), and :math:`\alpha` is an allowance of the leftover probability mass from
    discounting.

    Parameters
    ----------
    count_dicts : sequence
        A list of dictionaries. ``count_dicts[0]`` should correspond to unigram counts
        in a corpus, ``count_dicts[1]`` to bi-grams, etc. Keys are tuples of tokens
        (n-grams) of the appropriate length, with the exception of unigrams, whose keys
        are the tokens themselves. Values are the counts of those n-grams in the corpus.
    thresh : int, optional
        `k` is a threshold such that, if :math:`C(w) > k`, no discounting will be
        applied to the term. That is, the probability mass assigned for backoff will be
        entirely from n-grams s.t. :math:`C(w) \leq k`.
    eps_lprob : float, optional
        A very negative value substituted as "negligible probability."

    Warnings
    --------
    If the counts of the extensions of a prefix are all above `k`, no discounting will
    be applied to those counts, meaning no probability mass can be assigned to unseen
    events.

    For example, in the Brown corpus, "Hong" is always followed by "Kong". The bigram
    "Hong Kong" occurs something like 10 times, so it's not discounted. Thus
    :math:`P_{BO}(Kong|Hong) = 1` :math:`P_{BO}(not Kong|Hong) = 0`.

    A :class:`UserWarning` will be issued whenever ths happens. If this bothers you, you
    could try increasing `k` or, better yet, abandon Katz Backoff altogether.

    Returns
    -------
    prob_list : sequence
        Corresponding n-gram conditional probabilities. See
        :mod:`pydrobert.torch.util.parse_arpa_lm`

    Examples
    --------
    >>> from nltk.corpus import brown
    >>> from collections import Counter
    >>> text = tuple(brown.words())[:20000]
    >>> count_dicts = [
    >>>     Counter(
    >>>         text[offs:offs + order] if order > 1
    >>>         else text[offs]
    >>>         for offs in range(len(text) - order + 1)
    >>>     )
    >>>     for order in range(1, 4)
    >>> ]
    >>> del text
    >>> prob_list = count_dicts_to_prob_list_katz_backoff(count_dicts)

    References
    ----------
    .. [katz1987] S. Katz, "Estimation of probabilities from sparse data for the
       language model component of a speech recognizer," IEEE Transactions on Acoustics,
       Speech, and Signal Processing, vol. 35, no. 3, pp. 400-401, Mar. 1987.
    """
    if len(count_dicts) < 1:
        raise ValueError("At least unigram counts must exist")
    if thresh < 1:
        raise ValueError("k too low")
    prob_list = []
    max_order = len(count_dicts) - 1
    probs = _get_cond_mle(0, count_dicts[0], set(count_dicts[0]), 0)
    if 0 != max_order:
        probs = dict((ngram, (prob, 0.0)) for (ngram, prob) in probs.items())
    prob_list.append(probs)
    log_r_stars = [
        _get_katz_discounted_counts(counts, thresh) for counts in count_dicts[1:]
    ]
    if _cmu_hacks:
        # A note on CMU compatibility. First, the standard non-ML estimate of
        # P(w|p) = C(p, w) / C(p) instead of P(w|p) = C(p, w) / sum_w' C(p, w')
        # Second, this below loop. We add one to the count of a prefix whenever
        # that prefix has only one child and that child's count is greater than
        # k (in increment_context.cc). This ensures there's a non-zero backoff
        # to assign to unseen contexts starting with that prefix (N.B. this
        # hack should be extended to the case where all children have count
        # greater than k, but I don't want to reinforce this behaviour). Note
        # that it is applied AFTER the MLE for unigrams, and AFTER deriving
        # discounted counts.
        for order in range(len(count_dicts) - 1, 0, -1):
            prefix2children = dict()
            for ngram, count in count_dicts[order].items():
                prefix2children.setdefault(ngram[:-1], []).append(ngram)
            for prefix, children in prefix2children.items():
                if len(children) == 1 and count_dicts[order][children[0]] > thresh:
                    for oo in range(order):
                        pp = prefix[: oo + 1]
                        if not oo:
                            pp = pp[0]
                        count_dicts[oo][pp] += 1
    for order in range(1, len(count_dicts)):
        counts = count_dicts[order]
        probs = dict()
        # P_katz(w|pr) = C*(pr, w) / \sum_x C*(pr, x) if C(pr, w) > 0
        #                alpha(pr) Pr_katz(w|pr[1:]) else
        # alpha(pr) = (1 - sum_{c(pr, w) > 0} Pr_katz(w|pr)
        #             / (1 - sum_{c(pr, w) > 0} Pr_katz(w|pr[1:]))
        # note: \sum_w C*(pr, w) = \sum_w C(pr, w), which is why we can
        # normalize by the true counts
        lg_num_subtras = dict()  # logsumexp(log c*(pr,w)) for c(pr,w) > 0
        lg_den_subtras = dict()  # logsumexp(log Pr(w|pr[1:]) for c(pr, w) > 0
        lg_pref_counts = dict()  # logsumexp(log c(pr)) for c(pr,w) > 0
        for ngram, r in counts.items():
            if not r:
                continue
            log_r_star = log_r_stars[order - 1][r]
            probs[ngram] = log_r_star
            lg_num_subtras[ngram[:-1]] = _log10sumexp(
                lg_num_subtras.get(ngram[:-1], -np.inf), log_r_star
            )
            lg_den_subtras[ngram[:-1]] = _log10sumexp(
                lg_den_subtras.get(ngram[:-1], -np.inf), prob_list[-1][ngram[1:]][0]
            )
            lg_pref_counts[ngram[:-1]] = _log10sumexp(
                lg_pref_counts.get(ngram[:-1], -np.inf), np.log10(r)
            )
        for ngram in probs:
            prefix = ngram[:-1]
            if _cmu_hacks:
                if order == 1:
                    prefix = prefix[0]
                lg_norm = np.log10(count_dicts[order - 1][prefix])
            else:
                lg_norm = lg_pref_counts[prefix]
            probs[ngram] -= lg_norm
        for prefix, lg_num_subtra in lg_num_subtras.items():
            lg_den_subtra = lg_den_subtras[prefix]
            if _cmu_hacks:
                if order == 1:
                    lg_norm = np.log10(count_dicts[order - 1][prefix[0]])
                else:
                    lg_norm = np.log10(count_dicts[order - 1][prefix])
            else:
                lg_norm = lg_pref_counts[prefix]
            num_subtra = 10.0 ** (lg_num_subtra - lg_norm)
            den_subtra = 10.0**lg_den_subtra
            if np.isclose(den_subtra, 1.0):  # 1 - den_subtra = 0
                # If the denominator is zero, it means nothing we're backing
                # off to has a nonzero probability. It doesn't really matter
                # what we put here, but let's not warn about it (we've already
                # warned about the prefix)
                log_alpha = 0.0
            elif np.isclose(num_subtra, 1.0):
                warnings.warn(
                    "Cannot back off to prefix {}. Will assign negligible "
                    "probability. If this is an issue, try increasing k"
                    "".format(prefix)
                )
                # If the numerator is zero and the denominator is nonzero,
                # this means we did not discount any probability mass for
                # unseen terms. The only way to make a proper distribution is
                # to set alpha to zero
                log_alpha = eps_lprob
            else:
                log_alpha = np.log1p(-num_subtra) - np.log1p(-den_subtra)
                log_alpha /= np.log(10)
            log_prob = prob_list[-1][prefix][0]
            prob_list[-1][prefix] = (log_prob, log_alpha)
        if order != max_order:
            probs = dict((ngram, (prob, eps_lprob)) for (ngram, prob) in probs.items())
        prob_list.append(probs)
    prob_list[0] = dict((ngram[0], p) for (ngram, p) in prob_list[0].items())
    return prob_list


def _optimal_deltas(counts, y):
    N_r = Counter(counts.values())
    if not all(N_r[r] for r in range(1, y + 2)):
        raise ValueError(
            "Your dataset is too small to use the default discount "
            "(or maybe you removed the hapax before estimating probs?)"
        )
    Y = N_r[1] / (N_r[1] + 2 * N_r[2])
    deltas = [r - (r + 1) * Y * N_r[r + 1] / N_r[r] for r in range(1, y + 1)]
    if any(d <= 0.0 for d in deltas):
        raise ValueError(
            "Your dataset is too small to use the default discount "
            "(or maybe you removed the hapax before estimating probs?)"
        )
    return deltas


def _absolute_discounting(
    count_dicts: CountDicts, deltas, to_prune, temp: Optional[Path] = None
):
    V = len(set(count_dicts[0]) - to_prune)
    prob_list = [{tuple(): (-np.log10(V), 0.0)}]
    max_order = len(count_dicts) - 1

    for order, counts, delta in zip(range(len(count_dicts)), count_dicts, deltas):
        logging.info(f"Discounting order {order + 1}")
        delta = np.array(delta, dtype=float)
        n_counts, d_counts, pr2bin = dict(), dict(), dict()
        for ngram, count in counts.items():
            in_prune = ngram in to_prune
            if not order:
                ngram = (ngram,)
                if not in_prune:
                    n_counts.setdefault(ngram, 0)
            if not count:
                continue
            bin_ = min(count - 1, len(delta) - 1)
            d = delta[bin_]
            assert count - d >= 0.0
            prefix = ngram[:-1]
            d_counts[prefix] = d_counts.get(prefix, 0) + count
            prefix_bins = pr2bin.setdefault(prefix, np.zeros(len(delta) + 1))
            if in_prune:
                prefix_bins[-1] += count
            else:
                prefix_bins[bin_] += 1
                n_counts[ngram] = count - d
        for prefix, prefix_bins in pr2bin.items():
            if (order == 1 and prefix[0] in to_prune) or prefix in to_prune:
                continue
            with np.errstate(divide="ignore"):
                prefix_bins = np.log10(prefix_bins)
                prefix_bins[:-1] += np.log10(delta)
            gamma = _log10sumexp(prefix_bins)
            gamma -= np.log10(d_counts[prefix])
            lprob = prob_list[-1][prefix][0]
            prob_list[-1][prefix] = (lprob, gamma)
        if temp is None or order == 0:
            probs = dict()
        else:
            probs = open_prob_dict(
                temp / LPROBFILE_FMT_PREFIX.format(order=order + 1), "n"
            )
        for ngram, n_count in n_counts.items():
            prefix = ngram[:-1]
            if n_count:
                lprob = np.log10(n_count) - np.log10(d_counts[prefix])
            else:
                lprob = -float("inf")
            lower_order = prob_list[-1][prefix][1]  # gamma(prefix)
            lower_order += prob_list[-1][ngram[1:]][0]  # Pr(w|prefix[1:])
            lprob = _log10sumexp(lprob, lower_order)
            if order != max_order:
                # the only time the backoff will not be recalculated is if
                # no words ever follow the prefix. In this case, we actually
                # want to back off to a lower-order estimate
                # Pr(w|prefix) = P(w|prefix[1:]). We can achieve this by
                # setting gamma(prefix) = 1 and treating the higher-order
                # contribution to Pr(w|prefix) as zero
                lprob = (lprob, 0.0)
            probs[ngram] = lprob
        prob_list.append(probs)
    del prob_list[0]  # zero-th order
    prob_list_ = prob_list[0]
    if temp is None:
        prob_list[0] = dict()
    else:
        prob_list[0] = open_prob_dict(temp / LPROBFILE_FMT_PREFIX.format(order=1), "n")
    prob_list[0].update((ngram[0], p) for (ngram, p) in prob_list_.items())
    return prob_list


def count_dicts_to_prob_list_absolute_discounting(
    count_dicts: CountDicts, delta=None, to_prune=set()
):
    r"""Determine probabilities from n-gram counts using absolute discounting

    Absolute discounting (based on the formulation in [chen1999]_) interpolates between
    higher-order and lower-order n-grams as

    .. math::

        Pr_{abs}(w_n|w_{n-1}, \ldots w_1) =
            \frac{\max\left\{C(w_1, \ldots, w_n) - \delta, 0\right\}}
                {\sum_w' C(w_1, \ldots, w_{n-1}, w')}
            - \gamma(w_1, \ldots, w_{n-1})
               Pr_{abs}(w_n|w_{n-1}, \ldots, w_2)

    Where :math:`\gamma` are chosen so :math:`Pr_{abs}(\cdot)` sum to one. For the base
    case, we pretend there's such a thing as a zeroth-order n-gram, and
    :math:`Pr_{abs}(\emptyset) = 1 / \left\|V\right\|`.

    Letting

    .. math::

        N_c = \left|\left\{
                (w'_1, \ldots, w'_n): C(w'_1, \ldots, w'_n) = c
            \right\}\right|

    :math:`\delta` is often chosen to be :math:`\delta = N_1 / (N_1 + 2N_2)` for a given
    order n-gram. We can use different :math:`\delta` for different orders of the
    recursion.

    Parameters
    ----------
    count_dicts : sequence
        A list of dictionaries. ``count_dicts[0]`` should correspond to unigram counts
        in a corpus, ``count_dicts[1]`` to bi-grams, etc. Keys are tuples of tokens
        (n-grams) of the appropriate length, with the exception of unigrams, whose keys
        are the tokens themselves. Values are the counts of those n-grams in the corpus.
    delta : float or tuple or :obj:`None`, optional
        The absolute discount to apply to non-zero values. `delta` can take one of three
        forms: a :class:`float` to be used identically for all orders of the recursion;
        :obj:`None` specifies that the above formula for calculating `delta` should be
        used separately for each order of the recursion; or a tuple of length
        ``len(count_dicts)``, where each element is either a :class:`float` or
        :obj:`None`, specifying either a fixed value or the default value for every
        order of the recursion (except the zeroth-order), unigrams first.
    to_prune : set, optional
        A set of n-grams that will not be explicitly set in the return value. This
        differs from simply removing those n-grams from `count_dicts` in some key ways.
        First, pruned counts can still be used to calculate default `delta` values.
        Second, as per [chen1999]_, pruned counts are still summed in the denominator,
        :math:`\sum_w' C(w_1, \ldots, w_{n-1}, w')`, which then make their way into the
        numerator of :math:`gamma(w_1, \ldots, w_{n-1})`.

    Returns
    -------
    prob_list : sequence
        Corresponding n-gram conditional probabilities. See
        :mod:`pydrobert.torch.util.parse_arpa_lm`

    Examples
    --------
    >>> from collections import Counter
    >>> text = 'a man a plan a canal panama'
    >>> count_dicts = [
    >>>     Counter(
    >>>         tuple(text[offs:offs + order]) if order > 1
    >>>         else text[offs:offs + order]
    >>>         for offs in range(len(text) - order + 1)
    >>>     )
    >>>     for order in range(1, 4)
    >>> ]
    >>> count_dicts[0]['a']
    10
    >>> sum(count_dicts[0].values())
    27
    >>> len(count_dicts[0])
    7
    >>> sum(1 for k in count_dicts[1] if k[0] == 'a')
    4
    >>> sum(v for k, v in count_dicts[1].items() if k[0] == 'a')
    9
    >>> prob_list = count_dicts_to_prob_list_absolute_discounting(
    >>>     count_dicts, delta=0.5)
    >>> # gamma_0() = 0.5 * 7 / 27
    >>> # Pr(a) = (10 - 0.5) / 27 + 0.5 (7 / 27) (1 / 7) = 10 / 27
    >>> # BO(a) = gamma_1(a) = 0.5 * 4 / 9 = 2 / 9
    >>> prob_list[0]['a']  # (log10 Pr(a), log10 gamma_1(a))
    (-0.4313637641589874, -0.6532125137753437)
    >>> count_dicts[1][('a', 'n')]
    4
    >>> count_dicts[0]['n']
    4
    >>> sum(1 for k in count_dicts[2] if k[:2] == ('a', 'n'))
    2
    >>> sum(v for k, v in count_dicts[2].items() if k[:2] == ('a', 'n'))
    4
    >>> # Pr(n) = (4 - 0.5) / 27 + 0.5 (7 / 27) (1 / 7) = 4 / 27
    >>> # Pr(n|a) = (4 - 0.5) / 9 + gamma_1(a) Pr(n)
    >>> #         = (4 - 0.5) / 9 + 0.5 (4 / 9) (4 / 27)
    >>> #         = (108 - 13.5 + 8) / 243
    >>> #         = 102.5 / 243
    >>> # BO(a, n) = gamma_2(a, n) = 0.5 (2 / 4) = 1 / 4
    >>> prob_list[1][('a', 'n')]  # (log10 Pr(n|a), log10 gamma_2(a, n))
    (-0.37488240820653906, -0.6020599913279624)

    References
    ----------
    .. [chen1999] S. F. Chen and J. Goodman, "An empirical study of smoothing
       techniques for language modeling," Computer Speech & Language, vol. 13,
       no. 4, pp. 359-394, Oct. 1999, doi: 10.1006/csla.1999.0128.
    """
    if len(count_dicts) < 1:
        raise ValueError("At least unigram counts must exist")
    if not isinstance(delta, Iterable):
        delta = (delta,) * len(count_dicts)
    if len(delta) != len(count_dicts):
        raise ValueError(
            "Expected {} deltas, got {}".format(len(count_dicts), len(delta))
        )
    delta = tuple(
        _optimal_deltas(counts, 1) if d is None else [d]
        for (d, counts) in zip(delta, count_dicts)
    )
    return _absolute_discounting(count_dicts, delta, to_prune)


def count_dicts_to_prob_list_kneser_ney(
    count_dicts: CountDicts,
    delta=None,
    sos=None,
    to_prune=set(),
    temp: Optional[os.PathLike] = None,
):
    r"""Determine probabilities from counts using Kneser-Ney(-like) estimates

    Chen and Goodman's implemented Kneser-Ney smoothing [chen1999]_ is the same as
    absolute discounting, but with lower-order n-gram counts ((n-1)-grams, (n-2)-grams,
    etc.) replaced with adjusted counts:

    .. math::

        C'(w_1, \ldots, w_k) = \begin{cases}
            C(w_1, \ldots, w_k) & k = n \lor w_1 = sos \\
            \left|\left\{v : C(v, w_1, \ldots, w_k) > 0\right\}\right| & else\\
        \end{cases}

    The adjusted count is the number of unique prefixes the n-gram can occur with. We do
    not modify n-grams starting with the start-of-sequence `sos` token (as per
    [heafield2013]_) as they cannot have a preceding context.

    By default, modified Kneser-Ney is performed, which uses different absolute
    discounts for different adjusted counts:

    .. math::

        Pr_{KN}(w_1, \ldots, w_n) =
            \frac{C'(w_1, \ldots, w_n) - \delta(C'(w_1, \ldots, w_n))}
                 {\sum_{w'} C'(w_1, \ldots, w_{n-1}, w')}
            + \gamma(w_1, \ldots, w_{n-1}) Pr_{KN}(w_n|w_1, \ldots, w_{n-1})

    :math:`\gamma` are chosen so that :math:`Pr_{KN}(\cdot)` sum to one. As a base case,
    :math:`Pr_{KN}(\emptyset) = 1 / \left\|V\right\|`.

    Letting :math:`N_c` be defined as in
    :func:`count_dicts_to_prob_list_absolute_discounting`, and :math:`y = N_1 / (N_1 +
    2 N_2)`, the default value for :math:`\delta(\cdot)` is

    .. math::

        \delta(k) = k - (k + 1) y (N_{k + 1} / N_k)

    Where we set :math:`\delta(0) = 0` and :math:`\delta(>3) = \delta(3)`.

    Parameters
    ----------
    count_dicts : sequence
        A list of dictionaries. ``count_dicts[0]`` should correspond to unigram counts
        in a corpus, ``count_dicts[1]`` to bi-grams, etc. Keys are tuples of tokens
        (n-grams) of the appropriate length, with the exception of unigrams, whose keys
        are the tokens themselves. Values are the counts of those n-grams in the corpus.
    delta : float or tuple or :obj:`None`, optional
        The absolute discount to apply to non-zero values. `delta` may be a
        :class:`float`, at which point a fixed discount will be applied to all orders of
        the recursion. If :obj:`None`, the default values defined above will be
        employed. `delta` can be a :class:`tuple` of the same length as
        ``len(count_dicts)``, which can be used to specify discounts at each level of
        the recursion (excluding the zero-th order), unigrams first. If an element is a
        :class:`float`, that fixed discount will be applied to all nonzero counts at
        that order. If :obj:`None`, `delta` will be calculated in the default manner for
        that order. Finally, an element can be a :class:`tuple` itself of positive
        length. In this case, the elements of that tuple will correspond to the values
        of :math:`\delta(k)` where the i-th indexed element is :math:`\delta(i+1)`.
        Counts above the last :math:`\delta(k)` will use the same discount as the last
        :math:`\delta(k)`. Elements within the tuple can be either :class:`float`
        (use this value) or :obj:`None` (use defafult)
    sos : str or :obj:`None`, optional
        The start-of-sequence symbol. Defaults to ``'<S>'`` if that symbol is in the
        vocabulary, otherwise ``'<s>'``
    to_prune : set, optional
        A set of n-grams that will not be explicitly set in the return value. This
        differs from simply removing those n-grams from `count_dicts` in some key ways.
        First, nonzero counts of pruned n-grams are used when calculating adjusted
        counts of the remaining terms. Second, pruned counts can still be used to
        calculate default `delta` values. Third, as per [chen1999]_, pruned counts are
        still summed in the denominator, :math:`\sum_w' C(w_1, \ldots, w_{n-1}, w')`,
        which then make their way into the numerator of
        :math:`gamma(w_1, \ldots, w_{n-1})`.

    Returns
    -------
    prob_list : sequence
        Corresponding n-gram conditional probabilities. See
        :mod:`pydrobert.torch.util.parse_arpa_lm`

    Examples
    --------
    >>> from collections import Counter
    >>> text = 'a man a plan a canal panama'
    >>> count_dicts = [
    >>>     Counter(
    >>>         tuple(text[offs:offs + order]) if order > 1
    >>>         else text[offs:offs + order]
    >>>         for offs in range(len(text) - order + 1)
    >>>     )
    >>>     for order in range(1, 5)
    >>> ]
    >>> count_dicts[0]
    Counter({'a': 10, ' ': 6, 'n': 4, 'm': 2, 'p': 2, 'l': 2, 'c': 1})
    >>> adjusted_unigrams = dict(
    >>>     (k, sum(1 for kk in count_dicts[1] if kk[1] == k))
    >>>     for k in count_dicts[0]
    >>> )
    >>> adjusted_unigrams
    {'a': 6, ' ': 3, 'm': 2, 'n': 1, 'p': 1, 'l': 2, 'c': 1}
    >>> adjusted_bigrams = dict(
    >>>     (k, sum(1 for kk in count_dicts[2] if kk[1:] == k))
    >>>     for k in count_dicts[1]
    >>> )
    >>> adjusted_trigrams = dict(
    >>>     (k, sum(1 for kk in count_dicts[3] if kk[1:] == k))
    >>>     for k in count_dicts[2]
    >>> )
    >>> len(adjusted_unigrams)
    7
    >>> sum(adjusted_unigrams.values())
    16
    >>> sum(1 for k in adjusted_bigrams if k[0] == 'a')
    4
    >>> sum(v for k, v in adjusted_bigrams.items() if k[0] == 'a')
    7
    >>> prob_list = count_dicts_to_prob_list_kneser_ney(
    >>>     count_dicts, delta=.5)
    >>> # gamma_0() = 0.5 * 7 / 16
    >>> # Pr(a) = (6 - 0.5) / 16 + 0.5 * (7 / 16) * (1 / 7)
    >>> # Pr(a) = 3 / 8
    >>> # BO(a) = gamma_1(a) = 0.5 * 4 / 7 = 2 / 7
    >>> prob_list[0]['a']  # (log10 Pr(a), log10 BO(a))
    (-0.42596873227228116, -0.5440680443502757)
    >>> adjusted_bigrams[('a', 'n')]
    4
    >>> adjusted_unigrams['n']
    1
    >>> sum(1 for k in adjusted_trigrams if k[:2] == ('a', 'n'))
    2
    >>> sum(v for k, v in adjusted_trigrams.items() if k[:2] == ('a', 'n'))
    4
    >>> # Pr(n) = (1 - 0.5) / 16 + 0.5 (7 / 16) (1 / 7) = 1 / 16
    >>> # Pr(n|a) = (4 - 0.5) / 7 + gamma_1(a) Pr(n)
    >>> #         = (4 - 0.5) / 7 + (2 / 7) (1 / 16)
    >>> #         = (64 - 8 + 2) / 112
    >>> #         = 29 / 56
    >>> # BO(a, n) = gamma_2(a, n) = 0.5 (2 / 4) = 1 / 4
    (-0.2857900291072443, -0.6020599913279624)

    Notes
    -----
    As discussed in [chen1999]_, Kneser-Ney is usually formulated so that only unigram
    counts are adjusted. However, they themselves experiment with modified counts for
    all lower orders.

    References
    ----------
    .. [chen1999] S. F. Chen and J. Goodman, "An empirical study of smoothing techniques
       for language modeling," Computer Speech & Language, vol. 13, no. 4, pp. 359-394,
       Oct. 1999, doi: 10.1006/csla.1999.0128.
    .. [heafield2013] K. Heafield, I. Pouzyrevsky, J. H. Clark, and P. Koehn, "Scalable
       modified Kneser-Ney language model estimation," in Proceedings of the 51st Annual
       Meeting of the Association for Computational Linguistics, Sofia, Bulgaria, 2013,
       vol. 2, pp. 690-696.
    """
    if len(count_dicts) < 1:
        raise ValueError("At least unigram counts must exist")
    if not isinstance(delta, Iterable):
        delta = (delta,) * len(count_dicts)
    if len(delta) != len(count_dicts):
        raise ValueError(
            "Expected {} deltas, got {}".format(len(count_dicts), len(delta))
        )
    if sos is None:
        sos = "<S>" if "<S>" in count_dicts[0] else "<s>"

    if temp:

        class MyDict(DbNgramDict[int]):
            VALUE_FMT: Final[str] = "!Q"

        temp = Path(temp)
        acount_tmp = temp / "acount"
        acount_tmp.mkdir(exist_ok=True)

    new_count_dicts = [count_dicts[-1]]
    for order in range(len(count_dicts) - 2, -1, -1):
        if temp:
            new_counts = MyDict(
                get_store(acount_tmp / COUNTFILE_FMT_PREFIX.format(order=order + 1))
            )
        else:
            new_counts = dict()
        if order == 0:
            for ngram in count_dicts[order].keys():
                new_counts[ngram] = 0
        for ngram, count in count_dicts[order + 1].items():
            if not count:
                continue
            suffix = ngram[1:] if order else ngram[1]
            new_counts[suffix] = new_counts.get(suffix, 0) + 1
        new_counts.update(
            (k, v)
            for (k, v) in count_dicts[order].items()
            if ((order and k[0] == sos) or (not order and k == sos))
        )
        new_count_dicts.insert(0, new_counts)
    count_dicts = new_count_dicts
    delta = list(delta)
    for i in range(len(delta)):
        ds, counts = delta[i], count_dicts[i]
        if ds is None:
            ds = (None, None, None)
        if not isinstance(ds, Iterable):
            ds = (ds,)
        ds = tuple(ds)
        try:
            last_deft = len(ds) - ds[::-1].index(None) - 1
            optimals = _optimal_deltas(counts, last_deft + 1)
            assert len(optimals) == len(ds)
            ds = tuple(y if x is None else x for (x, y) in zip(ds, optimals))
        except ValueError:
            warnings.warn(f"Falling back to default discounts for {i+1}-th order")
            ds = FALLBACK_DELTAS
        delta[i] = ds
    return _absolute_discounting(count_dicts, delta, to_prune, temp)


def text_to_sents(
    text: str,
    sent_end_expr: Union[str, re.Pattern] = DEFT_SENT_END_EXPR,
    word_delim_expr: Union[str, re.Pattern] = DEFT_WORD_DELIM_EXPR,
    to_case: Literal["upper", "lower", None] = "upper",
    trim_empty_sents: bool = False,
) -> List[Tuple[str, ...]]:
    """Convert a block of text to a list of sentences, each a list of words

    Parameters
    ----------
    text : str
        The text to parse
    set_end_expr
        A regular expression indicating an end of a sentence. By default, this is one or
        more of the characters ".?!"
    word_delim_expr
        A regular expression used for splitting words. By default, it is one or more of
        any non-alphanumeric character (including ' and -). Any empty words are removed
        from the sentence
    to_case
        Convert all words to a specific case: ``'lower'`` is lower case, ``'upper'`` is
        upper case, anything else performs no conversion
    trim_empty_sents
        If :obj:`True`, any sentences with no words in them will be removed from the
        return value. The exception is an empty final string, which is always removed.

    Returns
    -------
    sents : list of tuple of str
        A list of sentences from `text`. Each sentence/element is actually a tuple of
        the words in the sentences
    """
    if not isinstance(sent_end_expr, re.Pattern):
        sent_end_expr = re.compile(sent_end_expr)
    sents = list(
        titer_to_siter(
            sent_end_expr.split(text), word_delim_expr, to_case, trim_empty_sents
        )
    )
    if sents and not sents[-1]:
        del sents[-1]
    return sents


def titer_to_siter(
    titer: Iterable[str],
    word_delim_expr: Union[str, re.Pattern] = DEFT_WORD_DELIM_EXPR,
    to_case: Literal["lower", "upper", None] = "upper",
    trim_empty_sents: bool = False,
) -> Generator[Tuple[str, ...], None, None]:
    if not isinstance(word_delim_expr, re.Pattern):
        word_delim_expr = re.compile(word_delim_expr)
    for sent in titer:
        sent = word_delim_expr.split(sent)
        sent = tuple(w for w in sent if w)
        if to_case == "lower":
            sent = tuple(w.lower() for w in sent)
        elif to_case == "upper":
            sent = tuple(w.upper() for w in sent)
        if trim_empty_sents and not sent:
            continue
        yield sent


def sents_to_count_dicts(
    sents: Iterable[str],
    N: Union[int, List[DbCountDict]],
    sos: str = "<S>",
    eos: str = "</S>",
    count_unigram_sos: bool = False,
) -> CountDicts:
    """Count n-grams in sentence lists up to a maximum order

    Parameters
    ----------
    sents
        A list of sentences, where each sentence is a tuple of its words.
    N
        Either the maximum order (inclusive) of n-gram to count, or a list of
        DbCountDict to populate of the same format as `count_dicts`. In the latter
        case, counts will be added to these dictionaries and returned.
    sos
        A token representing the start-of-sequence.
    eos
        A token representing the end-of-sequence.
    count_unigram_sos
        If :obj:`False`, the unigram count of the start-of-sequence token will always be
        zero (though higher-order n-grams beginning with the SOS can have counts).

    Returns
    -------
    count_dicts : list of dicts
        A list of the same length as the maximum order where ``count_dicts[0]`` is a
        dictionary of unigram counts, ``count_dicts[1]`` of bigram counts, etc.

    Notes
    -----
    The way n-grams count start-of-sequence and end-of-sequence tokens differ from
    package to package. For example, some tools left-pad sentences with n - 1
    start-of-sequence tokens (e.g. making ``Pr(x|<s><s>)`` a valid conditional). Others
    only count sos and eos tokens for n > 1.

    This function adds one (and only one) sos and eos to the beginning and end of each
    sentence before counting n-grams with only one exception. By default,
    `count_unigram_sos` is set to :obj:`False`, meaning the start-of-sequence token will
    not be counted as a unigram. This makes sense in the word prediction context since a
    language model should never predict the next word to be the start-of-sequence token.
    Rather, it always exists prior to the first word being predicted. This exception can
    be disabled by setting `count_unigram_sos` to :obj:`True`.

    See Also
    --------
    shelve
        A module for creating disk-backed mutable dictionaries. Use as `N` to decrease
        memory load.
    """
    if not isinstance(N, int):
        count_dicts, N = N, len(N)
    else:
        count_dicts = [Counter() for _ in range(N)]
    if N < 1:
        raise ValueError("max order must be >= 1")
    count_dicts[0].setdefault(sos, 0)
    for i, sent in enumerate(sents):
        if {sos, eos} & set(sent):
            raise ValueError(
                "start-of-sequence ({}) or end-of-sequence ({}) found in "
                'sentence "{}"'.format(sos, eos, " ".join(sent))
            )
        sent = (sos,) + tuple(sent) + (eos,)
        for order, counter in zip(range(1, N + 1), count_dicts):
            if order == 1:
                counter.update(sent if count_unigram_sos else sent[1:])
            else:
                counter.update(
                    sent[s : s + order] for s in range(len(sent) - order + 1)
                )
        if (i + 1) % SENTS_PER_INFO == 0:
            logging.info(f"Processed {i + 1} sentences")
    return count_dicts


def _pos_int(val):
    val = int(val)
    if val < 1:
        raise ValueError("value not positive")
    return val


def _nonneg_int(val):
    val = int(val)
    if val < 0:
        raise ValueError("value negative")
    return val


def main(args: Optional[Sequence[str]] = None):
    """Construct an n-gram LM

    Convenient, but slow. You should prefer KenLM (https://github.com/kpu/kenlm).

    Example call (5-gram, modified Kneser-Ney, hapax >1-gram pruning):

        python ngram_lm.py -o 5 -t 0 1 | gzip -c > lm.arpa.gz < train.txt.gz
    """
    logging.captureWarnings(True)

    parser = argparse.ArgumentParser(
        description=main.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Smoothing methods (--methods, -m):

- abs: absolute discounting
    Special flags:
      --delta,-d: Absolute discount to apply to non-zero values. A single value will
                  be applied to all orders; otherwise, one discount per order
- add-k: add k to counts, then mle
    Speckal flags:
      --k,-k: The amount to add
- katz: Katz backoff
      --katz-threshold: Discounting will not be applied above this threshold
- kn: modified Kneser-Ney
      --delta,-d: Absolute discount to apply to non-zero values. A single value will
                  be applied to all orders; otherwise, one discount per order. Note:
                  specifying more than one value per order is not supported by this
                  flag, whereas the default 
- sgt: simple Good-Turing
- mle: maximum likelihood estimate (no smoothing; no backoff)
""",
    )
    parser.add_argument(
        "--max-order",
        "-o",
        metavar="P_INT",
        type=_pos_int,
        default=6,
        help="Max order of n-gram in model",
    )
    parser.add_argument(
        "--in-file",
        "-f",
        metavar="PTH",
        type=argparse.FileType("rb"),
        default=argparse.FileType("rb")("-"),
        help="If specified, reads from this file instead of stdin",
    )
    parser.add_argument(
        "--out-file",
        "-O",
        metavar="PTH",
        type=argparse.FileType("wb"),
        default=None,
        help="If specified, writes to this file instead of stdout",
    )
    parser.add_argument(
        "--sos", "-s", metavar="STR", default="<s>", help="Start-of-sequence token"
    )
    parser.add_argument(
        "--eos", "-e", metavar="STR", default="</s>", help="End-of-sequence token"
    )
    parser.add_argument(
        "--compress",
        "-c",
        action="store_true",
        default=False,
        help="If set, write gzip-compressed format instead of plaintext",
    )
    parser.add_argument(
        "--sent-end-expr",
        metavar="REGEX",
        type=re.compile,
        default=None,
        help="If specified, the entire text file will be read at once, then split into "
        "sentences according to this expression",
    )
    parser.add_argument(
        "--word-delim-expr",
        metavar="REGEX",
        type=re.compile,
        default=DEFT_WORD_DELIM_EXPR,
        help="Delimiter to split sentences into tokens",
    )
    parser.add_argument(
        "--to-case",
        choices=("upper", "lower"),
        default=None,
        help="Convert all tokens to either upper or lower case",
    )
    parser.add_argument(
        "--trim-empty-sents",
        action="store_true",
        default=False,
        help="If specified, remove any empty sentences from counting",
    )
    parser.add_argument(
        "--count-unigram-sos",
        action="store_true",
        default=False,
        help="If specified, collect unigram counts of start-of-sequence-token",
    )
    parser.add_argument(
        "--prune-by-lprob-threshold",
        metavar="FLOAT",
        type=float,
        default=None,
        help="If specified, remove any n-grams with log-prob lte this threshold",
    )
    parser.add_argument(
        "--prune-by-entropy-threshold",
        metavar="FLOAT",
        type=float,
        default=None,
        help="If specified, prune by SRI's relative entropy criterion",
    )
    parser.add_argument(
        "--prune-by-count-thresholds",
        "-t",
        metavar="NN_INT",
        type=_nonneg_int,
        nargs="+",
        default=None,
        help="If specified, remove any n-grams with lte this count. If there are k "
        "counts specified, the first k-1 counts will apply to the first k-1 orders of "
        "n-gram, while the last count covers the remaing orders. E.g. 0 1 will not "
        "prune unigrams but prune everything above with a count <= 1. Note that 0 will "
        "not prune count-0 terms.",
    )
    parser.add_argument(
        "--temp-dir",
        "-T",
        metavar="PTH",
        nargs="?",
        const=1,
        help="If specified, counts and probs will be stored to files in this directory "
        "in order to reduce memory pressure. If the flag is passed an argument, count "
        "files will be stored in the provided directory. If count files up to the "
        "maximum order have been previously completed and are stored in this "
        "directory, those counts will be used instead of processing the input. If the "
        "flag is not passed an argument, a temporary directory will be used",
    )
    parser.add_argument(
        "--max-cache-age-seconds",
        metavar="SEC",
        type=float,
        default=DEFT_MAX_CACHE_AGE_SECONDS,
        help="When --count-directory/-T is specified, how many seconds before cache "
        "elements expire",
    )
    parser.add_argument(
        "--max-cache-len",
        metavar="SEC",
        type=float,
        default=DEFT_MAX_CACHE_LEN,
        help="When --count-directory/-T is specified, the maximum number of elements "
        "which can exist in the cache of a single order of n-gram counts",
    )
    parser.add_argument(
        "--eps-lprob",
        metavar="FLOAT",
        type=float,
        default=DEFT_EPS_LPROB,
        help="Log-probability considered approx. 0",
    )
    parser.add_argument(
        "--delta",
        "-d",
        metavar="FLOAT",
        nargs="+",
        type=float,
        default=None,
        help="See epilogue",
    )
    parser.add_argument(
        "--k",
        "-k",
        metavar="FLOAT",
        type=float,
        default=DEFT_ADD_K_K,
        help="See epilogue",
    )
    parser.add_argument(
        "--katz-threshold",
        metavar="PINT",
        type=_pos_int,
        default=DEFAULT_KATZ_THRESH,
        help="See epilogue",
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=["abs", "add-k", "katz", "kn", "sgt", "mle"],
        default="kn",
        help="Smoothing method (see epilogue)",
    )
    prune = parser.add_mutually_exclusive_group()
    prune.add_argument(
        "--prune-by-name-list",
        metavar="NGRAM",
        nargs="+",
        default=None,
        help="Each argument to this flag is an n-gram (first word to last) to be "
        "pruned. Same word delimiters as text",
    )
    prune.add_argument(
        "--prune-by-name-file",
        metavar="FILE",
        type=argparse.FileType("r"),
        default=None,
        help="A file storing n-grams (first word to last) to be pruned as a series of "
        "sentences. Same sentence and word delimeters as text",
    )
    parser.add_argument("--verbose", "-v", action="store_true", default=False)

    options = parser.parse_args(args)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO if options.verbose else logging.WARNING,
    )

    if options.prune_by_name_file is not None:
        if options.sent_end_expr is None:
            prune_names = (x.rstrip("\n") for x in options.prune_by_name_file)
        else:
            prune_names = options.sent_end_expr.split(options.prune_by_name_file.read())
    elif options.prune_by_name_list is not None:
        prune_names = options.prune_by_name_list
    else:
        prune_names = []
    prune_names = (tuple(options.word_delim_expr.split(x)) for x in prune_names if x)
    prune_names = {x[0] if len(x) == 1 else x for x in prune_names}

    temp_dir, N, count_dicts = options.temp_dir, options.max_order, None
    if temp_dir is not None:
        if temp_dir == 1:
            temp_dir_ = TemporaryDirectory()
            temp_dir = os.fspath(temp_dir_.name)
        os.makedirs(temp_dir, exist_ok=True)
        logging.info(f"Count directory set to '{temp_dir}'")
        count_prefixes = []
        all_complete = True
        for n in range(1, N + 1):
            count_prefix = os.path.join(temp_dir, COUNTFILE_FMT_PREFIX.format(order=n))
            complete_file = count_prefix + COMPLETE_SUFFIX
            complete_file_exists = os.path.exists(complete_file)
            all_complete &= complete_file_exists
            count_prefixes.append(count_prefix)
        if all_complete:
            logging.info("Counts already done! Just reading")
            # fire-hose read. Bad for cache
            count_dicts = [
                open_count_dict(p, "r", max_cache_len=1, max_cache_age_seconds=1)
                for p in count_prefixes
            ]
        else:
            N = []
            for count_prefix in count_prefixes:
                if os.path.exists(count_prefix + COMPLETE_SUFFIX):
                    os.unlink(count_prefix + COMPLETE_SUFFIX)
                N.append(
                    open_count_dict(
                        count_prefix,
                        "n",
                        max_cache_len=options.max_cache_len,
                        max_cache_age_seconds=options.max_cache_age_seconds,
                    )
                )

    if count_dicts is None:
        if options.in_file.peek()[:2] == b"\x1f\x8b":
            options.in_file = gzip.open(options.in_file, "rt")
        else:
            options.in_file = TextIOWrapper(options.in_file)
        if options.sent_end_expr is None:
            sents = (x.rstrip("\n") for x in options.in_file)
        else:
            sents = options.sent_end_expr.split(options.in_file.read())
        sents = titer_to_siter(
            sents,
            options.word_delim_expr,
            options.to_case,
            options.trim_empty_sents,
        )

        logging.info("Beginning to count n-grams")
        count_dicts = sents_to_count_dicts(
            sents, N, options.sos, options.eos, options.count_unigram_sos
        )
        logging.info("Done counting n-grams")
        del sents

        if temp_dir is not None:
            for count_prefix in count_prefixes:
                Path(count_prefix + COMPLETE_SUFFIX).touch()
    del N

    if options.prune_by_count_thresholds is not None:
        k = len(options.prune_by_count_thresholds) - 1
        for i, counts in enumerate(count_dicts):
            t = options.prune_by_count_thresholds[min(i, k)]
            if i == 0 and t > 0:
                logging.warning(
                    "Possibly pruning unigrams. You probably don't want this"
                )
            elif t > 0:
                logging.info(f"Pruning {i + 1}-grams with threshold {t}")
            else:
                continue
            prune_names.update(k for (k, v) in counts.items() if v <= t)
    logging.info(f"{len(prune_names)} n-grams will be pruned")

    if options.method == "abs":
        logging.info("Performing absolute discounting")
        if options.delta is not None and len(options.delta) == 1:
            options.delta = options.delta[0]
        prob_list = count_dicts_to_prob_list_absolute_discounting(
            count_dicts, options.delta, prune_names
        )
        prune_names = set()
    elif options.method == "add-k":
        logging.info("Performing add-k smoothing")
        prob_list = count_dicts_to_prob_list_add_k(
            count_dicts, options.eps_lprob, options.k
        )
    elif options.method == "katz":
        logging.info("Performing Katz backoff")
        prob_list = count_dicts_to_prob_list_katz_backoff(
            count_dicts, options.katz_threshold, options.eps_lprob
        )
    elif options.method == "kn":
        logging.info("Performing modified Kneser-Ney smoothing")
        if options.delta is not None and len(options.delta) == 1:
            options.delta = options.delta[0]
        prob_list = count_dicts_to_prob_list_kneser_ney(
            count_dicts, options.delta, options.sos, prune_names, temp_dir
        )
        prune_names = set()
    elif options.method == "sgt":
        logging.info("Performing Simple Good-Turing smoothing")
        prob_list = count_dicts_to_prob_list_simple_good_turing(
            count_dicts, options.eps_lprob
        )
    else:  # mle
        logging.info("Taking the MLE (no smoothing)")
        prob_list = count_dicts_to_prob_list_mle(count_dicts, options.eps_lprob)
    logging.info("Done collecting probabilities")
    if temp_dir is not None:
        for counts in count_dicts:
            counts.close()
    del count_dicts

    if (
        options.prune_by_entropy_threshold is not None
        or options.prune_by_lprob_threshold is not None
        or prune_names
    ):
        logging.info("Pruning necessary. Constructing trie")
        ngram_lm = BackoffNGramLM(prob_list, options.sos, options.eos)
        del prob_list
        if prune_names:
            logging.info("Pruning by name")
            ngram_lm.prune_by_name(prune_names, options.eps_lprob)
        if options.prune_by_lprob_threshold is not None:
            logging.info(
                "Pruning by log-probability threshold "
                f"{options.prune_by_lprob_threshold}"
            )
            ngram_lm.prune_by_threshold(options.prune_by_lprob_threshold)
        if options.prune_by_entropy_threshold is not None:
            logging.info(
                "Pruning by relative entropy threshold "
                f"{options.prune_by_entropy_threshold}"
            )
            ngram_lm.relative_entropy_pruning(options.prune_by_entropy_threshold)
        logging.info("Renormalizing backoffs")
        ngram_lm.renormalize_backoffs()
        logging.info("Reconstructing log probabilities from trie")
        prob_list = ngram_lm.to_prob_list()
        del ngram_lm

    logging.info(f"Writing out LM in arpa format")
    if options.out_file is None:
        if options.compress:
            raise ValueError(
                "compressed output does not work nicely with stdout. Pipe to gzip -c "
                "if you want this"
            )
        else:
            options.out_file = sys.stdout
    else:
        if options.compress:
            options.out_file = gzip.open(options.out_file, "wt")
        else:
            options.out_file = TextIOWrapper(options.out_file)
    write_arpa(prob_list, options.out_file)
    logging.info("Done")


if __name__ == "__main__":
    sys.exit(main())
