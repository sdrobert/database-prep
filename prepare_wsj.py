#! /usr/bin/env python

# Copyright 2019 Sean Robertson
#
# Adapted from kaldi/egs/wsj/s5/local/wsj_data_prep.sh:
#
# Copyright 2009-2012  Microsoft Corporation
#                      Johns Hopkins University (Author: Daniel Povey)
#
# and kaldi/egs/wsj/s5/local/{ndx2flist.pl,flist2scp.pl,find_transcripts.pl,
# normalize_transcript.pl}, kaldi/egs/wsj/s5/local/steps/utt2spk_to_spk2utt.pl:
#
# Copyright 2010-2011 Microsoft Corporation
#
# and kaldi/egs/wsj/s5/local/wsj_prepare_dict.sh:
#
# Copyright 2010-2012 Microsoft Corporation
#           2012-2014 Johns Hopkins University (Author: Daniel Povey)
#                2015 Guoguo Chen
#
# and kaldi/egs/wsj/s5/local/wsj_prepare_char_dict.sh
#
# Copyright 2017  Hossein Hadian
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

# some relevant files to the setup:
# https://catalog.ldc.upenn.edu/docs/LDC93S6A/readme.txt
# https://catalog.ldc.upenn.edu/docs/LDC94S13A/wsj1.txt


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import sys
import argparse
import re
import warnings
import locale
import gzip
import itertools

from collections import OrderedDict

import ngram_lm

from common import glob, mkdir, sort, cat, pipe_to, wc_l
from pydrobert.torch.util import parse_arpa_lm

try:
    import urllib.request as request
except ImportError:
    import urllib2 as request

from unlzw import unlzw

__author__ = "Sean Robertson"
__email__ = "sdrobert@cs.toronto.edu"
__license__ = "Apache 2.0"
__copyright__ = "Copyright 2020 Sean Robertson"


locale.setlocale(locale.LC_ALL, 'C')


def find_link_dir(wsj_subdirs, rel_path, required=True):
    '''find rel_path as suffix in wsj_subdirs, return it or None

    Kaldi makes soft links of all `wsj_subdirs` in a ``links/`` directory.
    `rel_path` is some path that would be in the run script as
    ``links/<rel_path>``, which means it should *really* exist as
    ``<wsj_subdir>/<rel_path>``, where ``wsj_subdir`` is an element of
    `wsj_subdirs` This removes the need to create symbolic links in a
    ``links/`` directory. Unix-style.
    '''
    for wsj_subdir in wsj_subdirs:
        dir_ = os.path.dirname(wsj_subdir)
        path = os.path.join(dir_, rel_path).replace(os.sep, '/')
        if (
                path.startswith(wsj_subdir.replace(os.sep, '/')) and
                os.path.exists(path)):
            return path
    if required:
        raise ValueError(
            '{} does not exist in {}'.format(rel_path).format(wsj_subdirs))
    return None


def ndx2flist(in_stream, wsj_subdirs):
    '''yields absolute file names from .ndx files'''

    dir_pattern = re.compile(r'.+/([0-9.-]+)/?$')
    disk2fn = dict()
    for fn in wsj_subdirs:
        fn = fn.replace(os.sep, '/')  # ensure unix-style file paths
        match = dir_pattern.match(fn)
        if match is None:
            raise ValueError(
                "Bad command-line argument {}".format(fn))
        disk_id = match.group(1)
        disk_id = disk_id.replace('.', '_').replace('-', '_')
        if fn.endswith(os.sep):
            fn = disk_id[:-1]
        disk2fn[disk_id] = fn

    line_pattern = re.compile(r'^([0-9_]+):\s*(\S+)$')
    for line_no, line in enumerate(in_stream):
        if line.startswith(';'):
            continue
        match = line_pattern.match(line)
        if match is None:
            raise ValueError("Could not parse line {}".format(line_no + 1))
        disk, filename = match.groups()
        if disk not in disk2fn:
            raise ValueError("Disk id {} not found".format(disk))
        yield "{}/{}".format(disk2fn[disk], filename)


def flist2scp(path):
    '''Yields <utt> <path> pairs from list of files'''
    line_pattern = re.compile(r'^\S+/(\w+)\.[wW][vV]1$')
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            match = line_pattern.match(line)
            if match is None:
                raise ValueError("Bad line {}".format(line))
            id_ = match.group(1).lower()
            yield "{} {}".format(id_, line)


def find_transcripts(in_stream, dot_flist):
    '''Yields <utt> <transcript> pairs from utts (stream) and transcripts'''

    spk2dot = dict()
    dot_pattern = re.compile(r'^\S+/(\w{6})00\.dot')
    with open(dot_flist) as f:
        for line in f:
            line = line.rstrip()
            match = dot_pattern.match(line)
            if match is None:
                raise ValueError('Bad line in dot file list {}'.format(line))
            spk = match.group(1)
            spk2dot[spk] = line

    utt_pattern = re.compile(r'^(\w{6})\w\w$')
    trans_pattern = re.compile(r'^(.+)\((\w{8})\)$')
    curspk = dotfile = None
    for uttid in in_stream:
        match = utt_pattern.match(uttid)
        if match is None:
            raise ValueError('Bad utterance id {}'.format(uttid))
        spk = match.group(1)
        if spk != curspk:
            utt2trans = dict()
            if spk not in spk2dot:
                raise ValueError('No dot file for speaker {}'.format(spk))
            dotfile = spk2dot[spk]
            with open(dotfile) as f:
                for line_no, line in enumerate(f):
                    line = line.rstrip()
                    match = trans_pattern.match(line)
                    if match is None:
                        raise ValueError(
                            'Bad line {} in dot file {} (line {})'
                            ''.format(line, dotfile, line_no + 1))
                    trans, utt = match.groups()
                    utt2trans[utt] = trans
            if uttid not in utt2trans:
                warnings.error(
                    'No transcript for utterance id {} (current dot file is '
                    '{})'.format(uttid, dotfile))
            yield "{} {}".format(uttid, utt2trans[uttid])


def normalize_transcript(in_stream, noiseword):
    '''Sanitize in_stream transcripts'''
    line_pattern = re.compile(r'^(\S+) (.+)$')
    del_pattern = re.compile(r'^([.~]|\[[</]\w+\]|\[\w+[>/]\])$')
    noise_pattern = re.compile(r'^\[\w+\]$')
    verbdel_pattern = re.compile(r"^<([\w']+)>$")
    for line in in_stream:
        match = line_pattern.match(line)
        if match is None:
            raise ValueError("Bad line {}".format(line))
        out, trans = match.groups()
        for w in trans.split(' '):
            w = w.upper().replace('\\', '')
            if del_pattern.match(w):
                continue
            elif w == '%PERCENT':
                w = 'PERCENT'
            elif w == '.POINT':
                w = 'POINT'
            elif w == '--DASH':
                w = '-DASH'
            elif noise_pattern.match(w):
                w = noiseword
            else:
                match = verbdel_pattern.match(w)
                if match:
                    w = match.group(1)
            out += " " + w
        yield out


def utt2spk_to_spk2utt(in_stream):
    spk_hash = OrderedDict()
    for line in in_stream:
        utt, spk = line.split(' ')
        spk_hash.setdefault(spk, []).append(utt)
    for spk, utts in spk_hash.items():
        yield "{} {}".format(spk, ' '.join(utts))


def wsj_data_prep(wsj_subdirs, data_root):
    # this follows part of kaldi/egs/wsj/s5/local/wsj_data_prep.sh, but factors
    # out the language modelling stuff to wsj_word_prep in case we don't care
    # about words

    dir_ = os.path.join(data_root, 'local', 'data')
    mkdir(dir_)

    wv1_pattern = re.compile(r'\.wv1$', flags=re.I)

    for rel_path in {'11-13.1', '13-34.1', '11-2.1'}:
        if find_link_dir(wsj_subdirs, rel_path, required=False) is None:
            raise ValueError('''\
wsj_data_prep: Spot check of command line arguments failed
Command line arguments must be absolute pathnames to WSJ directories
with names like 11-13.1
''')

    train_si84_flist = os.path.join(dir_, 'train_si84.flist')
    pipe_to(
        (
            x for x in sort(ndx2flist(
                cat(find_link_dir(
                    wsj_subdirs, '11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx',
                )),
                wsj_subdirs,
            ))
            if re.search(r'11-2.1/wsj0/si_tr_s/401', x, flags=re.I) is None
        ),
        train_si84_flist,
    )
    nl = wc_l(cat(train_si84_flist))
    if nl != 7138:
        warnings.warn(
            'expected 7138 lines in train_si84.flist, got {}'.format(nl))

    train_si284_flist = os.path.join(dir_, 'train_si284.flist')
    pipe_to(
        (
            x for x in sort(ndx2flist(
                cat(
                    find_link_dir(
                        wsj_subdirs,
                        '13-34.1/wsj1/doc/indices/si_tr_s.ndx',
                    ),
                    find_link_dir(
                        wsj_subdirs,
                        '11-13.1/wsj0/doc/indices/train/tr_s_wv1.ndx',
                    ),
                ),
                wsj_subdirs,
            ))
            if re.search(r'11-2.1/wsj0/si_tr_s/401', x, flags=re.I) is None
        ),
        train_si284_flist,
    )
    nl = wc_l(cat(train_si284_flist))
    if nl != 37416:
        warnings.warn(
            'expected 37416 lines in train_si284.flist, got {}'.format(nl))

    test_eval92_flist = os.path.join(dir_, 'test_eval92.flist')
    pipe_to(
        (
            x + '.wv1' for x in sort(ndx2flist(
                cat(find_link_dir(
                    wsj_subdirs,
                    '11-13.1/wsj0/doc/indices/test/nvp/si_et_20.ndx',
                )),
                wsj_subdirs,
            ))
        ),
        test_eval92_flist,
    )
    nl = wc_l(cat(test_eval92_flist))
    if nl != 333:
        warnings.warn(
            'expected 333 lines in test_eval92.flist, got {}'.format(nl))

    test_eval92_5k_flist = os.path.join(dir_, 'test_eval92_5k.flist')
    pipe_to(
        (
            x + '.wv1' for x in sort(ndx2flist(
                cat(find_link_dir(
                    wsj_subdirs,
                    '11-13.1/wsj0/doc/indices/test/nvp/si_et_05.ndx',
                )),
                wsj_subdirs,
            ))
        ),
        test_eval92_5k_flist,
    )
    nl = wc_l(cat(test_eval92_5k_flist))
    if nl != 330:
        warnings.warn(
            'expected 330 lines in test_eval92_5k.flist, got {}'.format(nl))

    test_eval93_flist = os.path.join(dir_, 'test_eval93.flist')
    pipe_to(
        sort(ndx2flist(
            (x.replace('13_32_1', '13_33_1') for x in cat(find_link_dir(
                wsj_subdirs,
                '13-32.1/wsj1/doc/indices/wsj1/eval/h1_p0.ndx',
            ))),
            wsj_subdirs,
        )),
        test_eval93_flist,
    )
    nl = wc_l(cat(test_eval93_flist))
    if nl != 213:
        warnings.warn(
            'expected 213 lines in test_eval93.flist, got {}'.format(nl))

    test_eval93_5k_flist = os.path.join(dir_, 'test_eval93_5k.flist')
    pipe_to(
        sort(ndx2flist(
            (x.replace('13_32_1', '13_33_1') for x in cat(find_link_dir(
                wsj_subdirs,
                '13-32.1/wsj1/doc/indices/wsj1/eval/h2_p0.ndx',
            ))),
            wsj_subdirs,
        )),
        test_eval93_5k_flist,
    )
    nl = wc_l(cat(test_eval93_5k_flist))
    if nl != 215:
        # I've found 215 entries, unlike the kaldi_data_prep.sh which suggests
        # 213. I've verified using kaldi_data_prep.sh
        warnings.warn(
            'expected 215 lines in test_eval93_5k.flist, got {}'.format(nl))

    test_dev93_flist = os.path.join(dir_, 'test_dev93.flist')
    pipe_to(
        sort(ndx2flist(
            cat(find_link_dir(
                wsj_subdirs,
                '13-34.1/wsj1/doc/indices/h1_p0.ndx',
            )),
            wsj_subdirs,
        )),
        test_dev93_flist,
    )
    nl = wc_l(cat(test_dev93_flist))
    if nl != 503:
        warnings.warn(
            'expected 503 lines in test_dev93.flist, got {}'.format(nl))

    test_dev93_5k_flist = os.path.join(dir_, 'test_dev93_5k.flist')
    pipe_to(
        sort(ndx2flist(
            cat(find_link_dir(
                wsj_subdirs,
                '13-34.1/wsj1/doc/indices/h2_p0.ndx',
                required=True,
            )),
            wsj_subdirs,
        )),
        test_dev93_5k_flist,
    )
    nl = wc_l(cat(test_dev93_5k_flist))
    if nl != 513:
        warnings.warn(
            'expected 513 lines in test_dev93_5k.flist, got {}'.format(nl))

    dir_13_16_1 = find_link_dir(wsj_subdirs, '13-16.1')
    dev_dt_20_flist = os.path.join(dir_, 'dev_dt_20.flist')
    dev_dt_05_flist = os.path.join(dir_, 'dev_dt_05.flist')
    pipe_to(
        sort((
            x for x in glob(dir_13_16_1, r'???1/??_??_20/**/*')
            if wv1_pattern.search(x)
        )),
        dev_dt_20_flist,
    )
    pipe_to(
        sort((
            x for x in glob(dir_13_16_1, r'???1/??_??_05/**/*')
            if wv1_pattern.search(x)
        )),
        dev_dt_05_flist,
    )

    dot_files_flist = os.path.join(dir_, 'dot_files.flist')
    pipe_to(
        itertools.chain(*(
            (
                y for y in glob(x, '**/*')
                if re.search(r'\.dot$', y, flags=re.I)
            ) for x in wsj_subdirs
        )),
        dot_files_flist,
    )

    noiseword = "<NOISE>"
    for x in {
            'train_si84', 'train_si284', 'test_eval92',
            'test_eval93', 'test_dev93', 'test_eval92_5k',
            'test_eval93_5k', 'test_dev93_5k', 'dev_dt_05',
            'dev_dt_20'}:
        src = os.path.join(dir_, x + '.flist')
        sph = os.path.join(dir_, x + '_sph.scp')
        trans1 = os.path.join(dir_, x + '.trans1')
        txt = os.path.join(dir_, x + '.txt')
        utt2spk = os.path.join(dir_, x + '.utt2spk')
        spk2utt = os.path.join(dir_, x + '.spk2utt')

        pipe_to(sort(flist2scp(src)), sph)
        pipe_to(
            find_transcripts(
                (x.split()[0] for x in cat(sph)), dot_files_flist),
            trans1
        )

        pipe_to(sort(normalize_transcript(cat(trans1), noiseword)), txt)

        # XXX(sdrobert): don't care about _wav.scp

        pipe_to(
            (
                '{} {}'.format(y, y[:3])
                for y in (x.split()[0] for x in cat(trans1))),
            utt2spk
        )

        pipe_to(utt2spk_to_spk2utt(cat(utt2spk)), spk2utt)

    spkrinfo = os.path.join(dir_, 'wsj0-train-spkrinfo.txt')
    if not os.path.isfile(spkrinfo):
        request.urlretrieve(
            'https://catalog.ldc.upenn.edu/docs/LDC93S6A/'
            'wsj0-train-spkrinfo.txt',
            spkrinfo
        )

    spk2gender = os.path.join(dir_, 'spk2gender')
    pipe_to(
        sort(set(
            ' '.join(x.lower().split()[:2])
            for x in cat(
                find_link_dir(wsj_subdirs, '11-13.1/wsj0/doc/spkrinfo.txt'),
                find_link_dir(
                    wsj_subdirs, '13-32.1/wsj1/doc/evl_spok/spkrinfo.txt'),
                find_link_dir(
                    wsj_subdirs, '13-34.1/wsj1/doc/dev_spok/spkrinfo.txt'),
                find_link_dir(
                    wsj_subdirs, '13-34.1/wsj1/doc/train/spkrinfo.txt'),
                spkrinfo
            )
            if not x.startswith(';') and not x.startswith('--')
        )),
        spk2gender
    )


def wsj_word_prep(wsj_subdirs, data_root, max_order=3):
    # We're doing things differently from Kaldi.
    # The NIST language model probabilities are really messed up. We train
    # up our own using Modified Kneser-Ney, but the same vocabulary that they
    # use.

    dir_13_32_1 = find_link_dir(wsj_subdirs, '13-32.1')
    base_lm_dir = os.path.join(
        dir_13_32_1, 'wsj1', 'doc', 'lng_modl', 'base_lm')
    lmdir = os.path.join(data_root, 'local', 'word_lm')
    cleaned_txt_gz = os.path.join(lmdir, 'cleaned.txt.gz')
    train_data_root = os.path.join(
        dir_13_32_1, 'wsj1', 'doc', 'lng_modl', 'lm_train', 'np_data')
    vocab2id_5_txt = os.path.join(lmdir, 'vocab2id_5.txt')
    vocab2id_20_txt = os.path.join(lmdir, 'vocab2id_20.txt')
    toprune_5_txt_gz = os.path.join(lmdir, 'toprune_5.txt.gz')
    toprune_20_txt_gz = os.path.join(lmdir, 'toprune_20.txt.gz')
    lm_5_arpa_gz = os.path.join(lmdir, 'lm_5.arpa.gz')
    lm_20_arpa_gz = os.path.join(lmdir, 'lm_20.arpa.gz')

    mkdir(lmdir)

    # get 5k open vocabulary from bcb05onp.z
    # bcb05onp.z is not gzipped, it's lzw zipped.
    # with open(os.path.join(base_lm_dir, 'bcb05onp.z'), 'rb') as file_:
    #     compressed = file_.read()
    # with io.StringIO(unlzw(compressed).decode('utf-8')) as file_:
    #     prob_list = parse_arpa_lm(file_)
    # prob_list[0]['<NOISE>'] = 0
    # vocab_5 = sorted(prob_list[0])
    # del prob_list
    # with open(vocab2id_5_txt, 'w') as file_:
    #     for i, v in enumerate(vocab_5):
    #         file_.write('{} {}\n'.format(v, i))
    # vocab_5 = set(vocab_5)

    # Get 20k open vocabulary from bcb20onp.z.
    # with open(os.path.join(base_lm_dir, 'bcb20onp.z'), 'rb') as file_:
    #     compressed = file_.read()
    # with io.StringIO(unlzw(compressed).decode('utf-8')) as file_:
    #     prob_list = parse_arpa_lm(file_)
    # prob_list[0]['<NOISE>'] = 0
    # vocab_20 = sorted(prob_list[0])
    # del prob_list
    # with open(vocab2id_20_txt, 'w') as file_:
    #     for i, v in enumerate(vocab_20):
    #         file_.write('{} {}\n'.format(v, i))
    # vocab_20 = set(vocab_20)

    # clean up training data. We do something similar to wsj_extend_dict.sh
    # assert os.path.isdir(train_data_root)
    # train_data_files = []
    # for subdir in ('87', '88', '89'):
    #     train_data_files.extend(
    #         glob(os.path.join(train_data_root, subdir), r'*.z'))
    # isword = vocab_5 | vocab_20
    # with gzip.open(cleaned_txt_gz, 'wt') as out:
    #     for train_data_file in train_data_files:
    #         with open(train_data_file, 'rb') as in_:
    #             compressed = in_.read()
    #         decompressed = unlzw(compressed)
    #         in_ = io.TextIOWrapper(io.BytesIO(decompressed))
    #         for line in in_:
    #             if line.startswith('<'):
    #                 continue
    #             A = line.strip().upper().split(' ')
    #             for n, a in enumerate(A):
    #                 if a not in isword and len(a) > 1 and a.endswith('.'):
    #                     out.write(a[:-1])
    #                     if n < len(A) - 1:
    #                         out.write("\n")
    #                 else:
    #                     out.write(a + " ")
    #             out.write("\n")
    #         del in_, compressed, decompressed
    # del isword, train_data_files

    # convert cleaned data into sentences
    # with gzip.open(cleaned_txt_gz, 'rt') as file_:
    #     text = file_.read()
    # sents = ngram_lm.text_to_sents(
    #     text, sent_end_expr='\n', word_delim_expr=' ')
    # del text

    # count n-grams in sentences
    # ngram_counts = ngram_lm.sents_to_ngram_counts(
    #     sents, max_order, sos='<s>', eos='</s>')
    # ensure all vocab terms have unigram counts (even if 0) for zeroton
    # interpolation
    # for v in vocab_5 | vocab_20:
    #     ngram_counts[0].setdefault(v, 0)
    # del sents

    # with gzip.open(os.path.join(lmdir, 'counts.txt.gz'), 'wt') as file_:
    #     for ngram_count in ngram_counts:
    #         for ngram, count in ngram_count.items():
    #             if isinstance(ngram, str):
    #                 file_.write('{} {}\n'.format(ngram, count))
    #             else:
    #                 file_.write('{} {}\n'.format(' '.join(ngram), count))

    # reread in ngram counts and vocab files (only necessary for line-by-line)
    ngram_counts = [dict() for _ in range(max_order)]
    with gzip.open(os.path.join(lmdir, 'counts.txt.gz'), 'rt') as file_:
        for line in file_:
            toks = line[:-1].split(' ')
            count = int(toks.pop())
            if len(toks) == 1:
                ngram_counts[0][toks[0]] = count
            else:
                ngram_counts[len(toks) - 1][tuple(toks)] = count
    assert sum(len(c) for c in ngram_counts) == 20306341
    with open(vocab2id_5_txt) as file_:
        vocab_5 = set(l.strip().split(' ')[0] for l in file_)
    with open(vocab2id_20_txt) as file_:
        vocab_20 = set(l.strip().split(' ')[0] for l in file_)

    # determine the higher-order hapax and prune them (we keep the lower-order
    # hapax so we don't reduce the vocabulary size)
    hapax = set()
    for ngram_count in ngram_counts[1:]:
        hapax |= set(k for (k, v) in ngram_count.items() if v == 1)

    # train and save the language models
    for vocab, toprune_txt_gz, lm_arpa_gz in (
            (vocab_5, toprune_5_txt_gz, lm_5_arpa_gz),
            (vocab_20, toprune_20_txt_gz, lm_20_arpa_gz)):
        to_prune = set(ngram_counts[0]) - vocab
        # prune all out-of-vocabulary n-grams
        for i, ngram_count in enumerate(ngram_counts[1:]):
            if i:
                to_prune |= set(
                    k for k in ngram_count
                    if k[:-1] in to_prune or k[-1] in to_prune
                )
            else:
                to_prune |= set(
                    k for k in ngram_count
                    if k[0] in to_prune or k[1] in to_prune)
        to_prune |= hapax
        assert not (to_prune & vocab)
        # with gzip.open(toprune_txt_gz, 'wt') as file_:
        #     for v in sorted(
        #             x if isinstance(x, str) else ' '.join(x)
        #             for x in to_prune):
        #         file_.write(v)
        #         file_.write('\n')
        prob_list = ngram_lm.ngram_counts_to_prob_list_kneser_ney(
            ngram_counts, sos='<s>', to_prune=to_prune)
        # remove start-of-sequence probability mass
        lm = ngram_lm.BackoffNGramLM(
            prob_list, sos='<s>', eos='</s>', unk='<UNK>')
        lm.prune_by_name({'<s>'})
        prob_list = lm.to_prob_list()
        with gzip.open(lm_arpa_gz, 'wt') as file_:
            ngram_lm.write_arpa(prob_list, file_)
        del lm, prob_list, to_prune


def wsj_char_prep(wsj_subdirs, data_root, max_order=5):
    dir_13_32_1 = find_link_dir(wsj_subdirs, '13-32.1')
    lmdir = os.path.join(data_root, 'local', 'char_lm')
    cleaned_txt_gz = os.path.join(lmdir, 'cleaned.txt.gz')
    train_data_root = os.path.join(
        dir_13_32_1, 'wsj1', 'doc', 'lng_modl', 'lm_train', 'np_data')
    lm_arpa_gz = os.path.join(lmdir, 'lm.arpa.gz')

    mkdir(lmdir)

    # clean up training data. We keep pretty messy though. All we do is convert
    # to upper case where applicable and replace spaces with underscores
    # assert os.path.isdir(train_data_root)
    # train_data_files = []
    # for subdir in ('87', '88', '89'):
    #     train_data_files.extend(
    #         glob(os.path.join(train_data_root, subdir), r'*.z'))
    # with gzip.open(cleaned_txt_gz, 'wt') as out:
    #     for train_data_file in train_data_files:
    #         with open(train_data_file, 'rb') as in_:
    #             compressed = in_.read()
    #         decompressed = unlzw(compressed)
    #         in_ = io.TextIOWrapper(io.BytesIO(decompressed))
    #         for line in in_:
    #             if line.startswith('<'):
    #                 continue
    #             A = line.strip().upper().replace(' ', '_')
    #             out.write(A)
    #             out.write("\n")
    #         del in_, compressed, decompressed
    # del train_data_files

    # split data into 'sentences.' This splits on every character
    with gzip.open(cleaned_txt_gz, 'rt') as file_:
        text = file_.read()
    sents = ngram_lm.text_to_sents(
        text, sent_end_expr='\n', word_delim_expr='')
    del text

    # count n-grams in sentences
    # ngram_counts = ngram_lm.sents_to_ngram_counts(
    #     sents, max_order, sos='<s>', eos='</s>')
    # del sents
    # ngram_counts[0]['<UNK>'] = ngram_counts[0]['<NOISE>'] = 0
    # with gzip.open(os.path.join(lmdir, 'counts.txt.gz'), 'wt') as file_:
    #     for ngram_count in ngram_counts:
    #         for ngram, count in ngram_count.items():
    #             if isinstance(ngram, str):
    #                 file_.write('{} {}\n'.format(ngram, count))
    #             else:
    #                 file_.write('{} {}\n'.format(' '.join(ngram), count))

    # reread in ngram_counts (only necessary for line-by-line)
    ngram_counts = [dict() for _ in range(max_order)]
    with gzip.open(os.path.join(lmdir, 'counts.txt.gz'), 'rt') as file_:
        for line in file_:
            toks = line[:-1].split(' ')
            count = int(toks.pop())
            if len(toks) == 1:
                ngram_counts[0][toks[0]] = count
            else:
                ngram_counts[len(toks) - 1][tuple(toks)] = count
    assert sum(len(c) for c in ngram_counts) == 579971

    # determine the hapax and put them in the list to prune
    to_prune = set()
    for ngram_count in ngram_counts:
        # this will exclude our special count-0 terms
        to_prune |= set(k for (k, v) in ngram_count.items() if v == 1)
    with gzip.open(os.path.join(lmdir, 'pruned.txt.gz'), 'wt') as file_:
        for v in sorted(
                x if isinstance(x, str) else ' '.join(x) for x in to_prune):
            file_.write('{}\n'.format(v))

    # train the language model
    # unigrams and bigrams cannot use default deltas because the count of
    # counts for two unique contexts is lower than that for 3.
    prob_list = ngram_lm.ngram_counts_to_prob_list_kneser_ney(
        ngram_counts,
        delta=[(0.5, 1., 1.5)] * max_order,
        sos='<s>', to_prune=to_prune)

    # determine the vocabulary. We might've pruned some characters so we only
    # do it here.
    vocab = sorted(prob_list[0])
    with open(os.path.join(lmdir, 'vocab2id.txt'), 'w') as file_:
        for i, v in enumerate(vocab):
            file_.write('{} {}\n'.format(v, i))

    # remove any unigram probability mass on the start-of-sequence token
    # (this should be the zeroton mass)
    lm = ngram_lm.BackoffNGramLM(
        prob_list, sos='<s>', eos='</s>', unk='<UNK>')
    del prob_list
    lm.prune_by_name({'<s>'})
    print('{} PP: {}'.format(lm_arpa_gz, lm.corpus_perplexity(sents)))
    prob_list = lm.to_prob_list()
    del lm

    # write to arpa file
    with gzip.open(lm_arpa_gz, 'wt') as file_:
        ngram_lm.write_arpa(prob_list, file_)


def build_preamble_parser(subparsers):
    parser = subparsers.add_parser(
        'preamble',
        help='Do all pre-initialization setup. Needs to be done only once.'
    )
    parser.add_argument(
        'wsj_roots', nargs='+', type=os.path.abspath,
        help='Location of WSJ data directories, corresponding to WSJ0 '
        '(LDC93S6A or LDC93S6B) and WSJ1 (LDC94S13A or LDC94S13B)'
    )


def build_init_word_parser(subparsers):
    parser = subparsers.add_parser(
        'init_word',
        help='Perform setup common to all word-based parsing. '
        'Needs to be done only once. Preceded by "preamble" command.'
    )
    parser.add_argument(
        'wsj_roots', nargs='+', type=os.path.abspath,
        help='Location of WSJ data directories, corresponding to WSJ0 '
        '(LDC93S6A or LDC93S6B) and WSJ1 (LDC94S13A or LDC94S13B)'
    )


def build_init_char_parser(subparsers):
    parser = subparsers.add_parser(
        'init_char',
        help='Perform setup common to all character-based parsing. '
        'Needs to be done only once. Preceded by "preamble" command.'
    )
    parser.add_argument(
        'wsj_roots', nargs='+', type=os.path.abspath,
        help='Location of WSJ data directories, corresponding to WSJ0 '
        '(LDC93S6A or LDC93S6B) and WSJ1 (LDC94S13A or LDC94S13B)'
    )


def build_parser():
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument(
        'data_root', type=os.path.abspath,
        help='The root directory under which to store data. Typically '
        '``data/``'
    )
    subparsers = parser.add_subparsers(
        title='commands', required=True, dest='command')
    build_preamble_parser(subparsers)
    build_init_word_parser(subparsers)
    build_init_char_parser(subparsers)
    return parser


def preamble(options):

    wsj_subdirs = []
    for wsj_root in options.wsj_roots:
        wsj_subdirs.extend(glob(wsj_root, r'??-?.?'))
        wsj_subdirs.extend(glob(wsj_root, r'??-??.?'))

    wsj_data_prep(wsj_subdirs, options.data_root)


def init_word(options):

    wsj_subdirs = []
    for wsj_root in options.wsj_roots:
        wsj_subdirs.extend(glob(wsj_root, r'??-?.?'))
        wsj_subdirs.extend(glob(wsj_root, r'??-??.?'))

    wsj_word_prep(wsj_subdirs, options.data_root)


def init_char(options):

    wsj_subdirs = []
    for wsj_root in options.wsj_roots:
        wsj_subdirs.extend(glob(wsj_root, r'??-?.?'))
        wsj_subdirs.extend(glob(wsj_root, r'??-??.?'))

    wsj_char_prep(wsj_subdirs, options.data_root)


def main(args=None):
    '''Prepare WSJ data for end-to-end pytorch training'''

    parser = build_parser()
    options = parser.parse_args(args)

    if options.command == 'preamble':
        preamble(options)
    elif options.command == 'init_word':
        init_word(options)
    elif options.command == 'init_char':
        init_char(options)


if __name__ == '__main__':
    sys.exit(main())
