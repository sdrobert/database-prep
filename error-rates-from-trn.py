#! /usr/bin/env python

# Copyright 2024 Sean Robertson
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
import sys
import argparse
import warnings

from typing import Dict, Sequence, Optional

import numpy as np
import jiwer

from pydrobert.torch.data import read_trn_iter
from pydrobert.torch.argcheck import as_nonnegi, as_open01

Samples = Sequence[jiwer.WordOutput]
MIN_PAD = 10


def _bootstrap_metric(samples: Samples, samples2: Optional[Samples] = None) -> float:
    errors = sum((s.insertions + s.deletions + s.substitutions) for s in samples)
    lens = sum((s.deletions + s.substitutions + s.hits) for s in samples)
    if samples2 is not None:
        assert all(
            (s1.deletions + s1.substitutions + s1.hits)
            == (s2.deletions + s2.substitutions + s2.hits)
            for (s1, s2) in zip(samples, samples2)
        )
        errors -= sum((s.insertions + s.deletions + s.substitutions) for s in samples2)
    return errors / lens


DOC = """Determine error rates between two or more trn files

An error rate measures the difference between reference (gold-standard) and hypothesis
(machine-generated) transcriptions by the number of single-token insertions, deletions,
and substitutions necessary to convert the hypothesis transcription into the reference
one.

A "trn" file is the standard transcription file without alignment information used in
the sclite (http://www1.icsi.berkeley.edu/Speech/docs/sctk-1.2/sclite.htm) toolkit. It
has the format

    here is a transcription (utterance_a) here is another (utterance_b)

WARNING! this command uses jiwer (https://github.com/jitsi/jiwer) as a backend, which
assumes a uniform cost for instertions, deletions, and substitutions. This is not suited
to certain corpora. Consult the corpus-specific page on the wiki
(https://github.com/sdrobert/pytorch-database-prep/wiki) for more details.
"""

EPILOGUE = """\
Bootstrapping
-------------
This script allows one to bootstrap confidence intervals on both absolute error rates
and differences between systems. If --bootstrap-samples is set to some positive value,
each error rate or difference (hereafter "the statistic") XX.X% is accompanied by a
braced pair [XX.X%,XX.X%] specifying the bootstrapped confidence interval: the range in
which we expect to see the statistic fall between (1 - alpha) * 100% of the time were we
to have sampled some other set of utterances, which we simulate by resampling from the
utterances we have.

Note that we're measuring the reliability of our estimate of the estimated statistic,
not how reliably different two distributions are. We expect the error rate to fall
between this lower bound and this upper bound. If --differences is set, the confidence
intervals give us lower and upper bounds on that difference. If a difference doesn't
include 0, it's reliably positive or negative.

The bootstrapped confidence intervals can be too tight when utterances depend on one
another, e.g. through speaker or topic. When possible, one should specify a --utt2grp
file, containing key-value pairs

    <utt-id> <grp>

Such that utterances *across* groups are roughly independent. Groups could be speaker
ids, recording ids, etc. If set, the bootstrap will resample by drawing one
representative from a group at a time.

For more information on bootstrapping in ASR, see

    Liu, Z. and Peng, F. (2020) "Statistical testing on ASR performance via blockwise
    bootstrap" https://www.isca-archive.org/interspeech_2020/liu20c_interspeech.html

or read the documentation from the confidence_intervals package

    Ferrer, L. and Riera, P. "Confidence intervals for evaluation in machine learning"
    https://github.com/luferrer/ConfidenceIntervals
"""


def main(args=None):

    parser = argparse.ArgumentParser(
        description=DOC,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOGUE,
    )
    parser.add_argument(
        "ref_file", type=argparse.FileType("r"), help="The reference trn file"
    )
    parser.add_argument(
        "hyp_files",
        nargs="+",
        type=argparse.FileType("r"),
        help="One or more hypothesis trn files",
    )
    parser.add_argument(
        "--suppress-warning",
        action="store_true",
        default=False,
        help="Suppress the warning about the backend",
    )
    parser.add_argument(
        "--ignore-empty-refs",
        action="store_true",
        default=False,
        help="If set, will throw out empty references from the analysis",
    )
    parser.add_argument(
        "--differences",
        action="store_true",
        default=False,
        help="If set, display differences in error rates",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=as_nonnegi,
        default=0,
        help="If nonzero, compute bootstrap confidence intervals with this many "
        "resamples. See epilogue for more details",
    )
    parser.add_argument(
        "--bootstrap-alpha",
        type=as_open01,
        default=0.05,
        help="Power level of bootstrap confidence interval. See epilogue for more "
        "details",
    )
    parser.add_argument(
        "--bootstrap-utt2grp",
        type=argparse.FileType("r"),
        default=None,
        help="Mapping file from utterance to group for bootstrap. See epilogue for "
        "more details",
    )
    options = parser.parse_args(args)

    if not options.suppress_warning:
        warnings.warn(
            "Not all corpora compute error rates the same way. Look at this command's "
            "documentation. To suppress this warning, use the flag '--suppress-warning'"
        )

    ref_dict = dict(read_trn_iter(options.ref_file, not options.suppress_warning))
    rname = options.ref_file.name
    print(f"ref '{rname}'")
    empty_refs = set(key for key in ref_dict if not ref_dict[key])
    if empty_refs:
        print(
            "One or more reference transcriptions are empty: "
            f"{', '.join(empty_refs)}",
            file=sys.stderr,
            end="",
        )
        if options.ignore_empty_refs:
            print(".", file=sys.stderr)
            for empty_ref in empty_refs:
                del ref_dict[empty_ref]
        else:
            print("! Consider adding --ignore-empty-refs", file=sys.stderr)
            return 1
    keys = sorted(ref_dict)
    refs = [" ".join(ref_dict[x]) for x in keys]
    del ref_dict

    if options.bootstrap_samples > 0:
        from confidence_intervals import evaluate_with_conf_int

        if options.bootstrap_utt2grp:
            utt2grp = dict(
                line.strip().split()
                for line in options.bootstrap_utt2grp
                if line.strip()
            )
            conditions = []
            grp2id = dict()
            for key in keys:
                if key not in utt2grp:
                    print(
                        f"'{options.bootstrap_utt2grp.name}' missing utterance {key}!",
                        file=sys.stderr,
                    )
                    return 1
                conditions.append(grp2id.setdefault(key, len(grp2id)))
            del utt2grp, grp2id
            conditions = np.asarray(conditions)
        else:
            conditions = None
    else:
        evaluate_with_conf_int = conditions = None

    hname2samples: Dict[str, Samples] = dict()
    er_hnames = []
    for hyp_file in options.hyp_files:
        hname = hyp_file.name
        hyp_dict = dict(
            (k, v)
            for (k, v) in read_trn_iter(hyp_file, not options.suppress_warning)
            if k not in empty_refs
        )
        if sorted(hyp_dict) != keys:
            keys_, keys = set(hyp_dict) - empty_refs, set(keys)
            print(
                f"ref and hyp file '{hname}' have different utterances!",
                file=sys.stderr,
            )
            diff = sorted(keys - keys_)
            if diff:
                print(f"Missing from hyp: " + " ".join(diff), file=sys.stderr)
            diff = sorted(keys - keys_)
            if diff:
                print(f"Missing from ref: " + " ".join(diff), file=sys.stderr)
            return 1
        hyps = [" ".join(hyp_dict[x]) for x in keys]
        if evaluate_with_conf_int:
            samples = np.asarray([jiwer.process_words(*rh) for rh in zip(refs, hyps)])
            er, ci = evaluate_with_conf_int(
                samples,
                _bootstrap_metric,
                num_bootstraps=options.bootstrap_samples,
                alpha=100 * options.bootstrap_alpha,
                conditions=conditions,
            )
            print(f"hyp '{hname}': {er:.1%} [{ci[0]:.1%},{ci[1]:.1%}]")
            hname2samples[hname] = samples
        else:
            er = jiwer.wer(refs, hyps)
            print(f"hyp '{hname}': {er:.1%}")
        er_hnames.append((er, hname))
    er_hnames.sort()
    print(f"best hyp '{er_hnames[0][1]}': {er_hnames[0][0]:.1%}")

    if options.differences and len(er_hnames) > 1:
        print("\nDifferences:")
        offs = ""
        width = max(max(len(x[1]) for x in er_hnames) + 1, MIN_PAD)
        cell_fmt = f"{{:{width}}}"

        print(cell_fmt.format(""), end="| to\n")
        print(cell_fmt.format("from"), end="| ")
        print(" ".join(cell_fmt.format(x[1]) for x in er_hnames[1:]))
        print("=" * (width + 1) * len(er_hnames))
        while len(er_hnames) > 1:
            start_er, start_hname = er_hnames.pop(0)
            print(cell_fmt.format(start_hname), end="| ")
            print(offs, end="")
            for end_er, end_hname in er_hnames:
                if evaluate_with_conf_int:
                    diff, ci = evaluate_with_conf_int(
                        hname2samples[end_hname],
                        _bootstrap_metric,
                        num_bootstraps=options.bootstrap_samples,
                        samples2=hname2samples[start_hname],
                        alpha=100 * options.bootstrap_alpha,
                        conditions=conditions,
                    )
                    s = f"{diff:.1%} [{ci[0]:.1%},{ci[1]:.1%}]"
                else:
                    s = f"{end_er - start_er:.1%}"
                print(cell_fmt.format(s), end=" ")
            print("")
            offs += " " * (width + 1)


if __name__ == "__main__":
    sys.exit(main())
