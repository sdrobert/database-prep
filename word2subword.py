#! /usr/bin/env python

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

import argparse
import sys


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Convert a word-level transcription to a subword- or"
        "charater-level one. Supposed to be used on hypothesis transcriptions; "
        'alternates (e.g. "{ foo / bar / @ }") are not permitted. By default is '
        "character-level: use the argument -s for subwords"
    )
    parser.add_argument(
        "word_trn",
        metavar="IN",
        type=argparse.FileType("r"),
        nargs="?",
        default=argparse.FileType("r")("-"),
        help="A word trn file to read. Defaults to stdin",
    )
    parser.add_argument(
        "subword_trn",
        metavar="OUT",
        type=argparse.FileType("w"),
        nargs="?",
        default=argparse.FileType("w")("-"),
        help="A subword or character trn file to write. Defaults to stdout",
    )
    parser.add_argument(
        "--space-char",
        metavar="CHAR",
        default="_",
        help="The character used in the character-level transcript that "
        "substitutes spaces",
    )
    unit_group = parser.add_mutually_exclusive_group()
    unit_group.add_argument(
        "--subword-model",
        "-s",
        metavar="PTH",
        default=None,
        help="Path to a sentencepiece model. If set, will use this sentencepiece "
        "model to produce subwords, replacing the space meta-symbol with --space-char",
    )
    unit_group.add_argument(
        "--dummy",
        action="store_true",
        default=False,
        help="If set, don't actually change the transcript: leave it word-level",
    )

    transcript_group = parser.add_mutually_exclusive_group()
    transcript_group.add_argument(
        "--both-raw",
        action="store_true",
        default=False,
        help="The input (and thus the output) are raw, newline-delimited "
        "transcriptions, without utterance ids",
    )
    transcript_group.add_argument(
        "--raw-out",
        action="store_true",
        default=False,
        help="The input is a trn file, but the output should be a raw, "
        "newline-delimited file without utterance ids",
    )

    options = parser.parse_args(args)
    if options.dummy:

        def splitter(trans: str) -> str:
            return trans

    elif options.subword_model is not None:
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor()
        sp.load(options.subword_model)

        def splitter(trans: str) -> str:
            pieces = sp.encode_as_pieces(trans)
            return " ".join(
                piece.replace("\u2581", options.space_char) for piece in pieces
            )

    else:

        def splitter(trans: str) -> str:
            return " ".join(trans.replace(" ", options.space_char))

    for line in options.word_trn:
        if options.both_raw:
            trans = line
        else:
            x = line.strip().rsplit(" ", maxsplit=1)
            if len(x) == 1:
                trans, utt = "", x[0]
            else:
                trans, utt = x
        trans = splitter(trans.strip()).strip()
        options.subword_trn.write(trans)
        options.subword_trn.write(" ")
        if not options.both_raw and not options.raw_out:
            options.subword_trn.write(utt)
        options.subword_trn.write("\n")


if __name__ == "__main__":
    sys.exit(main())
