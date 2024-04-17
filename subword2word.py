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
        description="Convert a subword or character-level transcription to a "
        "word-level one. Supposed to be used on hypothesis transcriptions; "
        'alternates (e.g. "{ foo / bar / @ }") are not permitted'
    )
    parser.add_argument(
        "subword_trn",
        metavar="IN",
        type=argparse.FileType("r"),
        nargs="?",
        default=argparse.FileType("r")("-"),
        help="A subword or character trn file to read. Defaults to stdin",
    )
    parser.add_argument(
        "word_trn",
        metavar="OUT",
        type=argparse.FileType("w"),
        nargs="?",
        default=argparse.FileType("w")("-"),
        help="A word trn file to write. Defaults to stdout",
    )
    parser.add_argument(
        "--space-char",
        metavar="CHAR",
        default="_",
        help="The character used in the character-level transcript that "
        "substitutes spaces",
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

    for line in options.subword_trn:
        if options.both_raw:
            trans = line
        else:
            x = line.strip().rsplit(" ", maxsplit=1)
            if len(x) == 1:
                trans, utt = "", x[0]
            else:
                trans, utt = x
        trans = (
            trans.replace(" ", "")
            .replace(options.space_char, " ")
            .replace("  ", " ")
            .strip()
        )
        options.word_trn.write(trans)
        options.word_trn.write(" ")
        if not options.both_raw and not options.raw_out:
            options.word_trn.write(utt)
        options.word_trn.write("\n")


if __name__ == "__main__":
    sys.exit(main())
