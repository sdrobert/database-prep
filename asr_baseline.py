#! /usr/bin/env python
#
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

import abc

from typing import Callable, Collection, Dict, Optional, Sequence, Tuple
from typing_extensions import Literal
from itertools import chain

import param
import torch

import pydrobert.torch.config as config
from pydrobert.torch.modules import (
    BeamSearch,
    ConcatSoftAttention,
    CTCPrefixSearch,
    ExtractableSequentialLanguageModel,
    ExtractableShallowFusionLanguageModel,
    MixableSequentialLanguageModel,
)

__all__ = [
    "CTCSpeechRecognizer",
    "Encoder",
    "EncoderDecoderSpeechRecognizer",
    "FeedForwardEncoder",
    "RecurrentDecoderWithAttention",
    "RecurrentEncoder",
    "SpeechRecognizer",
]


def check_in(name: str, val: str, choices: Collection[str]):
    if val not in choices:
        choices = "', '".join(sorted(choices))
        raise ValueError(f"Expected {name} to be one of '{choices}'; got '{val}'")


def check_positive(name: str, val, nonnegative=False):
    pos = "non-negative" if nonnegative else "positive"
    if val < 0 or (val == 0 and not nonnegative):
        raise ValueError(f"Expected {name} to be {pos}; got {val}")


def check_equals(name: str, val, other):
    if val != other:
        raise ValueError(f"Expected {name} to be equal to {other}; got {val}")


class Encoder(torch.nn.Module, metaclass=abc.ABCMeta):
    """Encoder module

    The encoder is the part of the baseline which, given some input, always performs
    the same computations in the same order. It contains no auto-regressive connections.

    This class serves both as a base class for encoders and

    Call Parameters
    ---------------
    input : torch.Tensor
        A tensor of shape ``(max_seq_len, batch_size, input_size)`` of sequences
        which have been right-padded to ``max_seq_len``.
    lens : torch.Tensor
        A tensor of shape ``(batch_size,)`` containing the actual lengths of sequences
        such that, for each batch element ``n``, only ``input[:lens[n], n]`` are valid
        coefficients.

    Returns
    -------
    output, lens _: torch.Tensor
        `output` is a  tensor of shape ``(max_seq_len', batch_size, hidden_size)`` with a
        similar interpretation to `input`. `lens_` similar to `lens`
    """

    __constants__ = "input_size", "hidden_size"

    input_size: int
    hidden_size: int

    def __init__(self, input_size: int, hidden_size: int) -> None:
        check_positive("input_size", input_size)
        check_positive("hidden_size", hidden_size)
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

    def reset_parameters(self):
        pass

    @abc.abstractmethod
    def forward(
        self, input: torch.Tensor, lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    __call__: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


class FeedForwardEncoder(Encoder):
    """Encoder is a simple feed-forward network"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        nonlinearity: Literal["relu", "tanh", "sigmoid"] = "relu",
        dropout: float = 0.0,
    ):
        check_positive("num_layers", num_layers)
        check_positive("dropout", dropout, nonnegative=True)
        check_in("nonlinearity", nonlinearity, {"relu", "tanh", "sigmoid"})
        super().__init__(input_size, hidden_size)
        drop = torch.nn.Dropout(dropout)
        if nonlinearity == "relu":
            nonlin = torch.nn.ReLU()
        elif nonlinearity == "tanh":
            nonlin = torch.nn.Tanh()
        else:
            nonlin = torch.nn.Sigmoid()
        self.stack = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, bias),
            nonlin,
            drop,
            *chain(
                *(
                    (torch.nn.Linear(hidden_size, hidden_size, bias), nonlin, drop)
                    for _ in range(num_layers - 1)
                )
            ),
        )

    def reset_parameters(self):
        super().reset_parameters()
        for n in range(0, len(self.stack), 2):
            self.stack[n].reset_parameters()

    def forward(
        self, input: torch.Tensor, lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.stack(input), lens


class RecurrentEncoder(Encoder):
    """Recurrent encoder

    Warnings
    --------
    If `bidirectional` is :obj:`True`, `hidden_size` must be divisible by 2. Each
    direction's hidden_state
    """

    rnn: torch.nn.RNNBase

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        bidirectional: bool = True,
        recurrent_type: Literal["lstm", "rnn", "gru"] = "lstm",
        dropout: float = 0.0,
    ) -> None:
        check_positive("num_layers", num_layers)
        check_in("recurrent_type", recurrent_type, {"lstm", "rnn", "gru"})
        check_positive("dropout", dropout, nonnegative=True)
        check_positive("hidden_size", hidden_size)
        if bidirectional:
            if hidden_size % 2:
                raise ValueError(
                    "for bidirectional encoder, hidden_size must be divisible by 2; "
                    f"got {hidden_size}"
                )
            hidden_size //= 2
        super().__init__(input_size, hidden_size)
        if recurrent_type == "lstm":
            class_ = torch.nn.LSTM
        elif recurrent_type == "gru":
            class_ = torch.nn.GRU
        else:
            class_ = torch.nn.RNN
        self.rnn = class_(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.rnn.reset_parameters()

    def forward(
        self, input: torch.Tensor, lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T = input.size(0)
        input = torch.nn.utils.rnn.pack_padded_sequence(
            input, lens.cpu(), enforce_sorted=False
        )
        output, lens_ = torch.nn.utils.rnn.pad_packed_sequence(
            self.rnn(input)[0], total_length=T
        )
        return output, lens_.to(lens)


class RecurrentDecoderWithAttention(MixableSequentialLanguageModel):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        embed_size: int = 128,
        dropout: float = 0.0,
    ):
        check_positive("vocab_size", vocab_size)
        check_positive("hidden_size", hidden_size)
        check_positive("embed_size", embed_size)
        check_positive("dropout", dropout, nonnegative=True)
        super().__init__(vocab_size)
        self.embed = torch.nn.Embedding(
            vocab_size + 1, embed_size, padding_idx=vocab_size
        )
        self.attn = ConcatSoftAttention(
            hidden_size, hidden_size, hidden_size=hidden_size
        )
        self.cell = torch.nn.LSTMCell(hidden_size + embed_size, hidden_size)
        self.ff = torch.nn.Linear(hidden_size, vocab_size)

    def reset_parameters(self):
        self.embed.reset_parameters()
        self.attn.reset_parameters()
        self.cell.reset_parameters()
        self.ff.reset_parameters()

    @torch.jit.export
    def extract_by_src(
        self, prev: Dict[str, torch.Tensor], src: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return {
            "input": prev["input"].index_select(1, src),
            "mask": prev["mask"].index_select(1, src),
            "hidden": prev["hidden"].index_select(0, src),
            "cell": prev["cell"].index_select(0, src),
        }

    @torch.jit.export
    def mix_by_mask(
        self,
        prev_true: Dict[str, torch.Tensor],
        prev_false: Dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        mask = mask.unsqueeze(1)
        return {
            "input": prev_true["input"],
            "mask": prev_true["mask"],
            "hidden": torch.where(mask, prev_true["hidden"], prev_false["hidden"]),
            "cell": torch.where(mask, prev_true["cell"], prev_false["cell"]),
        }

    @torch.jit.export
    def update_input(
        self, prev: Dict[str, torch.Tensor], hist: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if "input" not in prev:
            raise RuntimeError("'input' must be in prev")
        input = prev["input"]
        if "mask" not in prev:
            if "lens" in prev:
                prev["mask"] = (
                    torch.arange(input.size(0), device=input.device).unsqueeze(1)
                    < prev["lens"]
                )
            else:
                prev["mask"] = input.new_ones(input.shape[:-1], dtype=torch.bool)
        if "hidden" in prev and "cell" in prev:
            return prev
        N = hist.size(1)
        zeros = self.ff.weight.new_zeros((N, self.attn.query_size))
        return {
            "input": input,
            "mask": prev["mask"],
            "hidden": zeros,
            "cell": zeros,
        }

    @torch.jit.export
    def calc_idx_log_probs(
        self, hist: torch.Tensor, prev: Dict[str, torch.Tensor], idx: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        input, mask, h_0 = prev["input"], prev["mask"], prev["hidden"]
        i = self.attn(h_0, input, input, mask)  # (N, I)
        idx_zero = idx == 0
        if idx_zero.all():
            x = torch.full(
                (hist.size(1),), self.vocab_size, dtype=hist.dtype, device=hist.device
            )
        else:
            x = hist.gather(
                0, (idx - 1).expand(hist.shape[1:]).clamp_min(0).unsqueeze(0)
            ).squeeze(
                0
            )  # (N,)
            x = x.masked_fill(idx_zero.expand(x.shape), self.vocab_size)
        x = self.embed(x)  # (N, E)
        x = torch.cat([i, x], 1)  # (N, I + E)
        h_1, c_1 = self.cell(x, (h_0, prev["cell"]))
        logits = self.ff(h_1)
        return (
            torch.nn.functional.log_softmax(logits, -1),
            {"input": input, "mask": mask, "hidden": h_1, "cell": c_1},
        )


class SpeechRecognizer(torch.nn.Module, metaclass=abc.ABCMeta):
    """ABC for ASR"""

    encoder: Encoder

    def __init__(self, encoder: Encoder) -> None:
        super().__init__()
        self.encoder = encoder

    @abc.abstractmethod
    def train_loss_from_encoded(
        self,
        input: torch.Tensor,
        lens: torch.Tensor,
        refs: torch.Tensor,
        ref_lens: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def train_loss(
        self,
        feats: torch.Tensor,
        lens: torch.Tensor,
        refs: torch.Tensor,
        ref_lens: torch.Tensor,
    ) -> torch.Tensor:
        input, lens = self.encoder(feats, lens)
        return self.train_loss_from_encoded(input, lens, refs, ref_lens)

    @abc.abstractmethod
    def decode_from_encoded(
        self, input: torch.Tensor, lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...

    def decode(
        self, feats: torch.Tensor, lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input, lens = self.encoder(feats, lens)
        return self.decode_from_encoded(input, lens)

    def forward(
        self, feats: torch.Tensor, lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.decode(feats, lens)

    __call__: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class CTCSpeechRecognizer(SpeechRecognizer):
    """ASR + CTC transduction

    Warning
    -------
    The blank token will be taken to be the index `vocab_size`, not :obj:`0`.
    """

    def __init__(
        self,
        vocab_size: int,
        encoder: Encoder,
        beam_width: int = 8,
        lm: Optional[MixableSequentialLanguageModel] = None,
        beta: float = 0.0,
    ) -> None:
        check_positive("vocab_size", vocab_size)
        check_positive("beam_width", beam_width)
        if lm is not None:
            check_equals("lm.vocab_size", lm.vocab_size, vocab_size)
        super().__init__(encoder)
        self.ff = torch.nn.Linear(encoder.hidden_size, vocab_size + 1)
        self.search = CTCPrefixSearch(beam_width, beta, lm)
        self.ctc_loss = torch.nn.CTCLoss(vocab_size)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.ff.reset_parameters()
        self.search.reset_parameters()

    def train_loss_from_encoded(
        self,
        input: torch.Tensor,
        lens: torch.Tensor,
        refs: torch.Tensor,
        ref_lens: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = torch.nn.functional.log_softmax(self.ff(input), 2)
        return self.ctc_loss(log_probs, refs.T, lens, ref_lens)

    def decode_from_encoded(
        self, input: torch.Tensor, lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.ff(input)
        beam, lens, _ = self.search(logits, lens)
        return beam[..., 0], lens[..., 0]  # most probable


class EncoderDecoderSpeechRecognizer(SpeechRecognizer):
    """Encoder/decoder speech recognizer

    The end-of-speech token is assumed to be ``decoder.vocab_size - 1``.
    """

    __constants__ = (
        "encoded_name_train",
        "encoded_lens_name_train",
        "encoded_name_decode",
        "encoded_lens_name_decode",
        "max_iters",
    )
    encoded_name_train: str
    encoded_lens_name_train: str
    encoded_name_decode: str
    encoded_lens_name_decode: str
    max_iters: int

    decoder: ExtractableSequentialLanguageModel

    def __init__(
        self,
        encoder: Encoder,
        decoder: ExtractableSequentialLanguageModel,
        beam_width: int = 8,
        lm: Optional[ExtractableSequentialLanguageModel] = None,
        beta: float = 0.0,
        encoded_name: str = "input",
        encoded_lens_name: str = "lens",
        pad_value: int = config.INDEX_PAD_VALUE,
        max_iters: int = 10_000,
    ) -> None:
        check_positive("beam_width", beam_width)
        if lm is not None:
            check_equals("lm.vocab_size", lm.vocab_size, decoder.vocab_size)
        check_positive("max_iters", max_iters)
        super().__init__(encoder)
        self.decoder = decoder
        self.encoded_name_train = self.encoded_name_decode = encoded_name
        self.encoded_lens_name_train = self.encoded_lens_name_decode = encoded_lens_name
        self.max_iters = max_iters
        if lm is None:
            lm = decoder
        else:
            lm = ExtractableShallowFusionLanguageModel(
                decoder, lm, beta, first_prefix="first."
            )
            self.encoded_name_decode = "first." + encoded_name
            self.encoded_lens_name_decode = "first." + encoded_lens_name

        self.search = BeamSearch(
            lm,
            beam_width,
            eos=decoder.vocab_size - 1,
            pad_value=pad_value,
        )
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=pad_value)

    def train_loss_from_encoded(
        self,
        input: torch.Tensor,
        lens: torch.Tensor,
        refs: torch.Tensor,
        ref_lens: torch.Tensor,
    ) -> torch.Tensor:
        mask = torch.arange(refs.size(0), device=refs.device).unsqueeze(1) >= ref_lens
        refs = refs.masked_fill(mask, self.search.pad_value)
        prev = {self.encoded_name_train: input, self.encoded_lens_name_train: lens}
        log_probs = self.decoder(refs[:-1].clamp_min(0), prev)
        assert log_probs.shape[:-1] == refs.shape
        return self.ce_loss(log_probs.flatten(0, 1), refs.flatten())

    def decode_from_encoded(
        self, input: torch.Tensor, lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prev = {self.encoded_name_decode: input, self.encoded_lens_name_decode: lens}
        hyps, lens, _ = self.search(prev, input.size(1), self.max_iters)
        return hyps[..., 0], lens[..., 0]  # most probable


class FeedForwardEncoderParams(param.Parameterized):
    num_layers: int = param.Integer(2, bounds=(1, None), doc="Number of layers")
    bias: bool = param.Boolean(True, doc="Whether to add a bias vector")


class RecurrentEncoderParameters(param.Parameterized):
    num_layers: int = param.Integer(2, bounds=(1, None), doc="Number of layers")
    recurrent_type: Literal["lstm", "gru", "rnn"] = param.ObjectSelector(
        "lstm", ["lstm", "gru", "rnn"], doc="Type of recurrent cell"
    )
    bidirectional: bool = param.Boolean(
        True,
        doc="Whether layers are bidirectional. If so, each direction has "
        "a hidden size of half the parameterized hidden_size. In this case, the "
        "parameter must be divisible by 2",
    )


class BaselineParameters(param.Parameterized):
    """All parameters for a baseline model"""

    input_size: int = param.Integer(
        41, bounds=(1, None), doc="Size of input feature vectors"
    )
    hidden_size: int = param.Integer(
        512, bounds=(1, None), doc="Size of hidden states, including encoder output"
    )
    vocab_size: int = param.Integer(
        32, bounds=(1, None), doc="Size of output vocabulary"
    )

    encoder_type: Literal["recur", "id"] = param.ObjectSelector(
        "recur", ["recur", "id"], doc="What encoder structure to use"
    )
    transducer_type: Literal["ctc", "encdec"] = param.ObjectSelector(
        "ctc", ["ctc", "encdec"], doc="How to perform audio -> text transduction"
    )


def main(args: Optional[Sequence[str]] = None):
    pass
