import numpy as np
import pytest

from inferlab.config import TINY_RANDOM
from inferlab.foundations.tokenizer import ByteTokenizer, Llama2Tokenizer
from inferlab.foundations.weights import random_model


@pytest.fixture(scope="session")
def tiny_model():
    return random_model(TINY_RANDOM, seed=1)


@pytest.fixture(scope="session")
def byte_tok():
    return ByteTokenizer()


def make_bpe_tokenizer() -> Llama2Tokenizer:
    """A small Llama-2-style vocab: specials, 256 byte tokens, chars and a few merges."""
    vocab = [b"<unk>", b"\n<s>\n", b"\n</s>\n"] + [f"<0x{b:02X}>".encode() for b in range(256)]
    scores = [0.0] * len(vocab)
    pieces = [" ", "a", "b", "c", "d", "e", "h", "l", "o", "r", "w", "t", " h", "he", "ll", "llo", " w", "or",
              "orl", "orld", " hello", " world", "ab", "abc", "</", "s>"]
    for i, p in enumerate(pieces):
        vocab.append(p.encode())
        scores.append(float(len(p) * 10 - i * 0.01))
    return Llama2Tokenizer(vocab, scores)


@pytest.fixture(scope="session")
def bpe_tok():
    return make_bpe_tokenizer()


@pytest.fixture
def rng():
    return np.random.default_rng(0)
