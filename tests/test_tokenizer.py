# tests/test_tokenizer.py
from src.tokenizer import CharTokenizer, SimpleBPETokenizer


def test_char_tokenizer_roundtrip():
    corpus = "To be, or not to be, that is the question."
    tok = CharTokenizer(corpus)

    ids = tok.encode(corpus)
    decoded = tok.decode(ids)

    assert decoded == corpus
    assert len(ids) == len(corpus)
    assert tok.vocab_size == len(set(corpus))


def test_bpe_tokenizer_roundtrip_and_compression():
    corpus = (
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox jumps over the lazy dog again and again. "
        "attention is all you need, attention is all you need."
    )
    tok = SimpleBPETokenizer()
    tok.train(corpus, vocab_size=280)

    ids = tok.encode(corpus)
    decoded = tok.decode(ids)

    assert decoded == corpus
    # BPE must compress the sequence compared to raw UTF-8 byte length
    assert len(ids) < len(corpus.encode("utf-8"))


def test_bpe_handles_unseen_utf8_bytes():
    """Byte-level BPE should never hit out-of-vocabulary errors on arbitrary UTF-8."""
    tok = SimpleBPETokenizer()
    tok.train("hello world hello world", vocab_size=262)

    unseen = "Edge LLMs 🚀 — 100% café!"
    assert tok.decode(tok.encode(unseen)) == unseen