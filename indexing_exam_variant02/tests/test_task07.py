from indexing_exam.task07 import word_shingles, jaccard, minhash_signature, minhash_similarity


def test_assignment_example():
    a = word_shingles("a b c d", 2)
    b = word_shingles("a b c e", 2)
    assert jaccard(a, b) == 0.5


def test_minhash_reproducible():
    s = word_shingles("история архитектуры культурное наследие", 2)
    assert minhash_signature(s, 64, 42) == minhash_signature(s, 64, 42)
    assert minhash_similarity(minhash_signature(s, 64, 42), minhash_signature(s, 64, 42)) == 1.0
