import pytest
from indexing_exam.task04 import (
    delta_decode, delta_encode, vb_encode, vb_decode, compress_postings, decompress_postings,
    intersect_two_pointer, intersect_with_skips, CompressedPosting, intersect_compressed
)


def test_delta_example():
    ids = [3, 10, 11, 25]
    assert delta_encode(ids) == [3, 7, 1, 14]
    assert delta_decode([3, 7, 1, 14]) == ids


def test_vbyte_boundaries():
    nums = [0, 127, 128, 16384, 999999]
    assert vb_decode(vb_encode(nums)) == nums
    with pytest.raises(ValueError):
        vb_decode(bytes([1]))


def test_posting_roundtrip_and_intersections():
    a = [3, 10, 11, 25, 100]
    b = [1, 10, 25, 90, 100]
    assert decompress_postings(compress_postings(a)) == a
    expected = [10, 25, 100]
    assert intersect_two_pointer(a, b) == expected
    assert intersect_with_skips(a, b) == expected
    assert intersect_compressed(CompressedPosting(a), CompressedPosting(b)) == expected
