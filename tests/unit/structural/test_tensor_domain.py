import pytest

from stream.workload.tensor_domain import AffineBox, TensorTileDomain, validate_no_reduction_split


def test_balanced_partition_exactly_covers_dense_domain():
    domain = TensorTileDomain((5, 4), dtype_bytes=2)
    fragments = domain.partition(axis=0, parts=2)

    assert [fragment.domain for fragment in fragments] == [
        AffineBox((0, 0), (3, 4)),
        AffineBox((3, 0), (5, 4)),
    ]
    assert sum(fragment.domain.elements for fragment in fragments) == 20


def test_v0_rejects_reduction_axis_partition():
    with pytest.raises(ValueError, match="reduction-axis"):
        validate_no_reduction_split((0, 2), (2,))
