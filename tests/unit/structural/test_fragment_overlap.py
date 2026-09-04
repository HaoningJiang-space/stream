from stream.workload.tensor_domain import AffineBox, TensorFragment, fragment_overlap_bytes


def test_h_to_w_fragment_overlap_uses_logical_intersection():
    producer = TensorFragment(AffineBox((0, 0), (2, 4)), (0, 1))
    consumer = TensorFragment(AffineBox((0, 2), (4, 4)), (1, 0))

    assert fragment_overlap_bytes(producer, consumer, dtype_bytes=2) == 8


def test_disjoint_fragments_move_no_bytes():
    producer = TensorFragment(AffineBox((0, 0), (2, 2)), (0, 1))
    consumer = TensorFragment(AffineBox((2, 2), (4, 4)), (0, 1))

    assert fragment_overlap_bytes(producer, consumer, dtype_bytes=2) == 0
