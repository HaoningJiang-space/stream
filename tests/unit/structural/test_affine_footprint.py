import pytest
from xdsl.ir.affine import AffineBinaryOpExpr, AffineBinaryOpKind, AffineExpr, AffineMap

from stream.workload.affine_footprint import footprint


def test_conv_stride_and_dilation_footprint():
    index = AffineExpr.dimension(0) * 2 + AffineExpr.dimension(1) * 3
    affine_map = AffineMap(2, 0, (index,))

    assert footprint(affine_map, {0: range(0, 4), 1: range(0, 3)}) == (range(0, 13),)


def test_negative_coefficient_footprint():
    index = AffineExpr.dimension(0) * -1 + AffineExpr.constant(5)

    assert footprint(AffineMap(1, 0, (index,)), {0: range(0, 4)}) == (range(2, 6),)


def test_missing_dimension_and_non_box_operator_raise():
    with pytest.raises(ValueError, match="does not bound dimension"):
        footprint(AffineMap(2, 0, (AffineExpr.dimension(1),)), {0: range(0, 2)})

    mod = AffineBinaryOpExpr(AffineBinaryOpKind.Mod, AffineExpr.dimension(0), AffineExpr.constant(2))
    with pytest.raises(NotImplementedError):
        footprint(AffineMap(1, 0, (mod,)), {0: range(0, 4)})
