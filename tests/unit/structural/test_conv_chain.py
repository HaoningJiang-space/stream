from xdsl.ir.affine import AffineDimExpr, AffineMap

from stream.workload.tensor_domain import AffineBox, TensorTileDomain


def test_conv_affine_demand_includes_clipped_halo():
    tensor = TensorTileDomain((8, 8), dtype_bytes=2)
    affine_map = AffineMap(2, 0, (AffineDimExpr(0) + 1, AffineDimExpr(1) + 1))

    demand = tensor.demand_from_affine(affine_map, {0: range(0, 4), 1: range(0, 4)}, role="halo")

    assert demand.domain == AffineBox((1, 1), (5, 5))
    assert demand.role == "halo"
