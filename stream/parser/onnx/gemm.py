from xdsl.ir.affine import AffineMap

from stream.parser.onnx.operator_parser import OnnxOperatorParser
from stream.workload.workload import ComputationNode, Tensor


class GemmParser(OnnxOperatorParser):
    """Parses an ONNX Gemm operator into a ComputationNode"""

    EXPECTED_NB_OF_INPUTS = 2  # A and B are required
    SEMANTIC_INPUT_EXCLUSIONS = {2: "UNMODELED_ADDITIVE_OPERAND"}

    def generate_node(self, name_to_tensor_dict: dict[str, Tensor]) -> ComputationNode:
        mappings = (
            AffineMap.from_callable(lambda m, k, n: (m, k)),
            AffineMap.from_callable(lambda m, k, n: (k, n)),
            AffineMap.from_callable(lambda m, k, n: (m, n)),
        )

        all_inputs = tuple(name_to_tensor_dict[inp] for inp in self.node.input)
        assert len(all_inputs) >= self.EXPECTED_NB_OF_INPUTS, "Gemm must have at least activation and weight inputs."
        # The current affine Gemm IR models A and B. ONNXModelParser records any supplied C operand
        # as a declared semantic exclusion before removing its disconnected InEdge.
        inputs = all_inputs[: self.EXPECTED_NB_OF_INPUTS]

        return ComputationNode(
            type=self.node.op_type,
            name=self.node.name,
            inputs=inputs,
            outputs=self.get_output_tensors(),
            operand_mapping=mappings,
        )
