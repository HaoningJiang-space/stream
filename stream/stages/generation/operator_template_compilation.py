"""Callable pre-tiling seam for an exact paired operator-template assignment."""

from stream.stages.context import StageContext
from stream.stages.stage import Stage, StageCallable
from stream.structural.operator_template_contract import compile_operator_templates


class OperatorTemplateCompilationStage(Stage):
    """Compile an optional structural assignment before kernel and tiling stages.

    With no assignment this stage is an identity operation. With an assignment,
    a versioned library is mandatory and the mapping becomes singleton-valued for
    each selected operator before any consumer reads its iteration space. Gate 1A-v3
    composes this stage with the production downstream stages; public API wiring is
    intentionally outside that gate's claim boundary.
    """

    REQUIRED_FIELDS = ("workload", "mapping")

    def __init__(self, list_of_callables: list[StageCallable], ctx: StageContext):
        super().__init__(list_of_callables, ctx)
        self.workload = self.ctx.require_value("workload", self.__class__.__name__)
        self.mapping = self.ctx.require_value("mapping", self.__class__.__name__)

    def run(self):
        assignment = self.ctx.get("operator_template_assignment")
        library = self.ctx.get("operator_template_library")
        if (assignment is None) != (library is None):
            raise ValueError("operator_template_assignment and operator_template_library must be provided together")
        if assignment is not None:
            compilation = compile_operator_templates(self.workload, self.mapping, assignment, library)
            self.ctx.set(mapping=compilation.mapping, operator_template_compilation=compilation)
        sub_stage = self.list_of_callables[0](self.list_of_callables[1:], self.ctx)
        yield from sub_stage.run()
