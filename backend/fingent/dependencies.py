"""
Dependency resolution & "create the prerequisite first" (§7).

Runs at lifecycle step 5, before any agent is built. For a target spec/template:
  * read depends_on (template defaults + anything the LLM inferred);
  * for each, check whether a satisfying agent already exists & is ENABLED for this tenant;
  * missing HARD -> blocking notification (caller surfaces it / can auto-provision);
  * missing SOFT -> non-blocking warning;
  * auto-provision is recursive + topological (depth-first, bottom-up), creating the whole chain;
  * cycles (A->B->A) are detected and rejected with a clear error.
"""
from __future__ import annotations

from .schemas import AgentSpec, Dependency, DependencyCheck, DependencyType
from .store import Store


class DependencyResolver:
    def __init__(self, store: Store) -> None:
        self.store = store

    def _satisfied(self, tenant_id: str, dep_agent: str) -> bool:
        """Is there an enabled agent for this tenant that satisfies `dep_agent`?

        A dependency names an agent *type* (template name). It is satisfied if any enabled spec
        for the tenant was minted from that template (or shares the name).
        """
        for spec in self.store.list_specs(tenant_id):
            if spec.template == dep_agent or spec.name == dep_agent:
                return True
        return False

    def check(self, spec: AgentSpec) -> DependencyCheck:
        tenant_id = spec.security.tenant_id
        missing_hard, missing_soft = [], []
        for dep in spec.depends_on:
            if self._satisfied(tenant_id, dep.agent):
                continue
            (missing_hard if dep.type == DependencyType.HARD else missing_soft).append(dep)

        # build a topo-sorted provisioning chain for the missing HARD prerequisites
        order: list[str] = []
        cycle = None
        try:
            for dep in missing_hard:
                self._collect_chain(tenant_id, dep.agent, order, set(), [])
        except _CycleError as ce:
            cycle = ce.path
        order.append(spec.template or spec.name)  # the requested agent comes last

        return DependencyCheck(
            ok=len(missing_hard) == 0 and cycle is None,
            missing_hard=missing_hard, missing_soft=missing_soft,
            creation_order=order, cycle=cycle,
        )

    def _template_deps(self, template_name: str) -> list[Dependency]:
        tpl = self.store.get_template(template_name)
        return tpl.default_depends_on if tpl else []

    def _collect_chain(self, tenant_id, agent, order, visited, stack):
        """Depth-first, bottom-up. Raises _CycleError on a back-edge."""
        if agent in stack:
            raise _CycleError(stack + [agent])
        if agent in visited or self._satisfied(tenant_id, agent):
            return
        stack.append(agent)
        for dep in self._template_deps(agent):
            if dep.type == DependencyType.HARD:
                self._collect_chain(tenant_id, dep.agent, order, visited, stack)
        stack.pop()
        visited.add(agent)
        if agent not in order:
            order.append(agent)


class _CycleError(Exception):
    def __init__(self, path):
        self.path = path
        super().__init__(" -> ".join(path))
