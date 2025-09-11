from __future__ import annotations
from typing import Dict, List, Callable, Awaitable
from dataclasses import dataclass, field
from .task import Task
from .messages import Message

@dataclass
class SequentialWorkflow:
    name: str
    tasks: List[Task] = field(default_factory=list)

    def add(self, task: Task):
        self.tasks.append(task)
        return self

    async def run(self) -> Dict[str, Message]:
        results: Dict[str, Message] = {}
        for t in self.tasks:
            last_msg: Message | None = None
            for s in t.steps:
                last_msg = await s.run()
            if last_msg:
                results[t.name] = last_msg
        return results

@dataclass
class DAGWorkflow:
    name: str
    nodes: Dict[str, Callable[[], Awaitable[Message]]] = field(default_factory=dict)
    edges: Dict[str, List[str]] = field(default_factory=dict)  # from -> [to]

    def node(self, key: str, fn: Callable[[], Awaitable[Message]]):
        self.nodes[key] = fn
        return self

    def link(self, src: str, dst: str):
        self.edges.setdefault(src, []).append(dst)
        return self

    async def run(self) -> Dict[str, Message]:
        # simple Kahn-like topo ordering without cycle handling for brevity
        indeg = {k: 0 for k in self.nodes}
        for v in self.edges.values():
            for d in v:
                indeg[d] += 1
        ready = [k for k, d in indeg.items() if d == 0]
        results: Dict[str, Message] = {}
        while ready:
            cur = ready.pop(0)
            results[cur] = await self.nodes[cur]()
            for d in self.edges.get(cur, []):
                indeg[d] -= 1
                if indeg[d] == 0:
                    ready.append(d)
        return results