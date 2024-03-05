from __future__ import annotations
from typing import Callable
from agents.multi_agent import MultiAgent, State
from grid import Grid
from type_aliases import Node

class AdversarialAgent(MultiAgent):
    """Adversarial Agent Class"""
    def __init__(self, params: list[str], _: Grid) -> None:
        super().__init__(params, _)
        self.maxKeyFunc: Callable = lambda x: (x[0], x[1], -len(x[2]) if x[2] else float('-inf'))
        self.canBePruned: bool = True

    def Eval(self, state: State) -> list[float, float, list[Node]]:
        """
        Evaluates the given state and returns the difference in evaluation values between the agent and the other agent,
        the evaluation value of the agent itself, the sequence of nodes for the agent,
        and the sequence of nodes for the other agent.

        Args:
            state (State): The state to be evaluated.

        Returns:
            Tuple[float, float, list[Node], list[Node]]: A tuple containing the difference in evaluation values,
            the evaluation value of the agent, the sequence of nodes for the agent,
            and the sequence of nodes for the other agent.
        """
        selfEval: float = state.agent.AgentEval(state.grid)
        otherEval: float = state.otherAgent.AgentEval(state.grid)
        seq: list[Node] = state.agent.seq
        otherSeq: list[Node] = state.otherAgent.seq
        diffVal, selfVal = selfEval - otherEval, selfEval

        return [round(diffVal, 1), round(selfVal, 1), seq, otherSeq]
