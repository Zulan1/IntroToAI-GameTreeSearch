from __future__ import annotations
from typing import Callable
from agents.multi_agent import MultiAgent, State
from grid import Grid
from type_aliases import Node, MinimaxValueType

class SemiCoopAgent(MultiAgent):
    """Semi Cooperative Agent Class"""
    def __init__(self, params: list[str], _: Grid) -> None:
        super().__init__(params, _)
        self.maxKeyFunc: Callable = lambda x:\
            (x[1], x[2], -len(x[3]) if x[3] else float('-inf'), -len(x[4]) if x[4] else float('-inf'))
        self.allowPruning: bool = False

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
        diffEval = selfEval - otherEval
        return [round(diffEval,1), round(selfEval, 1), round(otherEval, 1), seq, otherSeq]

    def ReverseV(self, v):
        return [v[0], v[2], v[1], v[4], v[3]]

    def DebugMessage(self, v: MinimaxValueType, action: Node, optionNum: int) -> MinimaxValueType:
        """
        Converts the given MinimaxValueType to a debug format.

        Args:
            v (MinimaxValueType): The MinimaxValueType to convert.

        Returns:
            MinimaxValueType: The converted MinimaxValueType in debug format.
        """
        return (f"Option ({optionNum}): Action: {action}, Self Estimation Value: {v[1]}, "
                f"Opponent Estimation Value: {v[2]}, iterations: {MultiAgent.iterations}, "
                f"Prune Count: {MultiAgent.pruneCount}, Visited Count: {MultiAgent.visitedCount}\n"
                f"Sequence: {v[3]}\nOpponent Predicted Sequence: {v[4]}\n")
