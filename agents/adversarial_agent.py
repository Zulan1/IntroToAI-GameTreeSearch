from __future__ import annotations
import copy
from typing import Tuple
from agents.search_agent import SearchAgent
from grid import Grid
from type_aliases import Node

class AdversarialAgent(SearchAgent):
    """Adversarial Agent Class

    Args:
        Agent (Agent): inherits from Agent class
    """
    visitedStates = dict[MultiAgentState, int] = {}

    def __init__(self, params: list[str], _: Grid) -> None:
        super().__init__(params, _)
        self.cutOffLimit = 0
        self.alpha = float('-inf')
        self.beta = float('inf')
        self.cost = 0

    # def FormulateGoal(self, grid: Grid, _: int) -> set[Node]:
    #     """Formulates the goal of the agent"""
    #     from heuristics import GetPickUpsAndDropDowns

    #     return GetPickUpsAndDropDowns(grid, self)

    def GetActions(self, grid: Grid) -> set[Node]:
        """
        Returns a set of possible actions for the agent based on the current grid state.

        Parameters:
        grid (Grid): The grid representing the current state of the environment.

        Returns:
        set[Node]: A set of possible actions for the agent.
        """
        from utils import GetNeighbors

        actions = GetNeighbors(grid, self.coordinates)
        if any(self.coordinates == p.pickupLoc and self.cost < p.pickupTime
                for p in sum(grid.packages.values(), [])):
            print(f"Agent is at {self.coordinates} and might need to wait")
            actions.add(self.coordinates)
        return actions

    def Search(self, grid: Grid, _: set[Node], i: int, otherAgent: AdversarialAgent) -> list[Node]:
        """
        Performs a search for the best action to take based on the given grid state and other agent's information.

        Args:
            grid (Grid): The current grid state.
            _ (set[Node]): placeholder.
            otherAgent (AdversarialAgent): The other adversarial agent.
            i (int): The index of the current agent.

        Returns:
            list[Node]: The list of nodes representing the best action to take.
        """
        actions = self.GetActions(grid)
        nextAgent = copy.deepcopy(self)
        self.cost = i
        otherAgent.cost = i
        return max(actions, key=lambda a: otherAgent.MinValue(grid, nextAgent, a))

    def MinValue(self, grid: Grid, otherAgent: AdversarialAgent, action: Node,
                 alpha: float, beta: float, cutOffLimit: int) -> int:
        """
        Calculates the minimum value of the current agent's action.

        Args:
            grid (Grid): The current grid state.
            otherAgent (AdversarialAgent): The other adversarial agent.
            i (int): The index of the current agent.
            action (Node): The action to take.

        Returns:
            int: The minimum value of the current agent's action.
        """
        otherAgent.ProcessStep(grid, action, otherAgent.cost)
        otherAgent.cost += 1
        actions = self.GetActions(grid)
        if cutOffLimit == 0 or not actions: return self.Eval(otherAgent, grid)
        v = float('inf')
        for a in actions:
            v = min(v, otherAgent.MaxValue(grid, self, a, alpha, beta, cutOffLimit - 1)[0])
            if v <= alpha: return v
            beta = min(beta, v)
        return v

    def Eval(self, otherAgent: AdversarialAgent, grid: Grid) -> int:
        """
        Evaluates the current state of the game.

        Args:
            otherAgent (AdversarialAgent): The other adversarial agent.

        Returns:
            int: The evaluation of the current state of the game.
        """
        selfEval = self.score + 0.5 * len(self.packages) + 0.25 * len(grid.packages)
        otherEval = otherAgent.score + 0.5 * len(otherAgent.packages) + 0.25 * len(grid.packages)
        return selfEval - otherEval, selfEval, otherEval

    def MaxValue(self, grid: Grid, otherAgent: AdversarialAgent,
                 action: Node, alpha: float, beta: float, cutOffLimit: int) -> int:
        """
        Calculates the maximum value of the current agent's action.

        Args:
            grid (Grid): The current grid state.
            otherAgent (AdversarialAgent): The other adversarial agent.
            i (int): The index of the current agent.
            action (Node): The action to take.

        Returns:
            int: The maximum value of the current agent's action.
        """
        otherAgent.ProcessStep(grid, action, otherAgent.cost)
        otherAgent.cost += 1
        actions = self.GetActions(grid)
        if cutOffLimit == 0 or not actions: return self.Eval(otherAgent, grid)
        v = float('-inf')
        for a in actions:
            v = max(v, otherAgent.MinValue(grid, self, a, alpha, beta, cutOffLimit)[0])
            if v >= beta: return v
            alpha = max(alpha, v)
        return v

class MultiAgentState:
    """
    Represents the state of a multi-agent system.

    Attributes:
        grid (Grid): The grid representing the environment.
        agent1 (AStarAgent): The first A* agent.
        agent2 (AStarAgent): The second A* agent.
        interfering (InterferingAgent): The interfering agent.
    """

    def __init__(self, grid: Grid, agent: AdversarialAgent):
        self.grid: Grid = grid
        self.agent: AdversarialAgent = agent

    def __hash__(self):
        return hash(self.ToBaseClasses())

    def __eq__(self, other: MultiAgentState):
        return self.ToBaseClasses() == other.ToBaseClasses()

    def ToBaseClasses(self) ->\
        Tuple[Tuple[Node, int], Tuple[Node, Tuple[Node, int]], Tuple[Node ,Tuple[Node, int]], Tuple[Edge]]:
        """
        Converts the current state of the multi-agent to a tuple of base classes.

        Returns:
            Tuple[Tuple[Node, int], Tuple[Node, Tuple[Node, int]], Tuple[Node ,Tuple[Node, int]], Tuple[Edge]]: 
            A tuple containing the coordinates, pickups, dropdowns, and edges of the multi-agent.
        """
        return (self.agent.agent1.coordinates, self.agent.agent1.GetPickups(),
                self.agent.agent1.GetDropdowns(), self.agent.agent2.coordinates,
                self.agent.agent2.GetPickups(), self.agent.agent2.GetDropdowns(),
                tuple(self.grid.fragEdges))
