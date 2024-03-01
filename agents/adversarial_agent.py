from __future__ import annotations
import copy
from typing import Tuple
from agents.search_agent import SearchAgent
from agents.agent import Agent
from grid import Grid
from type_aliases import Node, Edge

def GetPickUpsAndDropDowns(grid: Grid, agent: SearchAgent) -> set[Node]:
        """Gets all the nodes of packages' pickups or dropoffs

        Args:
            grid (Grid): The Simulator's grid
            agent (GreedyAgent): The agent

        Returns:
            set[Node]: All the grid's pickup and dropoff locations
        """
        relevantNodes = set(grid.packages)
        relevantNodes = relevantNodes.union({p.dropoffLoc for s in grid.packages.values() for p in s})
        relevantNodes = relevantNodes.union({p.dropoffLoc for s in agent.packages.values() for p in s})
        return relevantNodes
    
class AdversarialAgent(SearchAgent):
    """Adversarial Agent Class

    Args:
        Agent (Agent): inherits from Agent class
    """
    cutOffLimit = 15

    def __init__(self, params: list[str], _: Grid) -> None:
        super().__init__(params, _)
        self.cost = 0
    
    def FormulateGoal(self, grid: Grid, _: int) -> set[Node]:
        """Formulates the goal of the agent"""
        return GetPickUpsAndDropDowns(grid, self)

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

    
    def Search(self, grid: Grid, _: set[Node], agents: list[Agent], i: int) -> list[Node]:
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
        print("Start Search")
        otherAgent =([agent for agent in agents if agent != self and isinstance(agent, AdversarialAgent)] or [None])[0]
        assert otherAgent is not None, "No other adversarial agent found"
        actions = self.GetActions(grid)
        visitedStates: dict[State, int] = {}
        alpha = float('-inf')
        beta = float('inf')
        cutOff = AdversarialAgent.cutOffLimit
        self.cost = i
        otherAgent.cost = i
        def key(action):
            if action == (0,1):
                print("Second option")
            v = otherAgent.MinValue(grid, self, action, alpha, beta, cutOff, visitedStates)
            print(f"Value: {v}, Action: {action}")
            return v
        nextAction = max(actions, key=key)
        print(f"Next Action: {nextAction}")
        return [nextAction]

    def MinValue(self, grid: Grid, otherAgent: AdversarialAgent, action: Node,
                 alpha: float, beta: float, cutOffLimit: int, visitedStates: dict[State, int]) -> int:
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
        defaultValue = (float('-inf'), float('-inf'))
        # nextAgent = copy.deepcopy(self)
        nextGrid = copy.deepcopy(grid)
        nextOtherAgent = copy.deepcopy(otherAgent)
        # print(f"Visited States: {visitedStates}\n\n")
        nextOtherAgent.cost += 1
        nextOtherAgent.ProcessStep(nextGrid, (nextOtherAgent.coordinates, action), nextOtherAgent.cost)
        nextState = State(nextGrid, nextOtherAgent, self)
        # print(f"len(visitedStates): {len(visitedStates)}")
        if nextState in visitedStates.keys() and visitedStates[nextState] <= nextOtherAgent.cost and\
            action != otherAgent.coordinates:
            # print("mashu")
            return defaultValue
        visitedStates[nextState] = nextOtherAgent.cost
        actions = self.GetActions(nextGrid)
        if nextOtherAgent.score >= 1:
            print(f"MinValue: {nextOtherAgent.Eval(self, nextGrid)}")  
        if cutOffLimit == 0 or not actions or len(nextGrid.packages) == 0:
            eval1 = nextOtherAgent.Eval(self, nextGrid)
            print(f"Eval: {eval1}, CutOff: {cutOffLimit}, Alpha: {alpha}, Beta: {beta},"
                  f"coords: {self.coordinates}, otherCoords: {otherAgent.coordinates},"
                  f"action: {actions}, len(nextGrid.packages): {len(nextGrid.packages)}\n")
            if eval1[1] == float(1):
                print("!!!"*100 + "\n")
            return eval1
        # print(f"cutOffLimit Min: {cutOffLimit}")
        v = (float('inf'), float('-inf'))
        for nextAction in actions:
            v = min(v, nextOtherAgent.MaxValue(nextGrid, self, nextAction, alpha, beta, cutOffLimit - 1, visitedStates),
                    key = lambda x: (x[0], -x[1]))
            #assert v != float('inf'), f"v: {v}, actions: {actions}, otherAgent: {nextOtherAgent.coordinates}"
            if v[0] <= alpha:
                # print(f"Prune Min Value: {v}, CutOff: {cutOffLimit}, Alpha: {alpha}, Beta: {beta}, coords: {self.coordinates}, otherCoords: {otherAgent.coordinates}")
                return v
            beta = min(beta, v[0])
        
        # print(f"Min Value: {v}, CutOff: {cutOffLimit}, Alpha: {alpha}, Beta: {beta}, coords: {self.coordinates}, otherCoords: {otherAgent.coordinates}")
        return v if v[0] != float('inf') else defaultValue

    def Eval(self, otherAgent: AdversarialAgent, grid: Grid) -> int:
        """
        Evaluates the current state of the game.

        Args:
            otherAgent (AdversarialAgent): The other adversarial agent.

        Returns:
            int: The evaluation of the current state of the game.
        """
        selfEval = self.score + 0.5 * len(self.packages) + 0.2 * len(grid.packages)
        otherEval = otherAgent.score + 0.5 * len(otherAgent.packages) + 0.2 * len(grid.packages)
        return round(selfEval - otherEval,1), selfEval#, otherEval

    def MaxValue(self, grid: Grid, otherAgent: AdversarialAgent,
                 action: Node, alpha: float, beta: float, cutOffLimit: int, visitedStates: dict[State, int]) -> int:
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
        defaultValue = (float('inf'), float('-inf'))
        # nextAgent = copy.deepcopy(self)
        nextGrid = copy.deepcopy(grid)
        nextOtherAgent = copy.deepcopy(otherAgent)
        # print(f"Visited States: {visitedStates}\n\n")
        nextOtherAgent.cost += 1
        nextOtherAgent.ProcessStep(nextGrid, (nextOtherAgent.coordinates, action), nextOtherAgent.cost)
        nextState = State(nextGrid, self, nextOtherAgent)
        # print(f"len(visitedStates): {len(visitedStates)}")
        if nextState in visitedStates.keys() and visitedStates[nextState] <= nextOtherAgent.cost and\
            action != otherAgent.coordinates:
            # print("mashu")
            return defaultValue
        visitedStates[nextState] = nextOtherAgent.cost
        actions = self.GetActions(nextGrid)
        if cutOffLimit == 0 or not actions or len(nextGrid.packages) == 0:
            eval1 = self.Eval(nextOtherAgent, nextGrid)
            print(f"Eval: {eval1}, CutOff: {cutOffLimit}, Alpha: {alpha}, Beta: {beta},"
                  f"coords: {self.coordinates}, otherCoords: {otherAgent.coordinates},"
                  f"action: {actions}, len(nextGrid.packages): {len(nextGrid.packages)}\n")
            if eval1[1] == float(1):
                print("!!!"*100 + "\n")
            return eval1
        # print(f"cutOffLimit Max: {cutOffLimit}")
        v = (float('-inf'), float('-inf'))
        if self.score >= 1:
            print(f"MinValue: {self.Eval(nextOtherAgent, nextGrid)}")
        for nextAction in actions:
            v = max(v, nextOtherAgent.MinValue(nextGrid, self, nextAction, alpha, beta, cutOffLimit, visitedStates),
                    key = lambda x: (x[0], x[1]))
            #if v != float('-inf'): continue#, f"v: {v}, actions: {actions}, otherAgent: {nextOtherAgent.coordinates}"
            if v[0] >= beta:
                # print(f"Prune Max Value: {v}, CutOff: {cutOffLimit}, Alpha: {alpha}, Beta: {beta}, coords: {self.coordinates}, otherCoords: {otherAgent.coordinates}")
                return v
            alpha = max(alpha, v[0])
        # print(f"Max Value: {v}, CutOff: {cutOffLimit}, Alpha: {alpha}, Beta: {beta}, coords: {self.coordinates}, otherCoords: {otherAgent.coordinates}")
        return v if v[0] != float('-inf') else defaultValue

class State:
    """
    Represents the state of a multi-agent system.

    Attributes:
        grid (Grid): The grid representing the environment.
        agent1 (AStarAgent): The first A* agent.
        agent2 (AStarAgent): The second A* agent.
        interfering (InterferingAgent): The interfering agent.
    """

    def __init__(self, grid: Grid, agent: AdversarialAgent, otherAgent: AdversarialAgent):
        self.grid: Grid = grid
        self.agent: AdversarialAgent = agent
        self.otherAgent: AdversarialAgent = otherAgent

    def __hash__(self):
        return hash(self.ToBaseClasses())

    def __eq__(self, other: State):
        return self.ToBaseClasses() == other.ToBaseClasses()

    def ToBaseClasses(self) ->\
        Tuple[Tuple[Node, int], Tuple[Node, Tuple[Node, int]], Tuple[Node ,Tuple[Node, int]], Tuple[Edge]]:
        """
        Converts the current state of the multi-agent to a tuple of base classes.

        Returns:
            Tuple[Tuple[Node, int], Tuple[Node, Tuple[Node, int]], Tuple[Node ,Tuple[Node, int]], Tuple[Edge]]: 
            A tuple containing the coordinates, pickups, dropdowns, and edges of the multi-agent.
        """
        return (self.agent.coordinates, self.agent.GetPickups(),
                self.agent.GetDropdowns(), self.otherAgent.coordinates,
                self.otherAgent.GetPickups(), self.otherAgent.GetDropdowns(),
                tuple(self.grid.fragEdges))
