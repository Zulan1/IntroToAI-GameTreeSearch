from __future__ import annotations
import copy
import heapq
from typing import Tuple
from agents.search_agent import SearchAgent
from agents.agent import Agent
from grid import Grid
from type_aliases import Node, Edge

class AdversarialAgent(SearchAgent):
    """Adversarial Agent Class

    Args:
        Agent (Agent): inherits from Agent class
    """
    cutOffLimit = 11
    iterations = 0
    agentNum = 0

    def __init__(self, params: list[str], _: Grid) -> None:
        super().__init__(params, _)
        self.cost = 0
        self.otherAgent: AdversarialAgent = None
        self.cutoff = 0

    def FormulateGoal(self, grid: Grid, _: int) -> set[Node]:
        """Formulates the goal of the agent"""
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
        return self.SortActions(actions, grid)

    def SortActions(self, actions: set[Node], grid: Grid) -> list[Node]:
        """
        Sorts the given set of actions based on the current state of the grid.

        Args:
            actions (set[Node]): The set of actions to sort.
            grid (Grid): The current state of the grid.

        Returns:
            list[Node]: The sorted list of actions.
        """
        from utils import Dijkstra
        packages = grid.GetPickups() + self.GetDropdowns()
        def SortKey(action: Node) -> tuple[int, int]:
            nonlocal packages

            lens = []
            for i, p in enumerate(packages):
                seq = Dijkstra(grid.graph, action, p[0])
                l = (max(len(seq), p[1] - (self.cost + 1)), i)
                heapq.heappush(lens, l)
            return lens[0] if lens else 0

        sortedActions = []
        for i, action in enumerate(actions):
            heapq.heappush(sortedActions, (SortKey(action), i, action))
        sortedActions = [action for _, _, action in sortedActions]
        return sortedActions

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
        print(f'Starting Search for step {i} for agent {AdversarialAgent.agentNum + 1}')
        AdversarialAgent.agentNum = 1 - AdversarialAgent.agentNum
        otherAgent =([agent for agent in agents if agent != self and isinstance(agent, AdversarialAgent)] or [None])[0]
        assert otherAgent is not None, "No other adversarial agent found"
        actions = self.GetActions(grid)
        visitedStates: dict[State, int] = {}
        alpha = (float('-inf'), float('-inf'), float('inf'))
        beta = (float('inf'), float('inf'), float('inf'))
        self.cost = i
        otherAgent.cost = i
        self.cutOffLimit = AdversarialAgent.cutOffLimit + i
        otherAgent.cutOffLimit = AdversarialAgent.cutOffLimit + i
        self.otherAgent = otherAgent
        self.otherAgent.otherAgent = self
        optionNum = 0
        def Key(action: Node) -> Tuple[int, int, list[Node]]:
            AdversarialAgent.iterations = 0
            nonlocal optionNum
            v = self.MinValue(grid, action, alpha, beta, visitedStates)
            print(f"Option ({optionNum}): Action: {action}, Value: {v[:2]}, iterations: {AdversarialAgent.iterations}\n"
                  f"Seq: {v[2] if v[2] != float('inf') else float('inf')}\n")
            optionNum += 1
            return MaxSortKey(v)
        nextAction = max(actions, key=Key)
        print(f"Next Action: {nextAction}\n\n")
        return [nextAction]

    def SimulateStep(self, grid: Grid, action: Node) -> tuple[Grid, AdversarialAgent]:
        """
        Simulates a step in the game by applying the given action to the grid.

        Args:
            grid (Grid): The current game grid.
            action (Node): The action to be applied to the grid.

        Returns:
            tuple[Grid, AdversarialAgent]: A tuple containing the updated grid and the updated agent.
        """
        nextGrid = copy.deepcopy(grid)
        nextAgent = copy.deepcopy(self)
        nextAgent.cost += 1
        nextAgent.seq.append(action)
        # seq = [(0, 3)]
        # if nextAgent.seq[:len(seq)] == seq:
        #     print(1)
        nextAgent.ProcessStep(nextGrid, (nextAgent.coordinates, action), nextAgent.cost)
        return nextGrid, nextAgent

    def CutoffTest(self, actions: set[Node], state: State) -> bool:
        """
        Determines whether the search should be cut off based on the given parameters.

        Args:
            actions (set[Node]): The set of possible actions.
            state (State): The current state of the game.
            cutOffLimit (int): The maximum depth to search.

        Returns:
            bool: True if the search should be cut off, False otherwise.
        """
        nextAgent = state.agent
        nextOtherAgent = state.otherAgent
        nextGrid = state.grid
        nodes = nextAgent.FormulateGoal(nextGrid, None).union(nextOtherAgent.FormulateGoal(nextGrid, None))
        if self.cost == self.cutOffLimit or not actions or not nodes:
            return True
        return False

    def IsVisited(self, state: State, visitedStates: dict[State, int], action: Node) -> bool:
        """
        Checks if the given state has been visited before.

        Args:
            state (State): The state to check.
            visitedStates (dict[State, int]): The dictionary of visited states.

        Returns:
            bool: True if the state has been visited before, False otherwise.
        """
        if state in visitedStates.keys() and action != self.coordinates:
            if len(visitedStates[state]) <= len(state.agent.seq):
                return True
            del visitedStates[state]
        return False

    def Eval(self, grid: Grid) -> int:
        """
        Evaluates the current state of the game.

        Args:
            otherAgent (AdversarialAgent): The other adversarial agent.

        Returns:
            int: The evaluation of the current state of the game.
        """
        def AgentEval(agent: AdversarialAgent) -> int:
            return agent.score + 0.5 * len(agent.packages) + 0.2 * len(grid.packages)
        selfEval = AgentEval(self)
        otherEval = AgentEval(self.otherAgent)
        return round(selfEval - otherEval, 1), selfEval, self.seq

    def MinValue(self, grid: Grid, action: Node, alpha: float, beta: float, visitedStates: dict[State, int]) -> int:
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
        defaultValue = (float('-inf'), float('-inf'), float('inf'))
        nextGrid, nextAgent = self.SimulateStep(grid, action)
        nextOtherAgent = nextAgent.otherAgent
        nextState = State(nextGrid, nextAgent, nextOtherAgent)
        if self.IsVisited(nextState, visitedStates, action):
            return defaultValue
        visitedStates[nextState] = nextAgent.seq
        AdversarialAgent.iterations += 1

        actions = nextOtherAgent.GetActions(nextGrid)
        if nextAgent.CutoffTest(actions, nextState):
            return nextAgent.Eval(nextGrid)

        v = (float('inf'), float('inf'), float('inf'))
        for nextAction in actions:
            v = min(v, nextAgent.MaxValue(nextGrid, nextAction, alpha, beta, visitedStates),
                    key=MinSortKey)

            flag = False
            if alpha[0] == v[0] and alpha[1] != v[1]:
                flag = True
                print(f"alpha: {alpha[:2]}, v: {v[:2]}")

            if alpha == max(alpha, v, key=MaxSortKey):
                return v

            beta = min(beta, v, key=MinSortKey)
            if flag:
                print(f"new beta: {beta[:2]}\n")

        return v if v[0] != float('inf') else defaultValue

    def MaxValue(self, grid: Grid, action: Node, alpha: float, beta: float, visitedStates: dict[State, int]) -> int:
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
        defaultValue = (float('inf'), float('inf'), float('inf'))
        nextGrid, nextOtherAgent = self.otherAgent.SimulateStep(grid, action)
        nextAgent = nextOtherAgent.otherAgent
        nextState = State(nextGrid, nextAgent, nextOtherAgent)
        if self.otherAgent.IsVisited(nextState, visitedStates, action):
            return defaultValue
        visitedStates[nextState] = nextOtherAgent.seq
        AdversarialAgent.iterations += 1

        actions = nextAgent.GetActions(nextGrid)
        if nextOtherAgent.CutoffTest(actions, nextState):
            return nextAgent.Eval(nextGrid)

        v = (float('-inf'), float('-inf'), float('inf'))

        for nextAction in actions:
            v = max(v, nextAgent.MinValue(nextGrid, nextAction, alpha, beta, visitedStates),
                    key=MaxSortKey)

            flag = False
            if beta[0] == v[0] and beta[1] != v[1] or v[:2] == (float(0), float(1)):
                flag = True
                print(f"beta: {beta[:2]}, v: {v[:2]}")

            if beta == min(beta, v, key=MinSortKey):
                return v
            alpha = max(alpha, v, key=MaxSortKey)
            if flag:
                print(f"new alpha: {alpha[:2]}\n")
        return v if v[0] != float('-inf') else defaultValue

def MinSortKey(eValue: Tuple[int, int, list[Node]]) -> tuple[int, int, int]:
    """
    Returns a tuple used for sorting based on the given eValue.

    Args:
        eValue (Tuple[int, int, list[Node]]): The eValue to be sorted.

    Returns:
        tuple[int, int, int]: A tuple containing the elements of eValue, with the length of the third element
                              being replaced with float('inf') if it is equal to float('inf').

    """
    return (eValue[0], eValue[1], len(eValue[2]) if eValue[2] != float('inf') else float('inf'))

def MaxSortKey(eValue: Tuple[int, int, list[Node]]) -> tuple[int, int, int]:
    """
    Returns a tuple used for sorting based on the maximum value of eValue.

    Args:
        eValue (Tuple[int, int, list[Node]]): The eValue tuple containing three elements.

    Returns:
        tuple[int, int, int]: A tuple used for sorting based on the maximum value of eValue.
    """
    return (eValue[0], eValue[1], -len(eValue[2]) if eValue[2] != float('inf') else float('-inf'))

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
                self.grid.GetPickups(),
                tuple(self.grid.fragEdges))
