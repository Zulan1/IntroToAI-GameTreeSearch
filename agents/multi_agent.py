from __future__ import annotations
import copy
import heapq
from abc import ABC, abstractmethod
from typing import Tuple, Generator, Callable
from agents.search_agent import SearchAgent
from agents.agent import Agent
from grid import Grid
from type_aliases import Node, Edge, MinimaxValueType

class MultiAgent(SearchAgent, ABC):
    """
    A class representing a multi-agent in a game.

    Attributes:
        cutOffLimit (int): The cutoff limit for the search algorithm.
        iterations (int): The number of iterations performed during the search.
        pruneCount (int): The number of pruned nodes during the search.
        visitedCount (int): The number of visited nodes during the search.
        visitedStates (dict[State, int]): A dictionary to store visited states and their values.

    Methods:
        __init__(self, params: list[str], _: Grid) -> None:
            Initializes a MultiAgent object.
        name(self) -> str:
            Returns the name of the agent.
        FormulateGoal(self, grid: Grid, _: int) -> set[Node]:
            Gets all the nodes of packages' pickups or dropoffs.
        GetActions(self, grid: Grid) -> set[Node]:
            Returns a set of possible actions for the agent based on the current grid state.
        SortActions(self, actions: set[Node], grid: Grid) -> list[Node]:
            Sorts the given set of actions based on the current state of the grid.
        Search(self, grid: Grid, _: set[Node], agents: list[Agent], __: int) -> list[Node]:
            Performs a search for the best action to take based on the current state of the grid.
        AlphaBetaSearch(self, state: State, alpha: tuple[float, float, list[Node], list[Node]],
                        beta: tuple[float, float, list[Node], list[Node]]) -> list[Node]:
            Performs an alpha-beta search on the given state.
        SimulateStep(self, grid: Grid, action: Node) -> State:
            Simulates a step in the game by applying the given action to the grid.
        ReverseAttributes(self, alpha: Tuple[float, float, list[Node], list[Node]],
                          beta: Tuple[float, float, list[Node], list[Node]]) -> State:
            Reverses the attributes of the agent.
        MinMaxValue(self, state: State, action: Node, alpha: Tuple[float, float, list[Node], list[Node]],
                    beta: Tuple[float, float, list[Node], list[Node]]) -> int:
            Performs the MinValue step in the minimax algorithm for adversarial search.
    """

    cutOffLimit: int = 11
    iterations: int = 0
    pruneCount: int = 0
    visitedCount: int = 0
    visitedStates: dict[State, Tuple[list[Node], MinimaxValueType]] = {}

    def __init__(self, params: list[str], _: Grid) -> None:
        super().__init__(params, _)
        self.cutoff = 0
        self.agentNum: int
        self.maxKeyFunc: Callable
        self.allowPruning: bool

    @property
    def name(self):
        """Returns the name of the agent."""
        return f'Agent {self.agentNum + 1}'

    def FormulateGoal(self, grid: Grid, _: int) -> set[Node]:
        """Gets all the nodes of packages' pickups or dropoffs

        Args:
            grid (Grid): The Simulator's grid
            agent (GreedyAgent): The agent

        Returns:
            set[Node]: All the grid's pickup and dropoff locations
        """
        return set(grid.packages).union({p.dropoffLoc for s in self.packages.values() for p in s})

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
        packages = grid.GetPickups() + tuple((p0,0) for p0,_ in self.GetDropdowns())
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

    def Search(self, grid: Grid, _: set[Node], agents: list[Agent], __: int) -> list[Node]:
        """
        Performs a search for the best action to take based on the current state of the grid.

        Args:
            grid (Grid): The current state of the grid.
            _: set[Node]: Unused parameter.
            agents (list[Agent]): The list of agents in the game.
            __: int: Unused parameter.

        Returns:
            list[Node]: A list containing the best action to take.

        Raises:
            AssertionError: If no other adversarial agent is found in the list of agents.
        """

        print(f'Starting Search for step {self.cost} for agent {self.agentNum + 1}')
        otherAgent = ([agent for agent in agents if agent != self and isinstance(agent, MultiAgent)] or [None])[0]
        assert otherAgent is not None, "No other adversarial agent found"
        MultiAgent.visitedStates: dict[State, Tuple[list[Node], MinimaxValueType]] = {}
        alpha: MinimaxValueType = (float('-inf'), float('-inf'), None, None)
        negBeta: MinimaxValueType = (float('-inf'), float('-inf'), None, None)
        self.cutoff: int = MultiAgent.cutOffLimit + self.cost
        otherAgent.cutoff = MultiAgent.cutOffLimit + otherAgent.cost
        state: State = State(grid, self, otherAgent)
        return self.AlphaBetaSearch(state, alpha, negBeta)

    @abstractmethod
    def ToDebugFormat(self, v: MinimaxValueType) -> MinimaxValueType:
        """
        Converts the given value to a debug format.

        Args:
            v (MinimaxValueType): The value to be converted.

        Returns:
            MinimaxValueType: The value in debug format.
        """

    def AlphaBetaSearch(self, state: State, alpha: MinimaxValueType, negBeta: MinimaxValueType) -> list[Node]:
        """
        Performs an alpha-beta search on the given state.

        Args:
            state (State): The state to perform the search on.
            alpha (tuple[float, float, list[Node], list[Node]]): The alpha value for the search.
            beta (tuple[float, float, list[Node], list[Node]]): The beta value for the search.

        Returns:
            tuple[float, float, list[Node], list[Node]]:
                The value of the search, the best action, and the best opponent action.
        """
        grid: Grid
        agent: MultiAgent
        grid, agent = state.grid, state.agent
        actions = agent.GetActions(grid)
        optionNum = 0
        def Key(action: Node) -> Tuple[int, int, list[Node]]:
            nonlocal optionNum
            MultiAgent.pruneCount, MultiAgent.visitedCount, MultiAgent.iterations = 0, 0, 0
            v = self.ReverseV(self.MaxValue(state, action, alpha, negBeta))
            vInDebug = self.ToDebugFormat(v)
            print(f"Option ({optionNum}): Action: {action}, Value: {vInDebug[:2]}, iterations: {MultiAgent.iterations}"
                  f", Prune Count: {MultiAgent.pruneCount}, Visited Count: {MultiAgent.visitedCount}\n"
                  f"Seq: {vInDebug[2]}\nOpponent Predicted Seq: {vInDebug[3]}\n")
            optionNum += 1
            return self.maxKeyFunc(v)
        nextAction = max(actions, key=Key) if len(actions) > 1 else actions[0]
        print(f"Next Action: {nextAction}\n\n\n")
        return [nextAction]

    def SimulateStep(self, grid: Grid, action: Node) -> State:
        """
        Simulates a step in the game by applying the given action to the grid.

        Args:
            grid (Grid): The current state of the game grid.
            action (Node): The action to be applied to the grid.

        Returns:
            tuple: A tuple containing the next state of the grid and the updated agent.
        """
        self.cost += 1
        self.seq.append(action)
        self.ProcessStep(grid, (self.coordinates, action), self.cost)

    def AgentEval(self, grid: Grid) -> float:
        """
        Evaluates the current state of the agent.

        Args:
            grid (Grid): The grid representing the current state.

        Returns:
            float: The evaluation score of the agent.
        """
        return self.score + 0.5 * len([p for p, d in self.GetDropdowns() if d >= self.cost]) +\
            0.2 * len([p for p, d in grid.GetDropdowns() if d >= self.cost])

    @abstractmethod
    def Eval(self, state: State) -> list[float, float, list[Node], list[Node]]:
        """
        Evaluate the given state and return a tuple containing the evaluation score,
        the maximum score, a list of nodes, and a list of nodes.

        Args:
            state (State): The state to be evaluated.

        Returns:
            Tuple[float, float, list[Node], list[Node]]: A tuple containing the evaluation score,
            the maximum score, a list of nodes, and a list of nodes.
        """
    @abstractmethod
    def ReverseV(self, v):
        """
        Reverses the value 'v'.

        Parameters:
        v (any): The value to be reversed.

        Returns:
        any: The reversed value.
        """

    def MaxValue(self, state: State, action: Node, alpha: MinimaxValueType, negBeta: MinimaxValueType) -> int:
        """
        Performs the MinValue step in the minimax algorithm for adversarial search.

        Args:
            action (Node): The action to be evaluated.
            alpha (float): The alpha value for alpha-beta pruning.
            beta (float): The beta value for alpha-beta pruning.

        Returns:
            int: The utility value of the action.
        """
        nextGrid: Grid
        nextAgent: MultiAgent
        nextOtherAgent: MultiAgent
        nextState: State = state.Successor()
        nextGrid, nextAgent, nextOtherAgent = nextState
        nextGrid = nextState.grid
        nextAgent.SimulateStep(nextGrid, action)
        if self.coordinates != action and nextState.IsVisited():
            retval = MultiAgent.visitedStates[nextState][1]
            seqPrefix = nextAgent.seq
            newSeq = seqPrefix + retval[3][len(seqPrefix):]
            retval[3] = newSeq
            return retval
        MultiAgent.iterations += 1

        actions = nextOtherAgent.GetActions(nextGrid)
        if nextState.CutoffTest(actions):
            v = nextAgent.Eval(nextState)
            return v if len(nextAgent.seq) == len(nextOtherAgent.seq) else self.ReverseV(v)

        v: MinimaxValueType = (float('-inf'), float('-inf'), float('-inf'), None, None)

        for nextAction in actions:
            maxValue = self.ReverseV(self.MaxValue(nextState, nextAction, negBeta, alpha))
            v = max(v, maxValue, key=self.maxKeyFunc)
            if self.allowPruning:
                if negBeta == max(negBeta, self.ReverseV(v), key=self.maxKeyFunc):
                    MultiAgent.pruneCount += 1
                    return v
                alpha = max(alpha, v, key=self.maxKeyFunc)

        MultiAgent.visitedStates[nextState] = (nextAgent.seq, v)
        return v


class State:
    """
    Represents a state in the multi-agent game.

    Attributes:
        grid (Grid): The grid object representing the game board.
        agent (MultiAgent): The agent object representing the main agent.
        otherAgent (MultiAgent): The agent object representing the other agent.
        reversed (bool): Indicates whether the state is reversed or not.
    """

    def __init__(self, grid: Grid, agent: MultiAgent, otherAgent: MultiAgent):
        self.grid: Grid = grid
        self.agent: MultiAgent = agent
        self.otherAgent: MultiAgent = otherAgent
        self.reversed = True

    def __hash__(self):
        return hash(self.ToBaseClasses())

    def __eq__(self, other: State):
        return self.ToBaseClasses() == other.ToBaseClasses()

    def __iter__(self) -> Generator[Grid, MultiAgent, MultiAgent]:
        yield self.grid
        if self.reversed:
            yield self.otherAgent
            yield self.agent
        else:
            yield self.agent
            yield self.otherAgent

    def ToBaseClasses(self) ->\
        Tuple[Tuple[Node, int], Tuple[Node, Tuple[Node, int]], Tuple[Node ,Tuple[Node, int]], Tuple[Edge]]:
        """
        Converts the current state of the multi-agent to a tuple of base classes.

        Returns:
            Tuple[Tuple[Node, int], Tuple[Node, Tuple[Node, int]], Tuple[Node ,Tuple[Node, int]], Tuple[Edge]]:
            A tuple containing the coordinates, pickups, dropdowns, and edges of the multi-agent.
        """
        agent: MultiAgent
        otherAgent: MultiAgent
        grid, agent, otherAgent = self
        return (agent.coordinates, agent.GetDropdowns(),
                otherAgent.coordinates, otherAgent.GetDropdowns(),
                grid.GetPickups(), agent.score, otherAgent.score,
                len(agent.seq), len(otherAgent.seq),
                tuple(grid.fragEdges))

    def Successor(self):
        """
        Creates a successor object by creating a deep copy of the
        current object and reversing its state.

        Returns:
            The successor object.
        """
        successor = copy.deepcopy(self)
        successor.reversed = not self.reversed
        return successor

    def IsVisited(self) -> bool:
        """
        Checks if the current state has been visited before.

        Returns:
            bool: True if the state has been visited, False otherwise.
        """
        if self in MultiAgent.visitedStates:
            # agent = self.agent if not self.reversed else self.otherAgent
            # if len(MultiAgent.visitedStates[self][0]) == len(agent.seq):
            MultiAgent.visitedCount += 1
            return True
        return False

    def CutoffTest(self, actions: set[Node]) -> bool:
        """
        Checks if the cutoff test condition is satisfied.

        Args:
            actions (set[Node]): The set of possible actions.

        Returns:
            bool: True if the cutoff test condition is satisfied, False otherwise.
        """

        nextGrid: Grid
        nextAgent: MultiAgent
        nextOtherAgent: MultiAgent
        nextGrid, nextAgent, nextOtherAgent = self
        nodes = nextAgent.FormulateGoal(nextGrid, None).union(nextOtherAgent.FormulateGoal(nextGrid, None))
        return self.otherAgent.cost == self.otherAgent.cutoff or not actions or not nodes
