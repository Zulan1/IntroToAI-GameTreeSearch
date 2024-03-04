from __future__ import annotations
import copy
import heapq
from typing import Tuple, Generator
from agents.search_agent import SearchAgent
from agents.agent import Agent
from grid import Grid
from type_aliases import Node, Edge

class AdversarialAgent(SearchAgent):
    """Adversarial Agent Class

    Args:
        Agent (Agent): inherits from Agent class
    """
    cutOffLimit: int = 11
    iterations: int = 0
    pruneCount: int = 0
    visitedCount: int = 0
    visitedStates: dict[State, int] = {}

    def __init__(self, params: list[str], _: Grid) -> None:
        super().__init__(params, _)
        self.cutoff = 0
        self.agentNum: int

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
        otherAgent = ([agent for agent in agents if agent != self and isinstance(agent, AdversarialAgent)] or [None])[0]
        assert otherAgent is not None, "No other adversarial agent found"
        actions = self.GetActions(grid)
        AdversarialAgent.visitedStates: dict[State, int] = {}
        alpha = (float('-inf'), float('-inf'), None, None)
        beta = (float('inf'), float('inf'), None, None)
        self.cutoff = AdversarialAgent.cutOffLimit + self.cost
        otherAgent.cutoff = AdversarialAgent.cutOffLimit + otherAgent.cost
        state = State(grid, otherAgent, self)
        optionNum = 0
        def Key(action: Node) -> Tuple[int, int, list[Node]]:
            AdversarialAgent.pruneCount, AdversarialAgent.visitedCount, AdversarialAgent.iterations = 0, 0, 0
            nonlocal optionNum
            v = state.MinValue(action, alpha, beta)
            print(f"Option ({optionNum}): Action: {action}, Value: {v[:2]}, iterations: {AdversarialAgent.iterations}, "
                  f"Prune Count: {AdversarialAgent.pruneCount}, Visited Count: {AdversarialAgent.visitedCount}\n"
                  f"Seq: {v[2]}\nOpponent Predicted Seq: {v[3]}\n")
            optionNum += 1
            return MaxSortKey(v)
        nextAction = max(actions, key=Key) if len(actions) > 1 else actions[0]
        print(f"Next Action: {nextAction}\n\n\n")
        return [nextAction]

def MinSortKey(eValue: Tuple[int, int, list[Node]]) -> tuple[int, int, int]:
    """
    Returns a tuple used for sorting based on the given eValue.

    Args:
        eValue (Tuple[int, int, list[Node]]): The eValue to be sorted.

    Returns:
        tuple[int, int, int]: A tuple containing the elements of eValue, with the length of the third element
                              being replaced with float('inf') if it is equal to float('inf').

    """
    return (eValue[0], eValue[1], len(eValue[2]) if eValue[2] else float('inf'))

def MaxSortKey(eValue: Tuple[int, int, list[Node]]) -> tuple[int, int, int]:
    """
    Returns a tuple used for sorting based on the maximum value of eValue.

    Args:
        eValue (Tuple[int, int, list[Node]]): The eValue tuple containing three elements.

    Returns:
        tuple[int, int, int]: A tuple used for sorting based on the maximum value of eValue.
    """
    return (eValue[0], eValue[1], -len(eValue[2]) if eValue[2] else float('-inf'))

class State:
    """
    Represents the state of the game for the adversarial agent.

    Attributes:
        grid (Grid): The grid representing the game board.
        agent (AdversarialAgent): The adversarial agent.
        otherAgent (AdversarialAgent): The other adversarial agent.

    Methods:
        __init__(self, grid: Grid, agent: AdversarialAgent, otherAgent: AdversarialAgent):
        Initializes a new instance of the State class.

        __hash__(self): Returns the hash value of the state.

        __eq__(self, other: State): Checks if two states are equal.

        __iter__(self) -> Generator[Grid, AdversarialAgent, AdversarialAgent]:
        Returns an iterator over the state's components.

        ToBaseClasses(self) ->
        Tuple[Tuple[Node, int], Tuple[Node, Tuple[Node, int]], Tuple[Node ,Tuple[Node, int]], Tuple[Edge]]:
        Converts the state to a tuple of base classes.

        Eval(self, isOther=False) -> Tuple[int, int, list[Node]]: Evaluates the current state of the agent.
        
        SimulateStep(self, action: Node, isOther=False) -> State:
        Simulates a step in the game by applying the given action to the current state.

        CutoffTest(self, actions: set[Node], isOther=False) -> bool:
        Determines whether the search should be cut off at the current state.

        IsVisited(self, isOther=False) -> bool: Checks if the current state has been visited before.

        MinValue(self, action: Node, alpha: float, beta: float) -> int:
        Performs the MinValue step in the minimax algorithm for adversarial search.

        MaxValue(self, action: Node, alpha: float, beta: float) -> int:
        Computes the maximum value for the current agent in the game tree search.
    """
    def __init__(self, grid: Grid, agent: AdversarialAgent, otherAgent: AdversarialAgent):
        self.grid: Grid = grid
        self.agent: AdversarialAgent = agent
        self.otherAgent: AdversarialAgent = otherAgent
        self.reverse = True

    def __hash__(self):
        return hash(self.ToBaseClasses())

    def __eq__(self, other: State):
        return self.ToBaseClasses() == other.ToBaseClasses()

    def __iter__(self) -> Generator[Grid, AdversarialAgent, AdversarialAgent]:
        yield self.grid
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
        return (self.agent.coordinates, self.agent.GetDropdowns(),
                self.otherAgent.coordinates, self.otherAgent.GetDropdowns(),
                self.grid.GetPickups(), self.agent.score, self.otherAgent.score,
                tuple(self.grid.fragEdges))

    def Successor(self):
        """
        Creates a successor object by creating a deep copy of the
        current object and reversing its state.

        Returns:
            The successor object.
        """
        successor = copy.deepcopy(self)
        successor.agent, successor.otherAgent = successor.otherAgent, successor.agent
        successor.reverse = not self.reverse
        return successor

    def Eval(self) -> int:
        """
        Evaluates the current state of the agent.

        Args:
            isOther (bool, optional): Indicates whether the evaluation is for the other agent. Defaults to False.

        Returns:
            Tuple[int, int, list[Node]]: A tuple containing the difference in
            evaluation scores between the agent and the other agent,
            the evaluation score of the agent, and the sequence of nodes representing the agent's actions.
        """

        grid: Grid
        agent: AdversarialAgent
        otherAgent: AdversarialAgent
        grid, agent, otherAgent = self

        def AgentEval(agentToEval: AdversarialAgent) -> float:
            return agentToEval.score + 0.5 * len(sum(agentToEval.packages.values(), [])) +\
                0.2 * len(sum(grid.packages.values(), []))

        selfEval: float = AgentEval(agent)
        otherEval: float = AgentEval(otherAgent)
        seq: list[Node] = agent.seq
        otherSeq: list[Node] = otherAgent.seq
        diffVal, selfVal = (selfEval - otherEval, selfEval) if not self.reverse else (otherEval - selfEval, otherEval)

        return round(diffVal, 1), round(selfVal, 1), seq, otherSeq

    def SimulateStep(self, action: Node) -> State:
        """
        Simulates a step in the game by applying the given action to the current state.

        Args:
            action (Node): The action to be applied to the current state.
            isOther (bool, optional): Indicates whether the action is for the other agent. Defaults to False.

        Returns:
            tuple[State]: The next state after applying the action.
        """
        nextGrid: Grid
        nextAgent: AdversarialAgent
        # nextOtherAgent: AdversarialAgent
        nextState = self.Successor()
        nextGrid, nextAgent, _ = nextState
        nextAgent.cost += 1
        nextAgent.seq.append(action)

        nextAgent.ProcessStep(nextGrid, (nextAgent.coordinates, action), nextAgent.cost)
        return nextState

    def CutoffTest(self, actions: set[Node]) -> bool:
        """
        Determines whether the search should be cut off at the current state.

        Args:
            actions (set[Node]): The set of possible actions at the current state.
            isOther (bool, optional): Flag indicating whether the current agent is the "other" agent. Defaults to False.

        Returns:
            bool: True if the search should be cut off, False otherwise.
        """

        nextGrid: Grid
        nextAgent: AdversarialAgent
        nextOtherAgent: AdversarialAgent
        nextGrid, nextAgent, nextOtherAgent = self
        cost = nextAgent.cost if self.reverse else nextOtherAgent.cost
        nodes = nextAgent.FormulateGoal(nextGrid, None).union(nextOtherAgent.FormulateGoal(nextGrid, None))
        if cost == nextAgent.cutoff or not actions or not nodes:
            return True
        return False

    def IsVisited(self) -> bool:
        """
        Checks if the current state has been visited before.

        Parameters:
        - isOther (bool): Indicates whether to check for the current agent or the other agent.

        Returns:
        - bool: True if the state has been visited before, False otherwise.
        """

        if self in AdversarialAgent.visitedStates:
            if len(AdversarialAgent.visitedStates[self][0]) <= len(self.agent.seq):
                AdversarialAgent.visitedCount += 1
                return True
        return False


    def MinValue(self, action: Node, alpha: float, beta: float) -> int:
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
        # nextAgent: AdversarialAgent
        nextOtherAgent: AdversarialAgent
        defaultValue = (float('-inf'), float('-inf'), None, None)
        nextState = self.SimulateStep(action)
        nextGrid, _, nextOtherAgent = nextState
        if self.otherAgent.coordinates != action and nextState.IsVisited():
            return AdversarialAgent.visitedStates[nextState][1]
        AdversarialAgent.iterations += 1

        actions = nextOtherAgent.GetActions(nextGrid)
        if nextState.CutoffTest(actions):
            return nextState.Eval()

        v = (float('inf'), float('inf'), None, None)
        for nextAction in actions:
            maxValue = nextState.MaxValue(nextAction, alpha, beta)
            v = min(v, maxValue, key=MinSortKey)
            if alpha == max(alpha, v, key=MaxSortKey):
                AdversarialAgent.pruneCount += 1
                return v
            beta = min(beta, v, key=MinSortKey)

        AdversarialAgent.visitedStates[nextState] = (v[2], v)
        return v if v[0] != float('inf') else defaultValue

    def MaxValue(self, action: Node, alpha: float, beta: float) -> int:
        """
        Computes the maximum value for the current agent in the game tree search.

        Args:
            action (Node): The action to be taken by the agent.
            alpha (float): The alpha value for alpha-beta pruning.
            beta (float): The beta value for alpha-beta pruning.

        Returns:
            int: The maximum value for the current agent.
        """
        nextGrid: Grid
        # nextAgent: AdversarialAgent
        nextOtherAgent: AdversarialAgent
        defaultValue = (float('inf'), float('inf'), None, None)
        nextState: State = self.SimulateStep(action)
        nextGrid, _, nextOtherAgent = nextState
        if self.otherAgent.coordinates != action and nextState.IsVisited():
            return AdversarialAgent.visitedStates[nextState][1]
        AdversarialAgent.iterations += 1

        actions = nextOtherAgent.GetActions(nextGrid)
        if nextState.CutoffTest(actions):
            return nextState.Eval()

        v = (float('-inf'), float('-inf'), None, None)

        for nextAction in actions:
            minValue = nextState.MinValue(nextAction, alpha, beta)
            v = max(v, minValue, key=MaxSortKey)
            if beta == min(beta, v, key=MinSortKey):
                AdversarialAgent.pruneCount += 1
                return v
            alpha = max(alpha, v, key=MaxSortKey)

        AdversarialAgent.visitedStates[nextState] = (v[2], v)
        return v if v[0] != float('-inf') else defaultValue
