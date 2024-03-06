import configparser
from os import path
from grid import Grid
from utils import InitGrid
from agents.agent import Agent
from agents.search_agent import SearchAgent
from agents.adversarial_agent import MultiAgent
from agents.human_agent import HumanAgent

def Main():
    """Main function of the project
    Args:
        argc (int): System Arguments Count
        argv (list[str]): System Arguments
    """
    config = configparser.ConfigParser()
    config.read('config.ini')
    filePath = config['settings'].get('grid_config_path', './tests/test1.txt')
    MultiAgent.cutOffLimit = int(config['settings'].get('cutoff', 10))
    HumanAgent.useButton = config['settings'].get('use_button', 'True') == 'True'
    assert path.exists(filePath), "Path to grid configuration file does not exist!"

    grid: Grid
    agents: list[Agent]
    grid, agents = InitGrid(filePath)
    
    i = 0
    while any(agent.done is not True for agent in agents) and i <= SearchAgent.dropOffTimes[0].dropOffMaxTime:
        for agent in agents:
            action = agent.AgentStep(grid, agents, i)
            agent.ProcessStep(grid, action, i + 1)
        i += 1

    _ = [print(f"Agent {agent.agentNum + 1} Score: {agent.score}\nSequence: {agent.totalseq}\n\n")\
            for agent in agents if isinstance(agent, MultiAgent)]

if __name__ == "__main__":
    Main()
