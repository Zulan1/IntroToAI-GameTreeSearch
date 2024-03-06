# Project Information

This repo tries to solve the MAPD problem using game tree search with alpha-beta pruning algorithms.
This project is using Python 3.11.

## Installation

1. (Optional) Create a venv using: "python3 -m venv venv".
2. (Optional) Activate venv using: "./venv/Scripts/activate"(Windows) or "source ./venv/bin/activate/" on linux
3. (On Linux) Make sure TkAgg backend (already defined in ./agents/human_agent.py) is used and Tkinter is installed on the machine.
4. Install python packages using: "pip install -r requirements.txt"
5. Configure config.ini with grid configuration file, Cutoff limit, and continuous (False to skip pushing continue in order to continue calculating)
6. You can use the available tests ./tests/. (/adversarial, /semi_cooperative, /cooperative).
7. Run the code with "python .".

## Clarification

The Agents available for this projects are:

1. Human Agent (denoted as H in the test file) should always be in the test for debugging.
2. Adversarial Agent (denoted as AD in the test file) an agent solving the problem adversarialy using minimax and alpha beta prunning.
   Tests in ./tests/adversarial section are example for that.
3. Semi Cooperative Agent (denoted as SC in the test file) an agent solving the problem semi cooperatively maxing it's own score breaking ties in favor of other agents.
   Tests in ./tests/semi_cooperative section are example for that.
4. Cooperative Agent (denoted as C in the test file) an agent solving the problem cooperatively maxing the sum of the agents scores breaking ties in favor of it self.
   Tests in ./tests/cooperative section are example for that.







