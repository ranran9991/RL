# RL
Project for RL course

## Requirements:
- pytorch
- gym

## Running the training:

run `python main.py` with the correct algorithms 
```
usage: main.py [-h] [--agent AGENT] [--num_iterations NUM_ITERATIONS]
               [--eval_every EVAL_EVERY] [--noisy]

optional arguments:
  -h, --help            show this help message and exit
  --agent AGENT         The type of agent to use
  --num_iterations NUM_ITERATIONS
                        The number of episodes our agent will train on
  --eval_every EVAL_EVERY
                        Run an evaluation after <> episodes
  --noisy               Whether the environment is noisy or not
```

The options for the agent are `td`, `ddqn`, `dueling`, `rainbow`.

## Example usage:
```
python main.py --agent ddqn --num_iterations 1000 --eval_every 50
```
which runs the ddqn agent with 1000 episodes, evaluating every 50 (the environment is not noisy)
```
python main.py --agent rainbow --num_iterations 1200 --eval_every 50 --noisy
```
which runs the rainbow agent with 1200 episodes, evaluating every 50 on a noisy environment
