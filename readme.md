# MA AIRL from scratch

## Tasks
- first implement the expert part of the algorithm
- the see if it is working for different envs
- then implement the inverse RL part of it.

## environement details.
- imprt the marlbase conda env to get the requirements.txt file.

## Running the Code

### Random Policy Evaluation
```bash
python scripts/run_random_policy.py --visualize --food=3
```

### Expert Policy Evaluation
```bash
python scripts/run_expert_policy.py --visualize --food=3
```

### MADDPG
```bash
 python main.py --n_episodes=5000
```

## Note
- used claude's help.

## Issues
- the output of lbforaging's states for each agent is diffferent, but i was assuming that it would be the same for all agents. since for making the Adverserial inverse RL working we need to have the same state for all agents.
