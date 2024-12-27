# MA AIRL from scratch

## Tasks
- ~~first implement the expert part of the algorithm~~
- ~~implement the MADDPG part of the algorithm~~
- make the inverse RL part working.
- the see if it is working for different envs

## environement details.
- ~~imprt the marlbase conda env to get the requirements.txt file.~~

To create the conda environment, run the following command:
```bash
conda env create --name recoveredenv --file environment.yml
```

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
