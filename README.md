This repo is built for the adv-ml course project. It contains the MADDPG submodule and multi-agent particle environment submodule from OpenAI, as well as a custom built environment.

How to use:
1. Go to root directory of repository
2. Start virtualenv using Python3.5 interpreter
3. pip install -r requirements.txt
4. Change directory to ./maddpg/experiments
5. python train.py --scenario simple --num-episodes 10000    # train agent
6. python train.py --scenario simple --display   # shows learned agent policy
