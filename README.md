This repo is built for the adv-ml course project. It contains the MADDPG submodule and multi-agent particle environment submodule from OpenAI, as well as a custom built environment.

Installation
1. Go to root directory of repository
2. Start virtualenv using Python3.6 interpreter
3. pip install -r requirements.txt

---

Option A: Run default MADDPG training (to check out how the agent learning works)
1. Change directory to ./maddpg/experiments
2. python train.py --scenario simple --num-episodes 10000    # train agent
3. python train.py --scenario simple --display   # shows learned agent policy  

Trained models saved in ./maddpg/experiments/tmp/policy  
Reward curve saved in ./maddpg/experiments/learning_curves

---

Option B: Run hyperband hyperparameter optimisation (relevant to our contribution: tuning the hyperparams)
1. Go to root directory of repository
2. python ./experiment/optimise_hyperband.py

Trained models saved in ./experiment/saved_models  
Reward curves for individual hyperparameter configurations saved in ./learning_curves  
Reward curves for each config collected and saved in ./  
Optimisation results (loss, iterations, parameters) saved in ./
