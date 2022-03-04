import pickle
from argparse import Namespace
from experiments.train import train
import tensorflow as tf

def run_trial(scenario, iterations, episodes, train_params):
    # train_params is a dictionary containing lr, gamma, batch_size, and num_units as keys
    # args is a Namespace object containing arguments as attributes as described below

    # # Environment
    # parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    # parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    # parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    # parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    # parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    # parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # # Core training parameters
    # parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    # parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    # parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    # parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # # Checkpointing
    # parser.add_argument("--exp-name", type=str, default='unnamed_exp', help="name of the experiment")
    # parser.add_argument("--save-dir", type=str, default="./tmp/policy/", help="directory in which training state and model should be saved")
    # parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    # parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # # Evaluation
    # parser.add_argument("--restore", action="store_true", default=False)
    # parser.add_argument("--display", action="store_true", default=False)
    # parser.add_argument("--benchmark", action="store_true", default=False)
    # parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    # parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    # parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    assert train_params and train_params['lr'] and train_params['gamma'] and train_params['batch_size'] and train_params['num_units'],\
    'train_params passed to run_trail() should contain lr, gamma, batch_size, and num_units, only {} were passed'\
    .format(str(train_params.keys())) # assert that all experiment parameters are provided by the caller of run_trail()
    experiment_name = '{}_{}_{}_{}_{}'.format(scenario, train_params['lr'], train_params['gamma'], train_params['batch_size'], train_params['num_units'])
    plots_dir='./learning_curves/'
    args = Namespace(
        # experiment parameters
        lr=train_params['lr'],
        gamma=train_params['gamma'],
        batch_size=train_params['batch_size'],
        num_units=train_params['num_units'],

        # custom arguments
        scenario=scenario,
        num_adversaries=0,
        num_episodes=iterations*episodes,
        exp_name=experiment_name,
        save_dir='./experiment/saved_models/{}'.format(experiment_name),
        save_rate=episodes, # save once for each trial, saves overhead
        plots_dir=plots_dir,

        # default arguments
        max_episode_len=25,
        good_policy='maddpg',
        adv_policy='maddpg',
        load_dir='',
        restore=True,
        display=False,
        benchmark=False,
        benchmark_iters=100000,
        benchmark_dir='./benchmark_files'
    )
    tf.reset_default_graph()
    train(args)

    # if scenario == 'simple':
    #     # there is only one agent
    #     with open(plots_dir+experiment_name+'_rewards.pkl', 'rb') as f:
    #         reward_curve=pickle.load(f)
        
    #     return reward_curve[-1]
    # else:
    #     # there are multiple agents
    #     with open(plots_dir+experiment_name+'_agrewards.pkl', 'rb') as f:
    #         reward_curve=pickle.load(f)
        
    #     # IMPORTANT: this only works when num_episodes=save_rate since data is only saved once and thus 
    #     # there is only one reward value for each agent in the reward_curve
    #     return sum(reward_curve)/len(reward_curve) # compute mean
    with open(plots_dir+experiment_name+'_rewards.pkl', 'rb') as f:
            reward_curve=pickle.load(f)
        
    return reward_curve[-1]

# example run
if __name__ == '__main__':
    print(run_trial('simple_tag', 2000, {'lr':0.02, 'gamma':0.95, 'batch_size':1024, 'num_units':64}))