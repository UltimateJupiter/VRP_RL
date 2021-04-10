import sys
import yaml
import argparse
from dqn.agent import DQN_Agent
from env.utils import log
from environ import make_environ
from reward import VRPReward

import numpy as np
import torch

parser = argparse.ArgumentParser(description='DQN')
parser.add_argument('--config', default='./training_conf/v1_test.yaml', metavar='PATH')
parser.add_argument('--save_dir', default='gym-results/', metavar='PATH')
parser.add_argument('--modelpath', default='', metavar='PATH')
args = parser.parse_args()

network_device = 'cuda' if torch.cuda.is_available() else 'cpu'
memory_device = 'cuda' if torch.cuda.is_available() else 'cpu'

np.random.seed(0)

def main(config):

    reward_rule = VRPReward(**config['reward_params'])
    log("Using Network Device {}".format(network_device))

    M = make_environ(reward_rule=reward_rule,
                     device=network_device,
                     verbose=False,
                     **config['env_params'])

    agent = DQN_Agent(M, config['agent_params'], network_device, memory_device)
    if config['mode'] == 'train':
        agent.train()
        agent.save()
    # if config['mode'] == 'eval':
    #     agent.eval(10)


if __name__ == '__main__':
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    config['save_dir'] = args.save_dir
    if args.modelpath:
        config['modelpath'] = args.modelpath

    if 'use_gpu' in config:
        config['use_gpu'] = config['use_gpu'] and torch.cuda.is_available()
    else:
        config['use_gpu'] = torch.cuda.is_available()
    # Get Atari games.
    print(config)
    main(config)