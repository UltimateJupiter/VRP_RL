import os, sys
import yaml
import argparse
from dqn.agent import DQN_Agent
from env.utils import log, flatten_dict
from environ import make_environ
from reward import VRPReward

import numpy as np
import torch

import neptune

parser = argparse.ArgumentParser(description='DQN')
parser.add_argument('--config', default='./training_conf/v1_test.yaml', metavar='PATH')
parser.add_argument('--save_dir', default='gym-results/', metavar='PATH')
parser.add_argument('--modelpath', default='', metavar='PATH')
args = parser.parse_args()

network_device = 'cuda' if torch.cuda.is_available() else 'cpu'
memory_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_params_dict(config):
    ret = flatten_dict(config['reward_params'])
    ret.update(flatten_dict(config['agent_params']))
    ret.update(flatten_dict(config['env_params']))
    return ret

def main(config):

    reward_rule = VRPReward(**config['reward_params'])
    log("Using Network Device {}".format(network_device))

    M = make_environ(reward_rule=reward_rule,
                     device=network_device,
                     verbose=False,
                     **config['env_params'])
    # M.print_events()
    config['use_neptune'] = config['use_neptune'] and config['mode'] == 'train'

    if config['use_neptune']:
        nep_args = config['neptune_params']
        token = os.environ['NEPTUNE_API_TOKEN']
        proj_name = nep_args['proj_name']
        neptune.init(project_qualified_name=proj_name, api_token=token)
        neptune.create_experiment(params=make_params_dict(config), upload_source_files=[args.config])

    agent = DQN_Agent(M, config['agent_params'], network_device, memory_device, config['use_neptune'])
    if config['mode'] == 'train':
        agent.train()
    # if config['mode'] == 'eval':
    #     agent.eval(10)


if __name__ == '__main__':

    for x in open(args.config):
        print(x[:-1])

    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    config['save_dir'] = args.save_dir
    if args.modelpath:
        config['modelpath'] = args.modelpath

    if 'use_gpu' in config:
        config['use_gpu'] = config['use_gpu'] and torch.cuda.is_available()
    else:
        config['use_gpu'] = torch.cuda.is_available()
    main(config)