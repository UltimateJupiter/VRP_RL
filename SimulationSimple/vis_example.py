import os, sys
import yaml
import argparse
from dqn.agent import DQN_Agent
from env.utils import log, flatten_dict
from environ import make_environ
from reward import VRPReward
from tqdm import trange
import imageio

from random_agents import UniRandAgent
from env.vis import *

import numpy as np
import torch
import cv2

from datetime import datetime

config_fl = './training_conf/3s_oneflow.yaml'
model_path = './sample_models/3s_oneflow_b1c12/epoch901.pth'
animation_path = './animations/3s_oneflow_b1c12/'
img_info = 'Episode 1000'
T = 400

config = yaml.load(open(config_fl), Loader=yaml.FullLoader)
action_rng = np.random.RandomState(0)

def get_size():
    img = cv2.imread('./dummy.jpg')
    os.remove('./dummy.jpg')
    return img.shape

def main(config):

    os.makedirs(animation_path, exist_ok=True)

    reward_rule = VRPReward(**config['reward_params'])
    log("Using Network Device {}".format('cpu'))
    M = make_environ(reward_rule=reward_rule,
                     device='cpu',
                     verbose=False,
                     **config['env_params'])
    # M.print_events()
    A = UniRandAgent(M)
    agent = DQN_Agent(M, config['agent_params'], 'cpu', 'cpu', config['use_neptune'])
    agent.load(model_path)

    vis_map(M, M.vec, background=False, margin=1.3, info=img_info, im_name='./dummy.jpg')
    img_shape = get_size()
    M.assign_auto_agent(A)
    print(img_shape)
    M.reset()

    video_name = os.path.join(animation_path, "{}.mp4".format(img_info.replace(' ', '')))
    gif_name = os.path.join(animation_path, "{}.gif".format(img_info.replace(' ', '')))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video=cv2.VideoWriter(video_name, fourcc, 8, (img_shape[1], img_shape[0]))
    with imageio.get_writer(gif_name, mode='I') as gif_writer:
        for t in trange(T):
            im_name = os.path.join(animation_path, "{}.jpg".format(t))
            action = agent.act(M.vec_flatten, 0, action_rng)
            M.feedback_step(action)
            # M.step_auto()
            vis_map(M, M.vec, background=False, margin=1.3, info=img_info, im_name=im_name)
            im_vec = cv2.imread(im_name)
            video.write(im_vec)
            gif_writer.append_data(im_vec)
            os.remove(im_name)
    
    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':

    main(config)