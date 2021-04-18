import os
import numpy as np
from copy import deepcopy
from itertools import count

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from IPython.core.debugger import Tracer; debug_here = Tracer()

from .models.BDQN import BranchingQNetwork
from .memory import Transition, ReplayMemory
from .schedule import LinearSchedule
from .env.playground import Map
from .env.utils import log
from .env.vis import vis_map

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class DQN_Agent(object):

    def __init__(self, M : Map, args, network_device, memory_device, use_neptune=False):

        if use_neptune:
            import neptune

        log("Initializing Agent")
        self.M = M
        self.dev = M.dev
        self.args = args

        self.info_list = ['queue', 'wait', 'opr', 'efficiency', 'invalid_args']
        assert self.dev == network_device

        self.state_size = M.state_dim
        self.n_action, self.action_size = M.action_dim

        # setup controller model and replay memory
        self.use_neptune = use_neptune
        self.setup_controller()

        self.memory = ReplayMemory(
            self.args['replay_memory_cap'],
            memory_device,
            state_size=self.state_size,
            n_action=self.n_action,
            hist_len=self.args['hist_len'])

        self.epsilon = LinearSchedule(
            schedule_timesteps=self.args['epsilon']['end_t'],
            initial_p=self.args['epsilon']['start'],
            final_p=self.args['epsilon']['end'])

        self.gamma = self.args['gamma']   # discount rate

        self.optimizer = optim.RMSprop(self.Qnet.parameters(), **self.args['optim_params'])

        self.stats = {"reward": []}
        self.stats.update({arg: [] for arg in self.info_list})
        self.save_dir = args['save_dir']

    def setup_controller(self):
        log("State size: {}, Action size: {}, #Actions/Frame {}".format(self.state_size, self.action_size, self.n_action))
        self.Qnet = BranchingQNetwork(self.state_size, self.n_action, self.action_size)
        # self.Qnet = NET(input_size=self.state_size, output_size=self.action_size)
        self.target_Qnet = deepcopy(self.Qnet)
        if self.dev != 'cpu':
            self.Qnet = self.Qnet.cuda()
            self.target_Qnet = self.target_Qnet.cuda()
        print(self.Qnet)

    def random_act_constrained(self, state):

        vec_bus, vec_station, t = self.M.unflatten_vec(state)
        schedule = torch.zeros(self.M.n_bus, device=self.dev) + self.M.n_routes
        for i, bus_info in enumerate(vec_bus):
            active, _, loc = bus_info[:3]
            if active:
                schedule[i] = self.M.n_routes
            else:
                r_ind = np.random.choice(self.M.routes_choice[loc.item()] + [self.M.n_routes])
                schedule[i] = r_ind
        return schedule
    
    def random_act(self, state):

        vec_bus, vec_station, t = self.M.unflatten_vec(state)
        schedule = torch.zeros(self.M.n_bus, device=self.dev)
        actions = list(range(self.M.n_routes))

        for i, bus_info in enumerate(vec_bus):
            r_ind = np.random.choice(actions + [self.M.n_routes])
            schedule[i] = r_ind
        return schedule

    def act(self, state, epsilon, rng):
        # select epsilon-greedy action
        # return self.random_act_constrained(state)
        # if np.random.random() <= epsilon / 5:
        # return self.random_act(state)
        if rng.random() <= epsilon:
            return self.random_act_constrained(state)
        else:
            state = Variable(state).unsqueeze(0)
            act_values = self.Qnet.forward(state)   # B*n_actions
            _, action = act_values.data.max(-1)
            return action[0]

    def replay(self):
        
        batch_size = self.args['batch_size']

        if len(self.memory) < batch_size:
            return

        batch = self.memory.sample(batch_size, out_device=self.dev)
        state_batch = Variable(batch.state)
        action_batch = Variable(batch.action)
        reward_batch = Variable(batch.reward)
        next_states = Variable(batch.next_state, requires_grad=False)

        # Compute Q(s_t, a)
        Q_state_action_raw = self.Qnet(state_batch)
        action_batch_reshaped = action_batch.unsqueeze(-1)
        Q_state_action = Q_state_action_raw.gather(-1, action_batch_reshaped).squeeze(-1)

        with torch.no_grad():
            # Double DQN - Compute V(s_{t+1}) for all next states.
            # V_next_state = Variable(torch.zeros(batch_size).type(Tensor))
            if self.args['double_dqn']:
                Q_next = self.Qnet(next_states)
                target_Q_next = self.target_Qnet(next_states)
                next_state_actions = torch.argmax(Q_next, dim=-1)
                V_next_state = target_Q_next.gather(-1, next_state_actions.unsqueeze(-1)).squeeze(-1)
                V_next_state = V_next_state.mean(1, keepdim=True)
            else:
                Q_next = self.Qnet(next_states)
                V_next_state = Q_next.max(-1)[0]
                V_next_state = V_next_state.mean(1, keepdim=True)

        # Compute the target Q values
        target_Q_state_action = reward_batch.unsqueeze(-1) + (self.gamma * V_next_state)
        target_Q_state_action = target_Q_state_action.repeat(1, self.n_action)
        if self.args['loss'] == 'mse':
            loss = F.mse_loss(Q_state_action, target_Q_state_action)
        else:
            loss = F.smooth_l1_loss(Q_state_action, target_Q_state_action)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        self.Qnet.adv_gradient_rescale()

        for param in self.Qnet.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def verbose_feedback(self, reward, feedback, info=''):
        res = info
        res += ' | reward: {:.5g} '.format(reward)
        for i, item in enumerate(feedback):
            res += ' {}: {:.4g} '.format(self.info_list[i], item)
        print(res)

    def train(self):
        
        num_updates = 0
        total_steps = 0

        action_rng = np.random.RandomState(0)

        for e in range(self.args['episodes']):

            log("Episode {}".format(e), color='red')
            if self.args['fix_date']:
                np.random.seed(0)
            else:
                np.random.seed(e)
            self.M.reset()

            episode_rewards = []
            episode_feedback = []

            # Iterate over frames of the episode
            for t in count():
                # get current state
                # vis_map(self.M, self.M.vec)
                state = self.M.vec_flatten
                if t > self.args['replay_start']:
                    action = self.act(state, self.epsilon.value(e), action_rng)
                else:
                    action = self.act(state, 1, action_rng)

                next_state, reward, done, feedback, scale = self.M.feedback_step(action)
                
                if t % self.args['reward_verbose_freq'] == 0:
                    self.verbose_feedback(reward, feedback, "\te{} t{} s{}".format(e, t, total_steps))
                
                if done:
                    mean_reward = np.mean(episode_rewards)
                    self.stats["reward"].append(mean_reward)
                    mean_feedback = np.mean(np.vstack(episode_feedback), axis=0)
                    for i, item in enumerate(mean_feedback):
                        self.stats[self.info_list[i]].append(item)

                    self.verbose_feedback(mean_reward, mean_feedback, "\nMEAN Episode {}".format(e))
                    self.verbose_feedback(mean_reward, mean_feedback * scale, "MEAN  Scaled {}".format(e))
                    print("exploration {:.4g}\n".format(self.epsilon.value(e)))
                    
                    if self.use_neptune:
                        neptune.log_metric('reward', e, mean_reward)
                        for i, item in enumerate(mean_feedback):
                            neptune.log_metric(self.info_list[i], item)
                    break
                
                self.memory.push(Transition(state, action, reward, next_state))
                episode_rewards.append(reward)
                episode_feedback.append(feedback)

                if total_steps > self.args['replay_start'] and total_steps % self.args['replay_freq'] == 0:
                    self.replay()
                    num_updates += 1
                    if num_updates % self.args['target_update_freq'] == 0:
                        self.target_Qnet.load_state_dict(self.Qnet.state_dict())

                total_steps += 1
        # print(self.stats)
    # def eval(self, n_episodes):
    #     for i_ep in range(n_episodes):
    #         state = np.zeros(self.state_size, dtype=np.uint8)
    #         screen = self.env.reset()
    #         state[-1] = screen.transpose(2, 0, 1)
    #         for t in count():
    #             self.env.render()
    #             action = self.act(state, 0)
    #             next_screen, reward, done, _ = self.env.step(action)
    #             if done:
    #                 break
    #             else:
    #                 state = np.vstack([state[1:], next_screen.transpose(2, 0, 1)])

    def load(self, modelpath):
        state_dict = torch.load(modelpath, map_location=lambda storage, loc: storage)
        self.Qnet.load_state_dict(state_dict)
        print("Loaded model from {}".format(modelpath))

    def save(self):
        modelpath = os.path.join(self.save_dir, 'QNetwork.pth.tar')
        torch.save(self.Qnet.state_dict(), modelpath)
        print("Saved to model to {}".format(modelpath))