import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Uniform, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import os
import pickle
import time
import numpy as np
from copy import deepcopy


class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 20
    buffer_capacity = 20
    batch_size = 10

    def __init__(self, actor, critic, encoder, node_type=None, edge_index=None):
        self.actor_net = actor
        self.critic_net = critic
        self.actor_encoder = deepcopy(encoder)
        self.critic_encoder = deepcopy(encoder)
        self.node_type = node_type
        self.edge_index = edge_index
        self.actor_net, self.critic_net, self.actor_encoder, self.critic_encoder = self.actor_net.cuda(), self.critic_net.cuda(), self.actor_encoder.cuda(), self.critic_encoder.cuda()
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.step_param = 1e-3
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gamma = 0.99
        self.dist = []

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 5e-3)
        self.actor_encoder_optimizer = optim.Adam(self.actor_encoder.parameters(), 1e-3)
        self.critic_encoder_optimizer = optim.Adam(self.critic_encoder.parameters(), 1e-3)


    def select_action(self, state):

        state = torch.from_numpy(state).float().unsqueeze(0).cuda()
        with torch.no_grad():
            hidden_state = self.actor_encoder(state, self.node_type, self.edge_index)
            mu, sigma = self.actor_net(hidden_state)

        dist = Normal(mu, sigma)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        action_tuple = (mu[0].item(), sigma[0].item(), action.item())
        self.dist.append(action_tuple)
        return action.item(), action_log_prob.item()

    def get_value(self, state):
        state = torch.from_numpy(state).cuda()
        with torch.no_grad():
            hidden_state = self.critic_encoder(state, self.node_type, self.edge_index)
            value = self.critic_net(hidden_state)
        return value.item()

    def save_param(self, path, episode):
        torch.save(self.actor_net.state_dict(), os.path.join(path, 'actor_net'+str(episode)+'.pkl'))
        torch.save(self.critic_net.state_dict(), os.path.join(path, 'critic_net'+str(episode)+'.pkl'))
        torch.save(self.actor_encoder.state_dict(), os.path.join(path, 'actor_encoder' + str(episode) + '.pkl'))
        torch.save(self.critic_encoder.state_dict(), os.path.join(path, 'critc_encoder' + str(episode) + '.pkl'))
        pickle.dump(self.dist, open(os.path.join(path, 'action' + str(episode) + '.pkl'), 'wb'))
        del self.dist[:]
        del self.buffer[:]

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    def update(self):
        self.training_step += 1
        lmbda = 0.95

        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float).to(self.device)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.float).view(-1, 1).to(self.device)
        next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float).to(self.device)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1).to(self.device)
        reward = [t.reward for t in self.buffer]
        is_done = [t.done for t in self.buffer]
        target_v = []
        x = []
        for i in range(len(reward)):
            if reward[i] is None:
                x.append(i)
            else:
                x.append(i)
                avg_r = reward[i] / len(x)
                for j in x:
                    reward[j] = avg_r - self.step_param*i
                x = []
        discounted_reward = 0

        gae = 0
        with torch.no_grad():
            hidden_state = self.critic_encoder(state, self.node_type, self.edge_index)
            state_v = self.critic_net(hidden_state).detach()
            hidden_next_state = self.critic_encoder(next_state, self.node_type, self.edge_index)
            next_state_v = self.critic_net(hidden_next_state).detach()
        for i in reversed(range(len(reward))):
            print(reward[i])
            delta = reward[i] + self.gamma * next_state_v[i] * (1 - is_done[i]) - state_v[i]
            gae = delta + self.gamma * lmbda * (1 - is_done[i]) * gae
            target_v.insert(0, gae + state_v[i])
        target_v = torch.tensor(target_v, dtype=torch.float32).to(self.device)
        advantage = target_v - state_v
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)

        for _ in range(self.ppo_epoch):  # iteration ppo_epoch
            hidden_state = self.actor_encoder(state, self.node_type, self.edge_index)
            mu, sigma = self.actor_net(hidden_state)
            n = Normal(mu, sigma)
            action_log_prob = n.log_prob(action)
            dist_entropy = n.entropy().mean()
            ratio = torch.exp(action_log_prob - old_action_log_prob)

            L1 = ratio * advantage
            L2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
            action_loss = -(torch.min(L1, L2)).mean() - dist_entropy * 0.01  # MAX->MIN desent
            value_loss = 0.5 * (self.critic_net(hidden_state) - target_v).pow(2).mean()
            total_loss = action_loss + value_loss
            self.actor_optimizer.zero_grad()
            self.critic_net_optimizer.zero_grad()
            self.actor_encoder_optimizer.zero_grad()
            self.critic_encoder_optimizer.zero_grad()
            total_loss.backward()
            self.actor_optimizer.step()
            self.critic_net_optimizer.step()
            self.actor_encoder_optimizer.step()
            self.critic_encoder_optimizer.step()
