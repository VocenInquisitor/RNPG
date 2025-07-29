# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 02:46:44 2025

@author: gangu
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np

# ----------------------------------------------
# Helper Functions and Base Networks
# ----------------------------------------------

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class Actor_Beta(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.alpha_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.beta_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]
        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.alpha_layer, gain=0.01)
            orthogonal_init(self.beta_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        alpha = F.softplus(self.alpha_layer(s)) + 1.0
        beta = F.softplus(self.beta_layer(s)) + 1.0
        return alpha, beta

    def get_dist(self, s):
        from torch.distributions import Beta
        alpha, beta = self.forward(s)
        return Beta(alpha, beta)

    def mean(self, s):
        alpha, beta = self.forward(s)
        return alpha / (alpha + beta)

class Actor_Gaussian(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]
        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mean = self.max_action * torch.tanh(self.mean_layer(s))
        return mean

    def get_dist(self, s):
        from torch.distributions import Normal
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        return Normal(mean, std)

class Critic(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]
        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        return self.fc3(s)

# ----------------------------------------------
# PPO Continuous with Cost Constraint
# ----------------------------------------------

class PPO_continuous():
    def __init__(self, args):
        self.policy_dist = args.policy_dist
        self.max_action = args.max_action
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        # Constraint-specific
        self.lam_constraint = args.lam_constraint     # λ (lambda) for constraint penalty
        self.cost_limit = args.cost_limit             # b (budget/cost bound)

        if self.policy_dist == "Beta":
            self.actor = Actor_Beta(args)
        else:
            self.actor = Actor_Gaussian(args)
        self.critic_objective = Critic(args)
        self.critic_cost = Critic(args)

        if self.set_adam_eps:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic_objective = torch.optim.Adam(self.critic_objective.parameters(), lr=self.lr_c, eps=1e-5)
            self.optimizer_critic_cost = torch.optim.Adam(self.critic_cost.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic_objective = torch.optim.Adam(self.critic_objective.parameters(), lr=self.lr_c)
            self.optimizer_critic_cost = torch.optim.Adam(self.critic_cost.parameters(), lr=self.lr_c)

    def evaluate(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        with torch.no_grad():
            if self.policy_dist == "Beta":
                a = self.actor.mean(s).detach().numpy().flatten()
            else:
                a = self.actor(s).detach().numpy().flatten()
        return a

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        with torch.no_grad():
            dist = self.actor.get_dist(s)
            a = dist.sample()
            if self.policy_dist == "Beta":
                a_logprob = dist.log_prob(a)
            else:
                a = torch.clamp(a, -self.max_action, self.max_action)
                a_logprob = dist.log_prob(a)
        return a.numpy().flatten(), a_logprob.numpy().flatten()

    def update(self, replay_buffer, total_steps):
        # Unpack: make sure your replay_buffer provides costs!
        s, a, a_logprob, r, cost, s_, dw, done = replay_buffer.numpy_to_tensor()

        adv_obj, adv_cost = [], []
        gae_obj, gae_cost = 0, 0

        with torch.no_grad():
            v_obj = self.critic_objective(s)
            v_obj_ = self.critic_objective(s_)
            v_cost = self.critic_cost(s)
            v_cost_ = self.critic_cost(s_)

            delta_obj = r + self.gamma * (1.0 - dw) * v_obj_ - v_obj
            delta_cost = cost + self.gamma * (1.0 - dw) * v_cost_ - v_cost

            for dlt_o, dlt_c, d in zip(reversed(delta_obj.flatten().numpy()),
                                      reversed(delta_cost.flatten().numpy()),
                                      reversed(done.flatten().numpy())):
                gae_obj = dlt_o + self.gamma * self.lamda * gae_obj * (1.0 - d)
                gae_cost = dlt_c + self.gamma * self.lamda * gae_cost * (1.0 - d)
                adv_obj.insert(0, gae_obj)
                adv_cost.insert(0, gae_cost)

            adv_obj = torch.tensor(adv_obj, dtype=torch.float).view(-1, 1)
            adv_cost = torch.tensor(adv_cost, dtype=torch.float).view(-1, 1)
            v_target_obj = adv_obj + v_obj
            v_target_cost = adv_cost + v_cost

            if self.use_adv_norm:
                adv_obj = (adv_obj - adv_obj.mean()) / (adv_obj.std() + 1e-5)
                adv_cost = (adv_cost - adv_cost.mean()) / (adv_cost.std() + 1e-5)

        for _ in range(self.K_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now = self.actor.get_dist(s[index])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)
                a_logprob_now = dist_now.log_prob(a[index])
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True))

                # Compute channel (ch): 0=objective, 1=constraint
                v_o = self.critic_objective(s[index]).detach()
                v_c = self.critic_cost(s[index]).detach()
                v_concat = torch.cat([
                    v_o,
                    self.lam_constraint * (v_c - self.cost_limit)
                ], dim=1)
                ch = torch.argmax(v_concat, dim=1, keepdim=True)

                # Select which advantage signal to use for each sample
                adv_signal = torch.where(
                    ch == 0,                 # If argmax selects objective
                    adv_obj[index],
                    self.lam_constraint * adv_cost[index]
                )

                surr1 = ratios * adv_signal
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv_signal

                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy

                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                # Update critics
                v_s_obj = self.critic_objective(s[index])
                critic_obj_loss = F.mse_loss(v_target_obj[index], v_s_obj)
                self.optimizer_critic_objective.zero_grad()
                critic_obj_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.critic_objective.parameters(), 0.5)
                self.optimizer_critic_objective.step()

                v_s_cost = self.critic_cost(s[index])
                critic_cost_loss = F.mse_loss(v_target_cost[index], v_s_cost)
                self.optimizer_critic_cost.zero_grad()
                critic_cost_loss.backward()
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.critic_cost.parameters(), 0.5)
                self.optimizer_critic_cost.step()

        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic_objective.param_groups:
            p['lr'] = lr_c_now
        for p in self.optimizer_critic_cost.param_groups:
            p['lr'] = lr_c_now