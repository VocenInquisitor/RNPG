# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 14:40:24 2025

@author: gangu
"""
import numpy as np
import torch
from torch import nn,optim
from torch.distributions import Categorical
import gymnasium as gym
import pandas as pd
# Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)

def compute_returns(values, gamma=0.99):
    returns = []
    G = 0
    for v in reversed(values):
        G = v + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns)

def collect_trajectory(env, actor, max_steps=500):
    state, _ = env.reset()
    states, actions, rewards, costs, log_probs = [], [], [], [], []

    for _ in range(max_steps):
        state_tensor = torch.FloatTensor(state)
        probs = actor(state_tensor)
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_state, reward, done, truncated, _ = env.step(action.item())

        cost = abs(state[0])  # custom cost: cart's distance from center

        states.append(state_tensor)
        actions.append(action)
        rewards.append(reward)
        costs.append(cost)
        log_probs.append(log_prob)

        if done or truncated:
            break
        state = next_state

    return states, actions, rewards, costs, log_probs


class EPIRC_PGS_CS:
    def __init__(self,K,alpha,env,T,b_1):
        self.cost_objects = [Critic(),Critic()]
        self.dict_vf = {'vf':[],'cf':[]}
        self.actor = Actor(128,2)
        self.reward_critic = Critic(128)
        self.constraint_critic = Critic(128)
        self.ppg_ob = PPG(alpha,T,env,b_1,self.cost_objects,self.dict_vf,self.reward_critic,self.constraint_critic,self.actor)
        self.K = K
        self.i = 0
        self.j = 500
        self.b_1 = b_1
        self.env = env
    def run_algo(self):
        for k in range(self.K):
            b = (self.i+self.j)//2
            V_o,V_c = self.ppg_ob.get_policy(b)
            #V_o,V_c = self.cost_objects[0].get_vf(self.pi),self.cost_objects[1].get_vf(self.pi)
            del_k = np.max([V_o - b,V_c-self.b_1])
            if del_k>0:
                self.i = b
            else:
                self.j = b
        dF = pd.dataFrame(self.dict_vf)
        dF.to_excel('EPIRC_PGS_CS_value_functions.xlsx')
        torch.save(self.actor,'EPIRC_actor.pth')
        return self.ppg_ob.get_policy(self.j)
            

class PPG:
    def __init__(self,alpha,T,env,b_1,critic_objs,dict_,rc,cc,ac):
        self.alpha = alpha
        self.T = T
        self.critic_objs = critic_objs
        self.dict_vf = dict_
        self.b_1 = b_1
        self.env = env
        self.reward_critic = rc
        self.constraint_critic = cc
        self.actor = ac
        self.optim_r = optim.Adam(self.reward_critic.parameters(),lr=1e-3)
        self.optim_c = optim.Adam(self.constraint_critic.parameters(),lr=1e-3)
    def get_policy(self,b):
        self.b = b
        V_0,V_c =0,0
        for t in range(self.T):
            states, actions, rewards, costs, log_probs = collect_trajectory(self.env, self.actor)
    
            states_tensor = torch.stack(states)
            log_probs_tensor = torch.stack(log_probs)
            reward_returns = compute_returns(rewards)
            cost_returns = compute_returns(costs)
            V_o = reward_returns[-1].item()
            V_c = cost_returns[-1].item()
            self.dict_vf['vf'].append(V_o)
            self.dict_vf['cf'].append(V_c)
    
            # === Train Critics ===
            self.optim_r.zero_grad()
            loss_r = nn.functional.mse_loss(self.reward_critic(states_tensor), reward_returns)
            loss_r.backward()
            self.optim_r.step()
    
            self.optim_c.zero_grad()
            loss_c = nn.functional.mse_loss(self.constraint_critic(states_tensor), cost_returns)
            loss_c.backward()
            self.optim_c.step()
    
            # === Evaluate Constraint ===
            total_cost = sum(costs)
            violation = total_cost - self.b_1
    
            if violation > 0:
                critic = self.constraint_critic
                sign = -1
            else:
                critic = self.reward_critic
                sign = +1
    
            # === Policy Gradient Step ===
            advantage = compute_returns(critic(states_tensor).detach().tolist())
            loss_pi = -sign * (log_probs_tensor * advantage).mean()
    
            self.actor.zero_grad()
            loss_pi.backward()
            for p in self.actor.parameters():
                p.data -= self.alpha * p.grad  # projected step
        return V_0,V_c
            
if __name__=='__main__':
    env = gym.make('Cartpole-v1')
    b_1 = 200
    K=20
    T=500
    alpha = 1e-4
    algo_obj = EPIRC_PGS_CS(K,alpha,env,T,b_1)
    algo_obj.run_algo()
    
        