# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 12:40:25 2025

@author: gangu
"""

import torch
import numpy as np
from torch import nn
#from torch.utils.data import DataLoader
#from torchvision import datasets, transforms
from Replay_Buffer import Replay_Buffer
import copy

#Initialization type denoted by--->  type_ = {0 - Random,1 - Xavier_uniform,2 - Xavier_normal,3 - Kaiming uniform,4 - Kaiming normal}

class Actor(nn.Module):
    def __init__(self,nS,nA,hid_sz,type_=1):
        super(Actor,self).__init__()
        self.nS = nS
        self.nA = nA
        self.fc1 = nn.Linear(nS,hid_sz)
        self.fc2 = nn.Linear(hid_sz,hid_sz)
        self.fc3 = nn.Linear(hid_sz,nA)
        #self.critic = nn.Linear(hid_sz,1)
        if type_==1:
            nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.fc3.weight, gain=0.01)
        elif type_==2:
            nn.init.xavier_normal_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_normal_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_normal_(self.fc3.weight, gain=0.01)
        elif type_==3:
            nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_uniform_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        elif type_==4:
            nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.fc3.weight, mode='fan_out', nonlinearity='relu')
            
    
    def forward(self,s):
        x = torch.relu(self.fc1(s))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

class Critic_vf(nn.Module):
    def __init__(self,nS,hid_sz,type_=1):
        super(Critic_vf,self).__init__()
        self.fc1 = nn.Linear(nS,hid_sz)
        self.fc2 = nn.Linear(hid_sz,hid_sz)
        self.critic_vf = nn.Linear(hid_sz,1)
        if type_==1:
            nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.critic_vf.weight, gain=0.01)
        elif type_==2:
            nn.init.xavier_normal_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_normal_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_normal_(self.critic_vf.weight, gain=0.01)
        elif type_==3:
            nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_uniform_(self.critic_vf.weight, mode='fan_in', nonlinearity='relu')
        elif type_==4:
            nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.critic_vf.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self,s):
        x = torch.relu(self.fc1(s))
        x = torch.relu(self.fc2(x))
        return self.critic_vf(x)

class Critic_cf(nn.Module):
    def __init__(self,nS,hid_sz,type_=1):
        super(Critic_cf,self).__init__()
        self.fc1 = nn.Linear(nS,hid_sz)
        self.fc2 = nn.Linear(hid_sz,hid_sz)
        self.critic_cf = nn.Linear(hid_sz,1)
        if type_==1:
            nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.critic_cf.weight, gain=0.01)
        elif type_==2:
            nn.init.xavier_normal_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_normal_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_normal_(self.critic_cf.weight, gain=0.01)
        elif type_==3:
            nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_uniform_(self.critic_cf.weight, mode='fan_in', nonlinearity='relu')
        elif type_==4:
            nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.critic_cf.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self,s):
        x = torch.relu(self.fc1(s))
        x = torch.relu(self.fc2(x))
        return self.critic_cf(x)

class train_Robust_CAC:
    def __init__(self,env,lr_c_vf,lr_c_cf,lr_a,gamma,bS,max_episode_length,delta,rho,lambda_,b,seed=42):
        self.env = env
        self.lr_critic_vf = lr_c_vf
        self.lr_critic_cf = lr_c_cf
        self.lr_actor = lr_a
        self.gamma = gamma
        self.nS = self.env.observation_space.shape[0]
        self.nA = self.env.action_space.n
        self.seed = seed
        self.rb = Replay_Buffer(bS, self.nS, self.nA)
        self.batch_sz = bS
        self.amount_perturbed = delta
        self.rho = rho
        self.lagrangian = lambda_
        self.b = b
        #self.max_episode_steps = max_episode_length
    def inititalize_networks(self,hidden_size1,hidden_size2,hidden_actor,a_type=0,cvf_type=0,ccf_type=0):
        self.V = Critic_vf(self.nS,hidden_size1,cvf_type)
        self.C = Critic_cf(self.nS,hidden_size2,ccf_type)
        self.pi = Actor(self.nS,self.nA,hidden_actor,a_type)
        self.V_loss = nn.MSELoss()
        self.C_loss = nn.MSELoss()
        self.V_opt = torch.optim.Adam(self.V.parameters(),lr=self.lr_critic_vf)
        self.C_opt = torch.optim.Adam(self.C.parameters(),lr=self.lr_critic_cf)
        self.pi_opt = torch.optim.Adam(self.pi.parameters(),lr=self.lr_actor)
    def collect_samples(self):
        s = self.env.reset()
        s = torch.tensor(s[0],dtype=torch.float)
        for i in range(self.batch_sz):
            with torch.no_grad():
                #print(s[0])
                #print("After", s)
                #input()
                action_prob = torch.distributions.Categorical(logits = self.pi(s))
                a = action_prob.sample()
                #print("After action:",a)
                #input()
                a_logprob = action_prob.log_prob(a)
                #print("After log_prob:",a_logprob)
                #input()
                s_,r,c,done,trunc,info = self.env.step(a.item()) #observation, reward, distance_cost, terminated, truncated, info
                #print(s_)
                #c = #Decide how to get cost
                if done and i!= self.batch_sz-1:
                    dw = True
                    s = self.env.reset()
                    s = torch.tensor(s[0],dtype=torch.float)
                else:
                    dw = False
                self.rb.store(s, a, a_logprob, r,c, s_, dw, done)#s, a, a_logprob, r,c, s_, dw, done
                s = torch.tensor(copy.deepcopy(s_))
    def train(self, epochs, num_episodes):
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(epochs):
            for j in range(int(num_episodes / self.batch_sz)):
    
                # Collect transitions
                self.collect_samples()
                s, a, a_logprob, r, c, s_, dw, done = self.rb.numpy_to_tensor()
    
                # Forward pass for value and cost critics
                V_s = self.V(s)
                V_s_s_ = self.V(s_)
    
                C_s = self.C(s)
                C_s_s_ = self.C(s_)
    
                # Compute targets
                V_target = r + self.gamma * (1 - dw) * V_s_s_
                C_target = c + self.gamma * (1 - dw) * C_s_s_
                
                # Compute advantages (delta)
                adv_vf = V_target.detach() - V_s
                adv_cf = C_target.detach() - C_s
    
                # Critic losses
                loss_vf = self.V_loss(V_target, V_s)
                loss_cf = self.C_loss(C_target, C_s)
                
                # Constraint term: ch = max(V_s / λ, C_s - b)
                ch = torch.max(V_s.detach() / self.lagrangian, C_s.detach() - self.b)
                
    
                # Backprop for critics
                self.V_opt.zero_grad()
                loss_vf.backward(retain_graph=True)
                self.V_opt.step()
    
                self.C_opt.zero_grad()
                loss_cf.backward(retain_graph=True)
                self.C_opt.step()
    
                
    
                # Policy loss: mixture of reward and cost depending on constraint
                actor_loss_vf = -a_logprob * adv_vf
                actor_loss_cf = -a_logprob * adv_cf
                actor_loss = actor_loss_vf * (1 - ch) + actor_loss_cf * ch
                actor_loss = actor_loss.mean()
    
                # Backprop for policy
                self.pi_opt.zero_grad()
                actor_loss.backward()
                self.pi_opt.step()
    
            print(f"Epoch {epoch + 1}/{epochs} completed.")

                
                
                
                
                
        
            
    