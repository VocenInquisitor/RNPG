# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 13:35:17 2025

@author: Sourav
"""
import numpy as np
from KL_uncertainity_evaluator import Robust_pol_Kl_uncertainity
import time
from Machine_Rep import *
from RNPG import *
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

'''class policy_gen(nn.Module):
    def __init__(self,states,action):'''

class policy_gen:
    def __init__(self,nS,nA):
        self.parameters = np.random.normal(0,1,(nS,nA))
    def _get_param_(self):
        return self.parameters
    def _set_param_(self,theta):
        self.paramaters = theta
    def forward(self,state):
        val = np.matmul(np.transpose(self.parameters),state)
        val = np.exp(val)
        return val/np.sum(val)
        

env = River_swim()
r,c = env.gen_expected_reward(),env.gen_expected_cost()
P = env.gen_probability()
cost_list = [r,c]
init_dist = np.exp(np.random.randn(env.nS))
init_dist = init_dist/np.sum(init_dist)
alpha = 0.0001
oracle = Robust_pol_Kl_uncertainity(env.nS, env.nA, cost_list, init_dist)
lambda_ = 50
T = 10
b = 40
pol_obj = policy_gen(env.nS,env.nA)
rpng_obj = RNPG(env,oracle,alpha,lambda_,pol_obj,P,cost_list,b)
rpng_obj.train_all(T)

