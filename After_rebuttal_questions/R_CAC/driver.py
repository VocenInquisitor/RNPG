# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 13:35:03 2025

@author: Sourav
"""
import torch
import numpy
import gymnasium as gym
import Cartpole_changed as cart
import new_RCAC

######   BEST FOR CARTPOLE ENVIRONMENT
cost_coeff = 1
env = cart.CustomCartPoleEnv(cost_coeff)
lr_c_vf = 0.001
lr_c_cf = 0.001
lr_a = 0.0001
gamma = 0.99
bS = 500#500
max_episode_length = 500
delta = 0.1
rho = None
lambda_ = 30
b = 50
hidden_size1 = 128
hidden_size2 = 128
hidden_actor = 128
a_type = 1
ccf_type = 1
cvf_type = 1

#############  NETWORK epochs and number of episodes
epochs = 100
num_episodes = 100
#######################################  Storage parameters
env_nm ="Cartpole"
store_file =env_nm+"_vf_and_cf2.xlsx"

AC_obj = new_RCAC.train_Robust_CAC(env, lr_c_vf, lr_c_cf, lr_a, gamma, bS, max_episode_length, delta, rho, lambda_, b,store_file)
AC_obj.inititalize_networks(hidden_size1, hidden_size2, hidden_actor,a_type,cvf_type,ccf_type)

AC_obj.train(epochs, num_episodes)
torch.save(AC_obj.pi,'actor_uncertain.pth')

