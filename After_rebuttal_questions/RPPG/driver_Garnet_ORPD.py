# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:27:22 2025

@author: Sourav
"""
#import pickle
import numpy as np
#from matplotlib import pyplot as plt
from Garnet import Garnet
from RC_PD_approach_forward_diff import SRTD,ORPD

nS,nA = 6,2
env = Garnet(nS, nA)
P,R,C = env.gen_nominal_prob(),env.gen_expected_reward(),env.gen_expected_constraint()
#print(R)
T = 100
sigma = -0.001
eps = 0.1
beta = 0.001
alpha = 0.001
b = 110
cost_list = [R,C]
init_state = 0
gamma = 0.99
delta = 0.3
np.random.seed(300)
init_dist = np.random.normal(loc = 1,scale=2,size=nS)
init_dist = np.exp(init_dist)
init_dist = init_dist/np.sum(init_dist)
approx = 0.5
max_Lambda = 2/(approx*(1-gamma))
env_nm ="Garnet_6_2"

model = ORPD(T, sigma, eps, beta, alpha, b, cost_list, nS, nA, P, init_state, gamma, delta, init_dist, max_Lambda, env_nm)

model.get_vf_cf()

print("All done and files stored")
