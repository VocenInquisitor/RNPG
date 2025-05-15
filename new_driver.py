# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 17:30:47 2025

@author: Sourav
"""
from Machine_Rep import Machine_Replacement
from new_RPNG import RNPG
import numpy as np
from KL_uncertainity_evaluator import Robust_pol_Kl_uncertainity

# Setup environment
env_model = Machine_Replacement()
P = env_model.gen_probability()


# Define initial theta
nS, nA = env_model.nS, env_model.nA
initial_theta = np.random.randn(nS, nA)

r,c = env_model.gen_expected_reward(),env_model.gen_expected_cost()
cost_list = [r,c]
init_dist = np.exp(np.random.randn(nS))
init_dist = init_dist/np.sum(init_dist)
oracle = Robust_pol_Kl_uncertainity(nS, nA, cost_list, init_dist)

# Instantiate RNPG with dummy oracle
rnpg = RNPG(env_model, oracle,P,r,c) #env, r_oracle_obj,P,cost_list, alpha, lambda_, theta

# Train for T iterations
final_policy, final_theta = rnpg.train_all(T=10)

print("Final Policy:")
print(final_policy)
print("Final Theta:")
print(final_theta)
