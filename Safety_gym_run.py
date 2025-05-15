# -*- coding: utf-8 -*-
"""
Created on Mon May 12 12:59:33 2025

@author: Sourav
"""

import gymnasium as gym
import safety_gym
import numpy as np
from collections import defaultdict

from KL_uncertainity_evaluator import Robust_pol_Kl_uncertainity
from RNPG import RNPG

# -----------------------------
# Tabular Policy for Discrete States
# -----------------------------
class TabularPolicy:
    def __init__(self, n_states, n_actions):
        self.nS = n_states
        self.nA = n_actions
        self.theta = np.ones((n_states, n_actions)) / n_actions

    def forward(self, state_vec):
        s = np.argmax(state_vec)
        return self.theta[s]

    def _get_param_(self):
        return self.theta.copy()

    def _set_param_(self, new_theta):
        self.theta = new_theta.copy()

# -----------------------------
# Discretization Helper
# -----------------------------
def discretize_state(obs, bins):
    idx = tuple(np.digitize(obs[i], bins[i]) for i in range(len(obs)))
    return idx

# -----------------------------
# Build Transition Matrix
# -----------------------------
def estimate_transition_model(env, policy, bins, nS, nA, episodes=500):
    P = np.zeros((nA, nS, nS))
    state_to_idx = {}
    idx_to_state = []
    count = defaultdict(lambda: np.zeros(nS))

    for ep in range(episodes):
        obs = env.reset()
        s_disc = discretize_state(obs['observation'], bins)
        if s_disc not in state_to_idx:
            state_to_idx[s_disc] = len(state_to_idx)
            idx_to_state.append(s_disc)
        s = state_to_idx[s_disc]

        done = False
        while not done:
            a = np.random.choice(nA)
            next_obs, _, done, _, _ = env.step(a)
            s2_disc = discretize_state(next_obs['observation'], bins)
            if s2_disc not in state_to_idx:
                state_to_idx[s2_disc] = len(state_to_idx)
                idx_to_state.append(s2_disc)
            s2 = state_to_idx[s2_disc]
            P[a, s, s2] += 1
            s = s2

    for a in range(nA):
        for s in range(nS):
            total = np.sum(P[a, s])
            if total > 0:
                P[a, s] /= total
            else:
                P[a, s] = np.ones(nS) / nS

    return P, state_to_idx, idx_to_state

# -----------------------------
# Main Script
# -----------------------------
env = gym.make('SafetyPointGoal0-v0')
n_bins = 4
obs_dim = env.observation_space['observation'].shape[0]
actions = [0, 1, 2, 3]  # Discrete actions
nA = len(actions)

# Define bins for discretization
bins = [np.linspace(-1, 1, n_bins) for _ in range(obs_dim)]

# Assume state size = 20 for now (adjust based on exploration)
nS = 20
init_dist = np.ones(nS) / nS
cost_list = [np.random.rand(nS, nA), np.random.rand(nS, nA)]  # Placeholder costs

pol_obj = TabularPolicy(nS, nA)
r_oracle_obj = Robust_pol_Kl_uncertainity(nS, nA, cost_list, init_dist)

# Estimate model
P, state_map, _ = estimate_transition_model(env, pol_obj, bins, nS, nA)

# RNPG Training
rnpg = RNPG(env=env,
            r_oracle_obj=r_oracle_obj,
            alpha=1.0,
            lambda_=0.5,
            pol_obj=pol_obj,
            P=P,
            cost_list=cost_list,
            b=0.1)

rnpg.train_all(T=10)  # Run for 10 iterations