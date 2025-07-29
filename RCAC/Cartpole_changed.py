# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 12:14:38 2025

@author: Sourav
"""

'''import safety_gym
import gym

env = gym.make('Safexp-PointGoal1-v0')
'''
import gymnasium as gym
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
import numpy as np

class CustomCartPoleEnv(CartPoleEnv):
    def __init__(self, cost_coefficient=0.1):
        super().__init__()
        self.cost_coefficient = cost_coefficient  # Weight of distance-based cost

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)

        x = observation[0]  # Cart position

        # Additional cost based on distance from center
        distance_cost = self.cost_coefficient * abs(x)

        return observation, reward, distance_cost, terminated, truncated, info