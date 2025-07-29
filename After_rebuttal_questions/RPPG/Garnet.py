# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 10:52:19 2025

@author: Sourav
"""
import numpy as np
class Garnet:
    def __init__(self,nS,nA):
        self.nA = nA
        self.nS = nS
    def gen_nominal_prob(self):
        self.P = np.zeros((self.nA,self.nS,self.nS))
        for s in range(self.nS):
            for a in range(self.nA):
                mu,sigma = np.random.uniform(0,100),np.random.uniform(0,100)
                self.P[a,s,:] = np.random.normal(mu,sigma,self.nS)
                self.P[a,s] = np.exp(self.P[a,s])
                self.P[a,s] = self.P[a,s]/np.sum(self.P[a,s])
        return self.P
       
    def gen_expected_reward(self):
        self.R = np.zeros((self.nS,self.nA))
        for s in range(self.nS):
            for a in range(self.nA):
                mu,sigma = np.random.uniform(0,10),np.random.uniform(0,10)
                self.R[s,a] = np.random.normal(mu,sigma)/10
        return self.R
    def gen_expected_constraint(self):
        self.R = np.zeros((self.nS,self.nA))
        for s in range(self.nS):
            for a in range(self.nA):
                mu,sigma = np.random.uniform(0,10),np.random.uniform(0,10)
                self.R[s,a] = np.random.normal(mu,sigma)/10
        return self.R
'''nS,nA = 3,2
env = Garnet(nS,nA)
print(env.gen_nominal_prob())
print("=======")
print(env.gen_expected_reward())'''