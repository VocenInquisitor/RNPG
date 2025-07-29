# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:39:42 2025

@author: Sourav
"""
import numpy as np
class Frozenlake:
    def __init__(self,n,nA):
        self.n = n
        self.nA = nA
        self.nS = int(n*n)
    def gen_grid(self):
        grid = np.zeros((self.n,self.n))
        grid[0,0] = -1
        grid[3,3] = 1
    def unfold_state_space(self,r,c):
        return r*self.n + c;
    def gen_prob_space(self):
        P = np.zeros((self.nA,self.nS,self.nS))
        for s in range(self.nS):
            if(s%self.n!=0):
                P[0,s,s-1] = 1.
            if(s<((self.n-1)*self.n)):
                P[0,s,s+self.n] = 1.
            if(s>=self.n):
                P[0,s,s-self.n] = 1.
                
            if(s%self.n!=self.n-1):
                P[1,s,s+1] = 1.
            if(s<((self.n-1)*self.n)):
                P[1,s,s+self.n] = 1.
            if(s>=self.n):
                P[1,s,s-self.n] = 1.
            P[0,s] = P[0,s]/np.sum(P[0,s])
            P[1,s] = P[1,s]/np.sum(P[1,s])
        return P;

env = Frozenlake(4, 2)
P = env.gen_prob_space()
print(P[0,2])
        
