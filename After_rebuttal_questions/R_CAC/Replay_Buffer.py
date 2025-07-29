# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 15:10:44 2025

@author: Sourav Ganguly
"""

import torch
import numpy as np

class Replay_Buffer:
    def __init__(self,bS,nS,nA):
        self.s = np.zeros((bS, nS))
        self.a = np.zeros((bS, nA))
        self.a_logprob = np.zeros((bS, nA))
        self.r = np.zeros((bS, 1))
        self.c = np.zeros((bS, 1))
        self.s_ = np.zeros((bS, nS))
        self.dw = np.zeros((bS, 1))
        self.done = np.zeros((bS, 1))
        self.count = 0
        self.dict={'vf':[],'cf':[]}
        self.tr = 0
        self.tc = 0
    
    def store(self, s, a, a_logprob, r,c, s_, dw, done):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.c[self.count] = c
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1
        self.tr+=r
        self.tc+=c
        if(dw==True):
            self.dict['vf'].append(self.tr)
            self.dict['cf'].append(self.tc)
            self.tr=0
            self.tc=0
    def numpy_to_tensor(self):
        s = torch.tensor(self.s, dtype=torch.float)
        a = torch.tensor(self.a, dtype=torch.float)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        c = torch.tensor(self.c, dtype=torch.float)
        s_ = torch.tensor(self.s_, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)

        return s, a, a_logprob, r,c, s_, dw, done