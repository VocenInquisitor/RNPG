# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 09:07:32 2025

@author: Sourav
"""
import numpy as np
import pickle
import torch
from torch import nn as nn

class SRTD:
    def __init__(self,nS,nA,P,cost_list,init_state,gamma,delta):
        self.nS = nS
        self.nA = nA
        self.P = P
        self.cost_list = cost_list
        self.state = init_state
        self.alpha = 0.1
        self.gamma = gamma
        self.delta = delta
    def LSE(self,sigma,V):
        return np.log(np.sum(np.exp(sigma*V)))/sigma
    def find_SRTD(self,pol,ch,sigma,T):
        s = self.state
        Q = np.zeros((self.nS,self.nA))
        V = np.zeros(self.nS)
        cost = self.cost_list[ch]
        for i in range(T):
            a = np.random.choice(self.nA,p=pol[s])
            c = cost[s,a]
            next_s = np.random.choice(self.nS,p=self.P[a,s,:])
            V = np.array([np.sum([pol[s,a]*Q[s,a] for a in range(self.nA)]) for s in range(self.nS)])
            Q[s,a] = Q[s,a] + self.alpha*(c+self.gamma*(1-self.delta)*V[s]+self.gamma*self.delta*self.LSE(sigma,V)-Q[s,a])
        return Q

class Policy_approx(nn.Module):
    def __init__(self,nS,nA,if_complex):
        super().__init__()
        self.linear1 = nn.Linear(in_features=nS,out_features=10)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(10,50)
        self.linear3 = nn.Linear(50,nA)
        self.linear23_bypass = nn.Linear(nS,nA) 
        self.act =nn.Sigmoid()
        self.check= if_complex
    def forward(self,x):
        if self.check:
            x = self.act(self.linear3(self.relu(self.linear2(self.relu(self.linear1(x))))))
        else:
            x = self.act(self.relu(self.linear23_bypass(x)))
        return x
      

class ORPD:
    def __init__(self,T,sigma,eps,beta,alpha,b,cost_list,nS,nA,P,init_state,gamma,delta,init_dist,max_Lambda,env_nm):
        self.T = T
        self.sigma = sigma
        self.eps = eps
        self.beta = beta
        self.alpha = alpha
        self.b = b
        self.cost_list = cost_list
        self.lambda_ = 0.2
        self.nS = nS
        self.nA = nA
        self.init_pol_theta = Policy_approx(nS, nA,0)
        self.obj = SRTD(nS, nA, P, cost_list, init_state, gamma, delta)
        self.init_dist = init_dist
        self.max_Lambda = max_Lambda
        self.env_nm = env_nm
    def proj(self,pol):
        pass
    def get_vf_cf(self):
        vf_list=[]
        cf_list=[]
        pol_list=[]
        for t in range(self.T):
            pol = []
            for s in range(self.nS):
                action = self.init_pol_theta(self.one_hot(s))
                pol.append(action)
            pol = torch.Tensor(pol)
            T_inn = int((t+1)**1.5/(self.sigma**2))
            Q_r,Q_c = self.obj.find_SRTD(pol, 0, self.sigma, T_inn),self.obj.find_SRTD(pol,1,self.sigma,T_inn)
            V_r = torch.diag(torch.matmul(pol, Q_r))
            V_c = torch.diag(torch.matmul(pol, Q_c))
            J_r,J_c = torch.matmul(self.init_dist,V_r),torch.matmul(self.init_dist,V_c)
            self.lambda_ = np.max(0,np.min(self.lambda_ - 1/self.beta*(J_c.item()-self.b)-self.b/self.beta*self.lambda_,self.max_Lambda))
            J_r.backward()
            grad_Vr = pol.grad
            J_c.backward()
            grad_Vc = pol.grad
            with torch.no_grad():
                for param, g_Vr, g_Vc in zip(self.Policy_approx.parameters(), grad_Vr, grad_Vc):
                    param += (1 /self.alpha) * (g_Vr + self.lambda_ * g_Vc)
            #pol = self.proj(pol+1/self.alpha*(grad_Vr+self.lambda_*grad_Vc))
            vf_list.append(J_r.item())
            cf_list.append(J_c.item())
            pol_list.append(pol)
        with open("ORPD_"+self.env_nm+"_vf") as f:
            pickle.dump(vf_list)
        f.close();
        with open("ORPD_"+self.env_nm+"_cf") as f:
            pickle.dump(cf_list)
        f.close();
        with open("ORPD_"+self.env_nm+"_policies") as f:
            pickle.dump(pol_list)
        f.close();
        
        
        
            
        