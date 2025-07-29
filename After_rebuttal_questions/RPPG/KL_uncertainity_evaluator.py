# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 14:00:27 2025

@author: Sourav
"""

import numpy as np
from Machine_Rep import Machine_Replacement

class Robust_pol_Kl_uncertainity:
    def __init__(self,nS,nA,cost_list,init_dist,alpha=1):
        self.nS = nS
        self.nA = nA
        self.cost_list = cost_list
        self.init_dist = init_dist
        self.gamma = 0.995
        self.alpha = alpha
    def calculate_infinite_Q(self,n,policy,P,C_KL):
        C = self.cost_list[n]
        Q = np.zeros((self.nS,self.nA))
        V = np.zeros(self.nS)
        tau = 1000
        s = np.random.choice(self.nS,p=self.init_dist)
        for t in range(tau):
            #print(policy[s])
            a = np.random.choice(self.nA,p = policy[s])
            next_state = np.random.choice(self.nS,p=P[a,s,:])
            #print("t=",self.t,a,s)
            P_star = np.array([P[a,s,i]*np.exp(self.alpha*V[i]/C_KL) for i in range(self.nS)])
            #print(P_star)
            #print("P_satr:",P_star)
            Q[s,a] = C[s,a] + self.gamma * np.dot(P_star,V)
            V = np.array([np.dot(policy[s],Q[s,:]) for s in range(self.nS)])
            #print("V=",V)
            s = next_state
        return Q,V
    def evaluate_policy(self,policy,P,C_KL,n,t):
        self.t = t
        policy = np.array(policy)
        Q,V = self.calculate_infinite_Q(n, policy, P, C_KL)
        P_star = np.zeros((self.nS,self.nA,self.nS))
        #Pi_pi = torch.zeros((self.nS,self.nS,self.nA))
        Q_ = np.zeros((self.nS,self.nA))
        T = np.zeros((self.nS,self.nS))
        for s in range(self.nS):
            for a in range(self.nA):
                P_star[s,a,:] = np.array([self.alpha*P_star[s,a,i]*np.exp(V[i]/C_KL) for i in range(self.nS)])
                #Q_[s,a] = self.cost_list[n][s,a] + self.gamma*torch.sum([P_star[s,a,s_next]*torch.sum([policy[s_next,a_next]*Q_[s_next,a_next] for a_next in range(self.nA)]) for s_next in range(self.nS)])#not correct
        for s in range(self.nS):
            for s_next in range(self.nS):
                T[s,s_next] = np.sum(np.array([policy[s,a]*P[a,s,s_next] for a in range(self.nA)]))
        I = np.eye(self.nS)
        Q_ = np.dot(np.linalg.inv(I-self.gamma*T),self.cost_list[n])
        d_pi = np.matmul(np.linalg.inv(I-self.gamma*T),self.init_dist)
        d_pi = d_pi/np.sum(d_pi)
        #print("d_pi:",d_pi)
        J = np.sum([self.init_dist[s]*np.sum([policy[s,a]*Q_[s,a] for a in range(self.nA)]) for s in range(self.nS)])
        J_grad = np.array([d_pi[s]*np.array([Q_[s,a] for a in range(self.nA)]) for s in range(self.nS)])
        return J,J_grad


'''mr_obj = Machine_Replacement()
nS,nA = mr_obj.nS,mr_obj.nA
cost_list = [mr_obj.gen_expected_reward(2),mr_obj.gen_expected_cost()]
init_dist = np.array([0.8,0.04,0.05,0.11])
rpe = Robust_pol_Kl_uncertainity(mr_obj.nS, mr_obj.nA, cost_list, init_dist)
policy = np.ones((nS,nA))*0.5
P = mr_obj.gen_probability()
C_KL = 0.05
print(rpe.evaluate_policy(policy, P, C_KL, 1,0))'''