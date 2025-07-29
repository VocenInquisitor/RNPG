# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 09:07:32 2025

@author: Sourav
"""
import numpy as np
import pickle

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
            #print(pol)
            a = np.random.choice(self.nA,p=pol[s])
            c = cost[s,a]
            next_s = np.random.choice(self.nS,p=self.P[a,s,:])
            V = np.array([np.sum([pol[s,a]*Q[s,a] for a in range(self.nA)]) for s in range(self.nS)])
            Q[s,a] = Q[s,a] + self.alpha*(c+self.gamma*(1-self.delta)*V[s]+self.gamma*self.delta*self.LSE(sigma,V)-Q[s,a])
            s = next_s
        return Q

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
        self.init_pol_theta = np.ones((nS,nA))*1/nA
        self.obj = SRTD(nS, nA, P, cost_list, init_state, gamma, delta)
        self.init_dist = init_dist
        self.max_Lambda = max_Lambda
        self.env_nm = env_nm
    def proj(self,policy):
        #print("1",policy)
        if(np.any(policy<0)):
            policy = np.exp(policy)
        for s in range(self.nS):
            policy[s] = policy[s]/np.sum(policy[s])
        #print("2",policy)
        #input()
        return policy
    def get_vf_cf(self):
        pol = self.init_pol_theta
        vf_list=[]
        cf_list=[]
        pol_list=[]
        eps = 0.1
        for t in range(self.T):
            T_inn = int((t+1)**1.5/(self.sigma**2))
            #print(T_inn)
            Q_r,Q_c = self.obj.find_SRTD(pol, 0, self.sigma, T_inn),self.obj.find_SRTD(pol,1,self.sigma,T_inn)
            #print(Q_r,Q_c)
            #input()
            pol1 = np.copy(pol)
            val = np.random.random()
            pol1[:,0] = pol1[:,0]+val/2
            pol1[:,1] = pol1[:,1]-val/2
            diff = pol1-pol
            #print(diff)
            #input();
            V_r = np.array([np.sum([pol[s,a]*Q_r[s,a] for a in range(self.nA)]) for s in range(self.nS)])
            V_r_1 = np.array([np.sum([pol1[s,a]*Q_r[s,a] for a in range(self.nA)]) for s in range(self.nS)])
            V_c = np.array([np.sum([pol[s,a]*Q_c[s,a] for a in range(self.nA)]) for s in range(self.nS)])
            V_c_1 = np.array([np.sum([pol1[s,a]*Q_c[s,a] for a in range(self.nA)]) for s in range(self.nS)])
            #print(V_r,V_c,V_r_1,V_c_1)
            #input();
            J_r,J_c,J_r1,J_c1 = np.dot(self.init_dist,V_r),np.dot(self.init_dist,V_c),np.dot(self.init_dist,V_r_1),np.dot(self.init_dist,V_c_1)
            #print(J_r,J_c,J_r1,J_c1)
            #input();
            grad_Vr = (J_r1-J_r)/diff
            grad_Vc = (J_c1-J_c)/diff
            #print(grad_Vr,grad_Vc)
            
            self.lambda_ = np.max([0,np.min([self.lambda_ - 1/self.beta*(J_c-self.b)-self.b/self.beta*self.lambda_,self.max_Lambda])])
            pol = self.proj(pol+self.alpha*(grad_Vr+self.lambda_*grad_Vc))
            #print(pol)
            #input()
            vf_list.append(J_r)
            cf_list.append(J_c)
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
        
        
        
            
        