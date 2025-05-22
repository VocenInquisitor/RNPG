# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 17:00:05 2025

@author: Sourav
"""

import numpy as np
import pickle
from KL_uncertainity_evaluator import Robust_pol_Kl_uncertainity
#from Machine_Rep import Machine_Replacement
import time
from Forzen_lk import my_env
class Epi_RC:
    def __init__(self,env,P,T,C_KL,cost_list,b_constraint,discount,env_nm):
        self.nS = env.nS
        self.nA = env.nA
        self.T = T
        self.P = P
        self.C_KL = C_KL
        self.cost_list = cost_list
        init_dist = np.random.normal(loc = 1,scale=2,size=self.nS)
        init_dist = np.exp(init_dist)
        init_dist = init_dist/np.sum(init_dist)
        self.alpha = 0.001
        self.eps = 0.01
        self.pol_eval = Robust_pol_Kl_uncertainity(self.nS, self.nA, self.cost_list, init_dist,self.alpha)
        self.b_constraint = b_constraint
        self.discount = discount
        self.objective_list = []
        self.constraint_list = []
        self.env_nm = env_nm
    def Proj(self,policy,V,grad,ch=0):
        alpha = 1
        #print(grad)
        if(ch==0):
            policy = policy - alpha* grad
        else:
            policy = policy - alpha*grad
        #smallest_distance = np.argmin([np.linalg.norm(policy-pi) for pi in Pi])
        nS,nA = policy.shape
        #print(np.any(policy<0))
        #print("Before change:",policy)
        if(np.any(policy<0)):
            policy = np.exp(policy)
        #print("After change:",policy)
        for s in range(nS):
            policy[s] = policy[s]/np.sum(policy[s])
        #print(policy)
        #input()
        return policy
    def find_pol_for_epi(self,b):
        pol = np.ones((self.nS,self.nA))*1/self.nA
        for t in range(self.T):
            Vr,gradr = self.pol_eval.evaluate_policy(pol, self.P, self.C_KL, 0,t)
            Vc,gradc = self.pol_eval.evaluate_policy(pol, self.P, self.C_KL, 1,t)
            #print(Vr,Vc)
            #input()
            V,g = 0,0
            if(b-Vr>Vc-self.b_constraint):
                V,g = Vr,gradr
            else:
                V,g = Vc,gradc
            #print(Vr)
            #print(pol)
            #input()
            self.objective_list.append(Vr)
            self.constraint_list.append(Vc)
            pol = self.Proj(pol,V,g)
        return pol
    def main_algo(self):
        K = 10
        low = 0
        upper = int(1/(1-self.discount))
        pol = None
        for k in range(K):
            b = (low+upper)/2
            #print("b=",b)
            pol = self.find_pol_for_epi(b)
            #print("pol=",pol)
            Vr,gradr = self.pol_eval.evaluate_policy(pol, self.P, self.C_KL, 0,k)
            Vc,gradc = self.pol_eval.evaluate_policy(pol, self.P, self.C_KL, 1,k)
            #Vr,Vc = Vr,Vc
            #print(Vr,Vc)
            
            #print(Vr-b)
            #print(Vc-self.b_constraint)
            del_k = np.max([b-Vr,Vc-self.b_constraint])
            #print(del_k)
            if(del_k>0):
                low = b;
                upper = upper;
            else:
                low = low
                upper = b
            #input()
        with open("Epi_RC_objective_"+self.env_nm+"_new_set","wb") as f:
            pickle.dump(self.objective_list,f)
        f.close()
        with open("Epi_RC_constrainte_"+self.env_nm+"_new_set","wb") as f:
            pickle.dump(self.constraint_list,f)
        f.close()
        return pol
#nS, nA = 6,2
env = my_env(4)
nS,nA = 16,4
P,R,C = env.__make__()
R = R/np.max(R)
C = np.exp(C)
C = C/np.max(C)
b_constraint = 50
cost_list = [R,C]
discount = 0.99
env_nm = "Frozen_lake"
T=100
C_KL = 0.1
model = Epi_RC(env, P, T, C_KL, cost_list, b_constraint, discount, env_nm)
start_tm = time.time()
fin_pol  = model.main_algo()
print("Execution time:",time.time()-start_tm )          
print("All files saved. Training complete")


'''env = Machine_Replacement()
ch,exp = 0,0
P,R,C = env.gen_probability(),env.gen_expected_reward(ch),env.gen_expected_cost(exp)
b_constraint = 25
cost_list = [R,C]
discount = 0.9
env_nm = "MR_4s_2a"
T = 250
C_KL = 0.1
model = Epi_RC(env, P, T, C_KL, cost_list, b_constraint, discount, env_nm)
start_tm = time.time()
fin_pol  = model.main_algo()
print("Execution time:",time.time()-start_tm )          
print("All files saved. Training complete")
print(fin_pol)'''
            
        