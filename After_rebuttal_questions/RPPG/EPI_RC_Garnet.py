# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 05:10:43 2025

@author: Sourav
"""

import numpy as np
import pickle
from KL_uncertainity_evaluator import Robust_pol_Kl_uncertainity
from Garnet import Garnet
import time
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
        self.alpha = 0.1
        self.eps = 0.01
        self.pol_eval = Robust_pol_Kl_uncertainity(self.nS, self.nA, self.cost_list, init_dist,self.alpha)
        self.b_constraint = b_constraint
        self.discount = discount
        self.objective_list = []
        self.constraint_list = []
        self.env_nm = env_nm
    def Proj(self,policy,V,grad,ch=0):
        alpha = 0.00001
        #print(grad)
        if(ch==0):
            policy = policy + alpha* grad
        else:
            policy = policy + alpha*grad
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
        pol = np.ones((self.nS,self.nA))*(1/self.nA)
        for t in range(self.T):
            Vr,gradr = self.pol_eval.evaluate_policy(pol, self.P, self.C_KL, 0,t)
            Vc,gradc = self.pol_eval.evaluate_policy(pol, self.P, self.C_KL, 1,t)
            self.stored_VR['vr'].append(Vr)
            self.stored_VR['cr'].append(Vc)
            #print(Vr,Vc)
            #input()
            V,g = 0,0
            if(b-Vr>self.b_constraint-Vc):
                V,g,ch  = Vr,gradr,0
            else:
                V,g,ch = Vc,gradc,1
            #print(Vr)
            #print(pol)
            #input()
            pol = self.Proj(pol,V,g,ch)
        return pol
    def main_algo(self):
        K = 50
        low = 0
        upper = 120
        pol = None
        self.stored_VR = {'vr':[],'cr':[]}
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
            del_k = np.max([b-Vr,self.b_constraint-Vc])
            #print(del_k)
            if(del_k>self.eps):
                low = b;
                upper = upper;
            else:
                low = low
                upper = b
            #input()
        with open("Epi_RC_objective_"+self.env_nm+"new_set_T="+str(self.T)+"_K="+str(K),"wb") as f:
            pickle.dump(self.stored_VR['vr'],f)
        f.close()
        with open("Epi_RC_constrainte_"+self.env_nm+"new_set","wb") as f:
            pickle.dump(self.stored_VR['cr'],f)
        f.close()
        return pol
nS, nA = 15,20
env = Garnet(nS,nA)
#ch,exp = 0,0 #check this might generate error
P,R,C = env. gen_nominal_prob(),env.gen_expected_reward(),env.gen_expected_constraint()
b_constraint = 90
cost_list = [R,C]
discount = 0.9
env_nm = "Garnet_15s_20a"
T = 250
C_KL = 0.1
model = Epi_RC(env, P, T, C_KL, cost_list, b_constraint, discount, env_nm)
start_tm = time.time()
fin_pol  = model.main_algo()
print("Execution time:",time.time()-start_tm )          
print("All files saved. Training complete")
print(fin_pol)