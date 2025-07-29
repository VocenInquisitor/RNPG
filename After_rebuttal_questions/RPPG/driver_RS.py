# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:20:48 2025

@author: Sourav
"""

import numpy as np
from Machine_Rep import River_swim
from KL_uncertainity_evaluator import Robust_pol_Kl_uncertainity
#import torch
import pickle

def onehot(policy_space,nS,nA):
    ret_pol = []
    for pol in policy_space:
        policy = np.zeros((nS,nA))
        for s,j in enumerate(pol):
            policy[s,j]=1
        ret_pol.append(policy)
    return np.array(ret_pol,dtype=np.int16)

def Proj(policy,V,grad,ch=0):
    alpha = 0.11
    #print(grad)
    if(ch==0):
        policy = (1-alpha)*policy + alpha* grad
    else:
        policy = (1-alpha)*policy - alpha*grad
    #smallest_distance = np.argmin([np.linalg.norm(policy-pi) for pi in Pi])
    nS,nA = policy.shape
    #print(np.any(policy<0))
    #print("Before change:",policy)
    if(np.any(policy<0)):
        policy = np.exp(policy)
    #print("After change:",policy)
    for s in range(nS):
        policy[s] = policy[s]/np.sum(policy[s])
    #print("Ne_pol_in_function,",policy)
    #input()
    return policy

env = River_swim()
alpha = 0.001
exp=0
nS,nA = env.nS,env.nA
P,R,C = env.gen_probability(),env.gen_expected_reward(),env.gen_expected_cost()
cost_list = [R,C]
np.random.seed(1)
init_dist = np.random.normal(loc = 1,scale=2,size=nS)
init_dist = np.exp(init_dist)
init_dist = init_dist/np.sum(init_dist)
pol_eval = Robust_pol_Kl_uncertainity(nS, nA, cost_list, init_dist,alpha)
C_KL = 1 # what will be this parameter
store=[]
eps = 0.01
b = 40

#####Remember to convert policy to one_hot encoding
vf_store = []
cf_store = []

policy = np.zeros((nS,nA))
for s in range(nS):
    policy[s,0] = np.random.random()
    policy[s,1] = 1 - policy[s,0]
#print(policy)
store_pol = []
#print(policy_space)

#f.close()
T = 1000
count = 0;

for t in range(T):
    Vr,gradr = pol_eval.evaluate_policy(policy, P, C_KL, 0,t)
    Vc,gradc = pol_eval.evaluate_policy(policy, P, C_KL, 1,t)
    #print(Vc)
    #print(gradc)
    #input()
    if(Vc <  b - eps):
        policy = Proj(policy,Vr,gradr) ##define this
    else:
        count+=1
        policy = Proj(policy,Vc,gradc,1)
    #print("New policy:",policy)
    #store.append(np.min(Vr,-np.clip((b-Vc),0,np.inf)))
    vf_store.append(Vr)
    cf_store.append(Vc)
    store_pol.append(policy)
    #print("One step done")
#print(np.argmax(store))
print(count)
with open("Store_robust_output_RS_new_way","wb") as f:
    pickle.dump(store,f)
f.close()

with open("Policy_stored_robust_output_RS_new_way","wb") as f:
    pickle.dump(store_pol,f)
f.close()
with open("Store_robust_vf_RS_new_way","wb") as f:
    pickle.dump(vf_store,f)
f.close()

with open("Store_robust_cf_RS_new_way","wb") as f:
    pickle.dump(cf_store,f)
f.close()
