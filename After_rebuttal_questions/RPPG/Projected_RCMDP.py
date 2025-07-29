# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:53:04 2025

@author: Sourav
"""

import numpy as np
from Garbage_collector import cleaning_bot
from KL_uncertainity_evaluator import Robust_pol_Kl_uncertainity
#import torch
import pickle
import time

def onehot(policy_space,nS,nA):
    ret_pol = []
    for pol in policy_space:
        policy = np.zeros((nS,nA))
        for s,j in enumerate(pol):
            policy[s,j]=1
        ret_pol.append(policy)
    return np.array(ret_pol,dtype=np.int16)

def Proj(policy,V,Pi,grad,ch=0):
    alpha = 0.1
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
    
env = cleaning_bot(4)
nS,nA = env.nS,env.nA
env.__make__()
P = env.get_nominal_model()
R,C = env.get_reward_cost()
b_constraint = 50
cost_list = [R,C]
discount = 0.99
env_nm = "Garbage_collector"
ch = 0
alpha = 0.001
exp=0
#P,R,C = mr_obj.gen_probability(),mr_obj.gen_expected_reward(ch),mr_obj.gen_expected_cost(exp)
#nS,nA = mr_obj.nS,mr_obj.nA
cost_list = [R,C]
init_dist = np.random.normal(loc = 1,scale=2,size=nS)
init_dist = np.exp(init_dist)
init_dist = init_dist/np.sum(init_dist)
pol_eval = Robust_pol_Kl_uncertainity(nS, nA, cost_list, init_dist,alpha)
C_KL = 0.05 # what will be this parameter
store=[]
eps = 0.01
b = 50

#####Remember to convert policy to one_hot encoding
policy_space= []
n_pol = np.power(nA,nS)
vf_store = []
cf_store = []
for i in range(1,n_pol):
    policy_space.append(list(map(int,format(i, '04b'))))
policy_space = np.array(policy_space)
policy_space = onehot(policy_space,nS,nA)
choice_of_policy = np.random.choice(len(policy_space))
policy = policy_space[choice_of_policy]
store_pol = []
#print(policy_space)
with open("nominal_model","rb") as f:
    P_nominal = pickle.load(f)
f.close()
T = 1000
count = 0;
start_time = time.time()
for t in range(T):
    Vr,gradr = pol_eval.evaluate_policy(policy, P_nominal, C_KL, 0,t)
    Vc,gradc = pol_eval.evaluate_policy(policy, P_nominal, C_KL, 1,t)
    if(Vc < b - eps):
        policy = Proj(policy,Vr,policy_space,gradr) ##define this
    else:
        count+=1
        policy = Proj(policy,Vc,policy_space,gradc,1)
    #print("New policy:",policy)
    #store.append(np.min(Vr,-np.clip((b-Vc),0,np.inf)))
    vf_store.append(Vr)
    cf_store.append(Vc)
    store_pol.append(policy)
    #print("One step done")
#print(np.argmax(store))
print("Execution time:",time.time()-start_time," secs")
'''print(count)
with open("Store_robust_output_cost","wb") as f:
    pickle.dump(store,f)
f.close()

with open("Store_robust_vf_cost","wb") as f:
    pickle.dump(vf_store,f)
f.close()

with open("Store_robust_cf_cost","wb") as f:
    pickle.dump(cf_store,f)
f.close()'''
