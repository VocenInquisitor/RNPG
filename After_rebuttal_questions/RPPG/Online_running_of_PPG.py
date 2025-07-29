# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:43:03 2025

@author: Sourav
"""
import numpy as np
from KL_uncertainity_evaluator import Robust_pol_Kl_uncertainity
from Machine_Rep import Machine_Replacement
import pickle

def onehot(policy_space,nS,nA):
    ret_pol = []
    for pol in policy_space:
        policy = np.zeros((nS,nA))
        for s,j in enumerate(pol):
            policy[s,j]=1
        ret_pol.append(policy)
    return np.array(ret_pol,dtype=np.int16)

def Proj(policy,V,Pi,grad,count,ch):
    alpha = 0.001
    alphac = 0.1 
    #print(grad)
    if(ch==0):
        policy = policy - alpha* grad
    else:
        policy = policy - alphac*grad
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

mr_obj = Machine_Replacement()
ch = 0
alpha = 0.001
exp=0
P,R,C = mr_obj.gen_probability(),mr_obj.gen_expected_reward(ch),mr_obj.gen_expected_cost(exp)
nS,nA = mr_obj.nS,mr_obj.nA
cost_list = [R,C]
init_dist = np.array([0.8,0.04,0.05,0.11])
pol_eval = Robust_pol_Kl_uncertainity(nS, nA, cost_list, init_dist,alpha)
C_KL = 0.5 # what will be this parameter
store=[]
lambda_ = 30
b = 30

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
count = 1;

for t in range(T):
    Vr,gradr = pol_eval.evaluate_policy(policy, P_nominal, C_KL, 0,t)
    Vc,gradc = pol_eval.evaluate_policy(policy, P_nominal, C_KL, 1,t)
    
    ch = np.argmax([Vr/lambda_,(Vc-b)])
    
    if(ch==0):
        policy = Proj(policy,Vr,policy_space,gradr/lambda_,count,0) ##define this
    else:
        count+=1
        policy = Proj(policy,Vc,policy_space,gradc,count,1)
    #print("New policy:",policy)
    #store.append(np.min(Vr,-np.clip((b-Vc),0,np.inf)))
    vf_store.append(Vr)
    cf_store.append(Vc)
    store_pol.append(policy)
    #print("One step done")
#print(np.argmax(store))
print(count)
with open("Store_robust_output_cost_without_oscillation_MR_10_final","wb") as f:#final is absent in previous version
    pickle.dump(store,f)
f.close()

with open("Store_robust_vf_cost_without_oscillation_MR_10_final","wb") as f:
    pickle.dump(vf_store,f)
f.close()

with open("Store_robust_cf_cost_without_oscillation_MR_10_final","wb") as f:
    pickle.dump(cf_store,f)
f.close()
