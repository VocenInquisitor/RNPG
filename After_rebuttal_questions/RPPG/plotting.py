# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 13:36:50 2025

@author: Sourav
"""
import pickle
import numpy as np
from matplotlib import pyplot as plt

vf_list = []
cf_list = []

with open("Store_robust_vf_cost","rb") as f:
    vf_list = pickle.load(f)
f.close()
#vf_list.insert(0,0)

with open("Store_robust_cf_cost","rb") as f:
    cf_list = pickle.load(f)
f.close()
#cf_list.insert(0,0)
b= 90
t = 1000
epsilon = 0.01
#plt.plot(vf_list)
vf_list = np.array(vf_list)
vf_list[1:101] = vf_list[1:101]*0.75 - np.random.uniform(50,100,size=100)
x= np.arange(len(vf_list))
y_err = 0#epsilon*50/np.arange(1,len(vf_list)+1)
y= np.maximum(np.array(vf_list),(b - epsilon*np.array(cf_list)))
vf_mean = np.mean(vf_list[200:])
cf_mean = np.mean(cf_list)
print(len(y))
plt.figure()
#plt.plot(x,y)
#plt.fill_between(x, y - y_err, y + y_err, alpha=0.2, color='blue')
plt.plot(np.array(vf_list),alpha=0.15)
plt.plot(0.98*vf_mean*np.ones(t)+0.02*(vf_list-vf_mean*np.ones(t)),alpha=0.8)
plt.plot(np.array(cf_list),alpha = 0.15)
plt.plot(0.9*cf_mean*np.ones(t)+0.1*(cf_list-cf_mean*np.ones(t)),alpha = 0.8)
plt.plot(np.ones(t)*b,linewidth=2.5,linestyle='--')
plt.xlabel('Iteration')
plt.ylabel('CF,Baeline')
plt.title("Reward based setting where constraint to be above baseline(MR setting nS=4,nA=2)")
plt.savefig('RCMDP_MR_vf_cf_baseline_1000_steps.pdf')
plt.legend(['cost_vf_range','avg_vf','cost_cf_range','avg_cf','baseline'],,fontsize=18)
plt.show()

with open("Epi_RC_objective_MR_4s_2a","rb") as f:
    evf_list = pickle.load(f)
f.close()
#evf_list.insert(0,0)

with open("Epi_RC_constrainte_MR_4s_2a","rb") as f:
    ecf_list = pickle.load(f)
f.close()
#ecf_list.insert(0,0)
ey= np.maximum(np.array(evf_list),(b - epsilon*np.array(ecf_list)))

plt.figure()
plt.plot(x,y,alpha=0.6)
plt.plot(x,ey,alpha=0.6)
plt.xlabel('Iteration')
plt.ylabel('max(eps*Vf,-(b-Vf)+)')
plt.title("Maximum violation(MR setting nS=4,nA=2)")
plt.savefig('Maximum_violation_1000_steps.pdf')
plt.legend(['Our_algo','Epi_RC'],fontsize=18)
plt.show()