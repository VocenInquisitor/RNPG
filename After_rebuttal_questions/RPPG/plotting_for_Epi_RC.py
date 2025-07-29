# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:40:53 2025

@author: gangu
"""

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

with open("Epi_RC_objective_RS_6s_2a_new_set_after_rej","rb") as f:
    vf_list = pickle.load(f)
f.close()
vf_list.insert(0,0)

with open("Epi_RC_constrainte_RS_6s_2a_new_set_after_rej","rb") as f:
    cf_list = pickle.load(f)
f.close()
cf_list.insert(0,0)
b= 40
t = 1001
epsilon = 0.01
#plt.plot(vf_list)
vf_list,cf_list = np.array(vf_list),np.array(cf_list)
'''vf_list[1:100] = vf_list[1:100] - np.random.uniform(50,100,size=99)
cf_list[0:100] = cf_list[0:100] + np.random.uniform(1,20,size=100)
x= np.arange(len(vf_list))
y_err = 0#epsilon*50/np.arange(1,len(vf_list)+1)
y= np.minimum(np.array(vf_list),b - epsilon*np.array(cf_list))
vf_mean = np.mean(vf_list[200:])
cf_mean = np.mean(cf_list)
print(len(y))
plt.figure()
#plt.plot(x,y)
#plt.fill_between(x, y - y_err, y + y_err, alpha=0.2, color='blue')'''
plt.plot(np.array(vf_list),alpha=0.15)
#plt.plot(0.98*vf_mean*np.ones(t)+0.02*(vf_list-vf_mean*np.ones(t)),alpha=0.8)
plt.plot(np.array(cf_list),alpha = 0.15)
#plt.plot(0.9*cf_mean*np.ones(t)+0.1*(cf_list-cf_mean*np.ones(t)),alpha = 0.8)
plt.plot(np.ones(t)*b,linewidth=2.5,linestyle='--')
plt.xlabel('Iteration')
plt.ylabel('CF,Baeline')
plt.title("Reward based setting where constraint to be above baseline(MR setting nS=4,nA=2)")
plt.savefig('EPI_RS_vf_cf_baseline_1000_steps_after_rej.pdf')
#plt.legend(['cost_vf_range','avg_vf','cost_cf_range','avg_cf','baseline'])
#plt.legend(['cost_vf','cost_cf','baseline'])
plt.show()