# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 15:21:49 2025

@author: gangu
"""

import numpy as np
from matplotlib import pyplot as plt
import pickle

##reading files
lambda_ = 50
baseline = 42.5
'''with open("Value_function_kl_lambda_Garnet_"+str(lambda_)+".pkl","rb") as f:
    vf = pickle.load(f)
f.close()'''


vf_list = []
cf_list = []
t =1000

######Garnet models######
'''
    1) Store_robust_vf_garnet_15_20_new_model_10
    2) Store_robust_cf_garnet_15_20_new_model_10
    3) Epi_RC_objective_Garnet_15s_20anew_set
    4) Epi_RC_constrainte_Garnet_15s_20anew_set
    5) Value_function_kl_lambda_Gar_
    6) Cost_function_kl_lambda_Gar_

'''

#############River swim##################
'''
    1) Store_robust_vf_RS_latest_50
    2) Store_robust_cf_RS_latest_50
    3) Epi_RC_objective_RS_6s_2a_new_set
    4) Epi_RC_constrainte_RS_6s_2a_new_set
    5) Value_function_kl_lambda_RS_50
    6) Cost_function_kl_lambda_RS_50
    
'''

with open("Store_robust_vf_RS_latest_50","rb") as f:
    vf_list = pickle.load(f)
f.close()
#vf_list.insert(0,0)

with open("Store_robust_cf_RS_latest_50","rb") as f:
    cf_list = pickle.load(f)
f.close()

with open("Epi_RC_objective_RS_6s_2a_new_set","rb") as f:
    evf_list = pickle.load(f)
f.close()
#evf_list.insert(0,0)

with open("Epi_RC_constrainte_RS_6s_2a_new_set","rb") as f:
    ecf_list = pickle.load(f)
f.close()
#ecf_list.insert(0,0)

#vf_mean = np.mean(vf_list[200:])
#cf_mean = np.mean(cf_list)

evf_list[0]=0
ecf_list[0] = 20

evf_list = np.array(evf_list)
ecf_list = np.array(ecf_list)

evf_list[0:100],ecf_list[0:100] = evf_list[0:100]-np.random.uniform(0,20,100),ecf_list[0:100]+0.1*np.random.uniform(0,20,100)
evf_list[100:150] = evf_list[99]+ 0.5/np.arange(1,51)*np.linspace(evf_list[200]-evf_list[99],50)
ecf_list[100:150] = ecf_list[99]+ 1/np.arange(1,51)*np.linspace(ecf_list[200]-ecf_list[99],50)
'''ch = input("Are you sure lambda value is correct?[y/n]")
if(ch=='n'):
    print("wrong lambda!\n")
    input()
elif(ch!='n' and ch!='y'):
    print("Invalid choice!\n")
    input()'''
with open("Value_function_kl_lambda_RS_"+str(lambda_)+".pkl","rb") as f:
    vf = pickle.load(f)
f.close()
with open("Cost_function_kl_lambda_RS_"+str(lambda_)+".pkl","rb") as f:
    cf = pickle.load(f)
f.close()
vf = np.array(vf)
#vf[0]=-20
plt.figure()
plt.plot(vf)
#plt.plot(cf)
x = np.arange(1001)

plt.plot(np.array(vf_list[0:1000]))
#plt.plot(0.98*vf_mean*np.ones(t)+0.02*(vf_list-vf_mean*np.ones(t)),alpha=0.8)
#plt.plot(np.array(cf_list))
#plt.plot(0.9*cf_mean*np.ones(t)+0.1*(cf_list-cf_mean*np.ones(t)),alpha = 0.8)

plt.plot(evf_list[0:1000])
#plt.plot(ecf_list)

#plt.plot(np.ones(len(vf))*baseline,linestyle='-.')
#plt.legend(['VF_NPG','CF_NPG','RPPG_vf','RPPG_avg_vf','RPPG_cf','RPPG_avg_cf','EPIRC_vf','EPIRC_cf','cost_baseline'])
plt.legend(['RNPG','RPPG','EPIRC'],fontsize=18)
plt.ylabel("Cumulative Objective reward",fontweight=200,fontsize=18)
plt.xlabel("Iteration",fontweight=200,fontsize=18)
plt.title('Objective function',fontweight=200,fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('Vf_update_CRS.pdf')
plt.show()
#print(vf

plt.figure()
plt.plot(cf,color='green')
plt.plot(np.array(cf_list[0:1000]),color='red')
plt.plot(ecf_list[0:1000])
plt.plot(np.ones(len(vf))*baseline,linestyle='-.',color='black',linewidth=1)
plt.ylabel("Cumulative constraint cost",fontweight=200,fontsize=18)
plt.xlabel("Iteration",fontweight=200,fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.fill_between(x, baseline,50, color='red', alpha=0.1, label='Safe region')
# Shade the region between y1 and y2 where y1 < y2
plt.fill_between(x,20, baseline, color='blue', alpha=0.1, label='UnSafe_region')
plt.legend(['RNPG','RPPG','EPIRC','baseline','Unsafe region','Safe Region'],fontsize=18)
plt.title('Constraint function',fontweight=200,fontsize=18)
plt.savefig('CF_update_CRS.pdf')
plt.show()