# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 18:18:16 2025

@author: gangu
"""

import numpy as np
from matplotlib import pyplot as plt
import pickle

b = 42.5
x = np.arange(0,1001)
with open("Value_function_kl_lambda_RS_10.pkl","rb") as f:
    vf_list1 = pickle.load(f)
f.close()


with open("Value_function_kl_lambda_RS_15.pkl","rb") as f:
    vf_list2 = pickle.load(f)
f.close()


with open("Value_function_kl_lambda_RS_30.pkl","rb") as f:
    vf_list3 = pickle.load(f)
f.close()


with open("Value_function_kl_lambda_RS_50.pkl","rb") as f:
    vf_list4 = pickle.load(f)
f.close()

with open("Cost_function_kl_lambda_RS_10.pkl","rb") as f:
    cf_list1 = pickle.load(f)
f.close()

with open("Cost_function_kl_lambda_RS_15.pkl","rb") as f:
    cf_list2 = pickle.load(f)
f.close()

with open("Cost_function_kl_lambda_RS_30.pkl","rb") as f:
    cf_list3 = pickle.load(f)
f.close()

with open("Cost_function_kl_lambda_RS_50.pkl","rb") as f:
    cf_list4 = pickle.load(f)
f.close()

#"Value function comparison"

plt.figure()
plt.plot(vf_list1)
plt.plot(vf_list2)
plt.plot(vf_list3)
plt.plot(vf_list4)
plt.legend(['lambda=10','lambda=30','lambda=15','lambda=50'],fontsize=18)
plt.xlabel('Iteration',fontsize=18)
plt.ylabel('Expected values of vf',fontsize=18)
plt.title('Effect of lambda(RNPG)',fontsize=18)
plt.savefig('Comparison for Value functions.pdf')
plt.show()

plt.figure()
plt.plot(cf_list1)
plt.plot(cf_list2)
plt.plot(cf_list3)
plt.plot(cf_list4)
plt.plot(np.ones(len(cf_list1))*b,linestyle="-.")
plt.fill_between(x, b,50, color='red', alpha=0.1, label='Safe region')
# Shade the region between y1 and y2 where y1 < y2
plt.fill_between(x,20, b, color='blue', alpha=0.1, label='UnSafe_region')
plt.legend(['lambda=10','lambda=30','lambda=15','lambda=50','baseline','Unsafe region','Safe region'],fontsize=18)
plt.xlabel('Iteration',fontsize=18)
plt.ylabel('Expected values of cf',fontsize=18)
plt.title('Effect of lambda(RNPG)',fontsize=18)
plt.savefig('Comparison for Cost functions.pdf')
plt.show()

