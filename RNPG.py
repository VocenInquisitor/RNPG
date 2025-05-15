# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 20:47:19 2025

@author: Sourav
"""

import numpy as np
import pickle


'''class TabularPolicy(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(n_states, n_actions))

    def forward(self, state):
        probs = F.softmax(self.logits[state], dim=-1)
        return torch.distributions.Categorical(probs=probs)'''

class RNPG:
    def __init__(self,env,r_oracle_obj,alpha,lambda_,pol_obj,P,cost_list,b,policy_space=None,finite_pol=1):
        self.env = env
        self.oracle_obj = r_oracle_obj
        self.alpha = alpha
        self.lambda_ = lambda_
        #self.theta = theta
        self.value_func_store = []
        self.cost_func_store = []
        self.value_func_grad_store = []
        self.cost_func_grad_store = []
        self.pol_obj = pol_obj
        self.P = P
        self.cost_list = cost_list
        self.C_KL = 0.5
        self.policy_space = policy_space
        self.finite_pol = finite_pol
        self.b = b
    def find_choice(self,pol,t):
        J,J_grad = None,None
        J_v,J_v_grad = self.oracle_obj.evaluate_policy(pol,self.P,self.C_KL,0,t)#send_for_value function
        J_c,J_c_grad = self.oracle_obj.evaluate_policy(pol,self.P,self.C_KL,1,t)#send_for_cost_function
        self.value_func_store.append(J_v)
        self.value_func_grad_store.append(J_v_grad)
        self.cost_func_store.append(J_c)
        self.cost_func_grad_store.append(J_c_grad)
        choice = np.max([J_v/self.lambda_,J_c-self.b])
        #choice = np.max([J_v,np.max(0,J_c-self.b)])
        #choice = np.max([J_v,J_c-self.b])
        if(choice == 0):
            J,J_grad = J_v,J_v_grad
        else:
            J,J_grad = J_c,J_c_grad
        return J,J_grad
    def onehot(self,s):
        ret_val = np.zeros(self.env.nS,dtype=np.int16)
        ret_val[s] = 1
        return ret_val
    def get_pol(self):
        states,actions = self.env.nS,self.env.nA
        ret_pol = np.zeros((states,actions))
        for s in range(states):
            ret_pol[s] = self.pol_obj.forward(self.onehot(s))
        return ret_pol
    def compute_fisher(self):
        d, nA = self.env.nS,self.env.nA
        fisher = np.zeros((d * nA, d * nA))
    
        for s in range(d):
            state = self.onehot(s)
            pi = self.pol_obj.forward(state)  # shape: [nA]
            #pi = softmax(logits)  # shape: [nA]
    
            for a in range(nA):
                one_hot_a = np.zeros(nA)
                one_hot_a[a] = 1.0
                #print("one_hot:",one_hot_a)
                #print("pi:",pi)
                #print("mod:",one_hot_a - pi)
                grad_matrix = np.outer(state, one_hot_a - pi)  # shape: [d, nA]
                #print(grad_matrix.shape)
                #print("grad_matrix=",grad_matrix)
                grad_vector = grad_matrix.flatten()  # shape: [d * nA]
                #print(pi.shape)
                fisher += pi[a] * np.outer(grad_vector, grad_vector)
        #print("Fischer without average:",fisher)
        fisher /= d
        return fisher
    def train_all(self,T):
        #print("initial theta:",self.pol_obj._get_param_())
        for t in range(T):
            #pol = self.get_policy(self.theta)
            pol = self.get_pol()
            #print(pol)
            J,J_grad = self.find_choice(pol,t)
            F = self.compute_fisher()
            #print(F.shape)
            J_grad_flat = J_grad.flatten()
            g_tilde = np.matmul(np.linalg.pinv(F),J_grad_flat)
            g = np.reshape(g_tilde,(self.env.nS,self.env.nA))
            print(g)
            self.theta = self.pol_obj._get_param_()
            self.theta = self.theta - 0.5/self.lambda_*g
            #print(self.theta)
            self.pol_obj._set_param_(self.theta)
        #print(self.theta)
        with open("Value_function_"+str(self.lambda_),"wb") as f:
            pickle.dump(self.value_func_store,f)
        f.close()
        with open("Value_function_grad_store_"+str(self.lambda_),"wb") as f:
            pickle.dump(self.value_func_grad_store,f)
        f.close()
        with open("Cost_function_"+str(self.lambda_),"wb") as f:
            pickle.dump(self.cost_func_store,f)
        f.close()
        with open("Cost_function_grad_store_"+str(self.lambda_),"wb") as f:
            pickle.dump(self.cost_func_grad_store,f)
        f.close()
        
            