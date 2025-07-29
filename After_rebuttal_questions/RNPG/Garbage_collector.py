# -*- coding: utf-8 -*-
"""
Created on Tue May 20 17:05:44 2025

@author: gangu
"""
import numpy as np

class cleaning_bot:
    def __init__(self,d):
        self.d = d
        self.nS = int(np.power(d,2))
        self.nA = 4
    def wrap(self,s):
        return int(s[0]*self.d+s[1])
    def unwrap(self,s):
        s = int(s)
        ret = np.zeros(2)
        ret[-1] = s%self.d;
        s = s//self.d
        ret[0] = s%self.d
        return ret
    def __make__(self):
        P = np.zeros((self.nA,self.nS,self.nS))
        R = np.zeros((self.nS,self.nA))
        C = np.ones((self.nS,self.nA))*0.001
        for x in range(self.d):
            for y in range(self.d):
                if(x==self.d-1 and y ==self.d-1):
                    x1,x2,x3,x4 = 3,3,3,3
                    y1,y2,y3,y4 = 3,3,3,3
                else:
                    if(x==0):
                        x1,x2,x3,x4 = x,x,x+1,x #(UP,LEFT,DOWN,RIGHT)
                    elif(x==3):
                        x1,x2,x3,x4 = x-1,x,x,x #(UP,LEFT,DOWN,RIGHT)
                    else:
                        x1,x2,x3,x4 = x-1,x,x+1,x #(UP,LEFT,DOWN,RIGHT)
                    if(y==0):
                        y1,y2,y3,y4 = y,y,y,y+1 #(UP,LEFT,DOWN,RIGHT)
                    elif(y==self.d-1):
                        y1,y2,y3,y4 = y,y-1,y,y #(UP,LEFT,DOWN,RIGHT)
                    else:
                        y1,y2,y3,y4 = y,y-1,y,y+1 #(UP,LEFT,DOWN,RIGHT)
                current = self.wrap([x,y])
                up = self.wrap([x1,y1])
                left = self.wrap([x2,y2])
                down = self.wrap([x3,y3])
                right = self.wrap([x4,y4])
                #print(current,up,left,down,right)
                P[0,current,up] = P[0,current,up]+1/2.
                P[0,current,down] = P[0,current,down]+1/6.
                P[0,current,left] = P[0,current,left]+1/6.
                P[0,current,right] = P[0,current,right]+1/6.
                ########################################################
                P[1,current,up] = P[1,current,up]+1/6.
                P[1,current,down] = P[1,current,down]+1/6.
                P[1,current,left] = P[1,current,left]+1/2.
                P[1,current,right] = P[1,current,right]+1/6.
                ###########################################################
                P[2,current,up] = P[2,current,up]+1/6.
                P[2,current,down] = P[2,current,down]+1/2.
                P[2,current,left] = P[2,current,left]+1/6.
                P[2,current,right] = P[2,current,right]+1/6.
                ###########################################################
                P[3,current,up] = P[3,current,up]+1/6.
                P[3,current,down] = P[3,current,down]+1/6.
                P[3,current,left] = P[3,current,left]+1/6.
                P[3,current,right] = P[3,current,right]+1/2.
                #print(P[0,current])
                #return
        R[15,:] = 1
        R[:-15,:] = 0.001
        self.P,self.R,self.C = P,R,C
    def get_reward_cost(self):
        for i in range(3):
            x = np.random.choice(self.d)
            y = np.random.choice(self.d)
            if((x==3) and (y==3)):
                continue
            else:
                s = self.wrap([x,y])
                rv = np.random.choice([0,1])
                self.C[s,:] = rv      
        return self.R,self.C
    def get_nominal_model(self):
        return self.P