# -*- coding: utf-8 -*-
"""
Created on Tue May 20 13:08:43 2025

@author: gangu
"""

import numpy as np
class my_env:
    def __init__(self,d):
        self.nS,self.nA = int(d*d),4
        self.d = d
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
        end_state = 15
        P = np.zeros((self.nA,self.nS,self.nS))
        R = np.zeros((self.nS,self.nA))
        C = np.zeros((self.nS,self.nA))
        
        for x in range(4):
            for y in range(4):
                if(x==3 and y ==3):
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
                    elif(y==3):
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
        #print(P[0,0])
        R[15,:] = 10
        R[:-15,:] = 0.01
        C[15,:] = -2
        C[:-15,:] = 0.1
        #P[0,0],P[0,3],P[0,12],P[0,15] = P[0,0]
        return P,R,C