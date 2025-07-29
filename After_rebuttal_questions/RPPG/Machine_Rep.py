import gym
from gym import spaces
import numpy as np


class River_swim:
    def __init__(self,nS = 6,nA =2):
        self.nS = nS
        self.nA = nA
    def gen_probability(self):
        self.P = np.zeros((self.nA,self.nS,self.nS))
        self.P[0,0,0] = 0.9
        self.P[0,0,1] = 0.1
        self.P[1,self.nS-1,self.nS-1] = 0.9
        self.P[1,self.nS-1,self.nS-2] = 0.1
        for s in range(1,self.nS-1):
            self.P[0,s,s] = 0.6
            self.P[1,s,s] = 0.6
            self.P[0,s,s-1] = 0.3
            self.P[1,s,s-1] = 0.1
            self.P[0,s,s+1] = 0.1
            self.P[1,s,s+1] = 0.3
        self.P[1,0,0] = 0.7
        self.P[1,0,1] = 0.3
        self.P[0,self.nS-1,self.nS-1] = 0.7
        self.P[0,self.nS-1,self.nS-2] = 0.3
        return self.P
    def gen_reward(self):
        self.R = np.zeros((self.nA,self.nS,self.nS))
        self.R[0,0,:] = 0.01
        self.R[1,self.nS-1,:] = 1
    def gen_expected_reward(self):
        self.R = np.zeros((self.nS,self.nA))
        self.R[0,0] = 0.1
        self.R[self.nS-1,1] = 1
        return self.R
    def gen_cost(self):
        self.C = np.zeros((self.nA,self.nS,self.nS))
        for s in range(self.nS):
            self.C[:,s,:] = s/10
        return self.C
    def gen_expected_cost(self):
        self.C = np.zeros((self.nS,self.nA))
        for s in range(self.nS):
            self.C[s,:] = s/20
        return self.C

class Machine_Replacement:
    def __init__(self,rep_cost=0.7,safety_cost=0.4,nS=4,nA=2):
        self.nS = nS;
        self.nA = nA;
        self.cost = np.linspace(0.2, 0.99,nS);
        self.rep_cost = rep_cost;
        self.safety_cost = safety_cost
    def gen_probability(self):
        self.P = np.zeros((self.nA,self.nS,self.nS));
        for i in range(self.nS):
            for j in range(self.nS):
                if(i<=j):
                    self.P[0,i,j]=(i+1)*(j+1);
                else:
                    continue;
            self.P[0,i,:]=self.P[0,i,:]/np.sum(self.P[0,i,:])
            self.P[1,i,0]=1;
        return self.P;
    def gen_reward(self,ch=2): #ch=0 means cost based, ch=1 means -cost based and ch=2 means rew = (1- cost) based
        self.R=np.zeros((self.nA,self.nS,self.nS));
        for i in range(self.nS):
            self.R[0,i,:] = self.cost[i];
            self.R[1,i,0] = self.rep_cost+self.cost[0];
        if(ch==0):
            return self.R;
        elif(ch==1):
            return -self.R;
        elif(ch==2):
            for s in range(self.nS):
                for a in range(self.nA):
                    for s_next in range(self.nS):
                        self.R[a,s,s_next] = 1 - self.R[a,s,s_next]
            return self.R
        else:
            print("Incorrect choice")
    def gen_expected_reward(self,ch=2):
        self.R = np.zeros((self.nS,self.nA));
        for i in range(self.nS):
            self.R[i,0] = self.cost[i];
            self.R[i,1] = self.rep_cost + self.cost[0];
        if(ch==0):
            return self.R
        elif(ch==1):
            return -self.R;
        elif(ch==2):
            for s in range(self.nS):
                for a in range(self.nA):
                    self.R[s,a] = 1 - self.R[s,a]
            return self.R
        else:
            print("Illegal choice")
    def gen_cost(self):
        self.C = np.zeros((self.nA,self.nS,self.nS));
        for i in range(self.nS):
            self.C[0,i] = np.ones(self.nS)*self.cost[i];
            self.C[1,i] = np.ones(self.nS)*(self.safety_cost + self.cost[0]);
        return self.C;
    def gen_expected_cost(self,exp=1):
        self.C = np.zeros((self.nS,self.nA));
        for i in range(self.nS):
            self.C[i,0] = self.cost[i];
            self.C[i,1] = self.safety_cost + self.cost[0];
        if(exp==0):
            self.C = self.gen_expected_reward()
        return self.C;
    
class MachineReplacementEnv(gym.Env):
    """
    Gym Wrapper for Machine Replacement Environment.
    """
    def __init__(self, rep_cost=0.7, safety_cost=0.4, nS=4, nA=2):
        super(MachineReplacementEnv, self).__init__()
        self.nS = nS
        self.nA = nA
        self.rep_cost = rep_cost
        self.safety_cost = safety_cost
        self.init_state = 0

        # Define action and observation spaces
        self.action_space = spaces.Discrete(nA)
        self.observation_space = spaces.Discrete(nS)

        # Initialize the Machine Replacement model
        self.machine_replacement = Machine_Replacement(rep_cost, safety_cost, nS, nA)
        self.P = self.machine_replacement.gen_probability()
        self.R = self.machine_replacement.gen_expected_reward()
        self.C = self.machine_replacement.gen_expected_cost()

        # Initialize the state
        self.state = 0
    def one_hot(self,s):
        ret_val = np.zeros(self.nS)
        ret_val[s] = 1
        return ret_val

    def reset(self, seed=None):
        """
        Resets the environment to the initial state.
        """
        np.random.seed(seed)
        self.state = self.init_state
        return self.one_hot(self.state)

    def step(self, action):
        """
        Executes an action in the environment.

        Args:
            action: The action to take (0 or 1).

        Returns:
            next_state: The next state after taking the action.
            reward: The reward received for the action.
            done: Whether the episode has ended (always False for this environment).
            info: A dictionary containing additional information (e.g., cost).
        """
        assert self.action_space.contains(action), "Invalid action"

        # Sample next state based on the transition probabilities
        next_state = np.random.choice(self.nS, p=self.P[action, self.state])

        # Get the expected reward and cost
        reward = self.R[action, self.state]
        cost = self.C[action, self.state]

        # Update the current state
        self.state = next_state

        # The environment is never "done"
        done = False

        return self.one_hot(next_state), reward, done, {"cost": cost}
    
class RiverSwimEnv(gym.Env):
    def __init__(self, nS=6, nA=2):
        super(RiverSwimEnv, self).__init__()
        self.nS = nS
        self.nA = nA

        # Define action and observation space
        self.action_space = spaces.Discrete(nA)
        self.observation_space = spaces.Discrete(nS)

        # Initialize dynamics
        self.river_swim = River_swim(nS, nA)
        self.P = self.river_swim.gen_probability()
        self.R = self.river_swim.gen_expected_reward()
        self.C = self.river_swim.gen_expected_cost()

        self.state = 3  # Start at the first state   
    def one_hot(self,s):
         ret_val = np.zeros(self.nS)
         ret_val[s] = 1
         return ret_val
    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state.
        Args:
            seed (int, optional): Random seed for reproducibility.
            options (dict, optional): Additional options for reset (currently unused).
        Returns:
            state (int): The initial state.
        """
        if seed is not None:
            np.random.seed(seed)
        self.state = 3  # Reset to the initial state
        return self.one_hot(self.state)

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        # Transition to the next state
        next_state = np.random.choice(
            self.nS, p=self.P[action, self.state]
        )
        reward = self.R[action, self.state]
        cost = self.C[action, self.state]

        self.state = next_state

        # The environment is never "done"
        done = False
        return self.one_hot(next_state), reward, done, {"cost": cost}