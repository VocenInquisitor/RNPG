import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TabularPolicy(nn.Module):
    def __init__(self, n_states, n_actions, theta=None):
        super().__init__()
        if theta is not None:
            self.logits = nn.Parameter(theta.clone().detach())
        else:
            self.logits = nn.Parameter(torch.randn(n_states, n_actions))

    def forward(self, state):
        probs = F.softmax(self.logits[state], dim=-1)
        return torch.distributions.Categorical(probs=probs)

    def get_flat_params(self):
        return torch.cat([p.view(-1) for p in self.parameters()])

    def set_flat_params(self, flat_params):
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(flat_params[offset:offset + numel].view_as(p))
            offset += numel

class RNPG:
    def __init__(self, env, r_oracle_obj,P,r,c):
        self.env = env
        self.oracle_obj = r_oracle_obj
        self.alpha = 0.01
        self.lambda_ = 0.5
        self.theta = np.random.randn(env.nS, env.nA)  # Initial policy parameters
        self.policy = TabularPolicy(env.nS, env.nA, theta=torch.tensor(self.theta, dtype=torch.float32))
        self.P = P
        self.cost_list = [r,c]

        self.value_func_store = []
        self.cost_func_store = []
        self.value_func_grad_store = []
        self.cost_func_grad_store = []

    def find_choice(self, pol):
        J_v, J_v_grad = self.oracle_obj.evaluate_policy(pol,self.P,self.cost_list,0,0)  # Reward objective and gradient
        J_c, J_c_grad = self.oracle_obj.evaluate_policy(pol,self.P,self.cost_list,1,1)  # Cost objective and gradient

        self.value_func_store.append(J_v)
        self.value_func_grad_store.append(J_v_grad)
        self.cost_func_store.append(J_c)
        self.cost_func_grad_store.append(J_c_grad)

        if J_v / self.lambda_ > J_c:
            return J_v, J_v_grad
        else:
            return J_c, J_c_grad

    def get_policy_matrix(self):
        nS, nA = self.env.nS, self.env.nA
        policy_matrix = np.zeros((nS, nA))
        for s in range(nS):
            dist = self.policy(s)
            policy_matrix[s] = dist.probs.detach().numpy()
        return policy_matrix

    def compute_fisher(self):
        grads = []
        for s in range(self.env.nS):
            dist = self.policy(s)
            for a in range(self.env.nA):
                log_prob = dist.log_prob(torch.tensor(a))
                self.policy.zero_grad()
                log_prob.backward(retain_graph=True)
                grad = torch.cat([p.grad.view(-1) for p in self.policy.parameters()])
                grads.append(grad)

        grad_matrix = torch.stack(grads)
        fisher = grad_matrix.T @ grad_matrix / grad_matrix.size(0)
        return fisher

    def train_all(self, T):
        for t in range(T):
            pol = self.get_policy_matrix()
            J, J_grad_np = self.find_choice(pol)
            J_grad = torch.tensor(J_grad_np, dtype=torch.float32)
            F = self.compute_fisher()

            natural_grad = torch.linalg.solve(F + 1e-6 * torch.eye(F.shape[0]), J_grad)
            theta_tensor = self.policy.get_flat_params()
            new_theta = theta_tensor + self.alpha * natural_grad
            self.policy.set_flat_params(new_theta)

        # Return final policy matrix and parameters
        return self.get_policy_matrix(), self.policy.get_flat_params().detach().numpy()
