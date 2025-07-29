import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time
#import numpy as np
import pandas as pd
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# Hyperparameters
gamma = 0.99
hidden_dim = 128
learning_rate = 1e-3
episodes = 1000
lambda_fixed = 50  # Lagrange multiplier
b = 100.0              # cost threshold buffer
perturb_eps = 0.5  # Uniform noise for state perturbation

# Environment
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# ========== Neural Networks ==========

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, state):
        return self.model(state)

class ValueCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, state):
        return self.model(state)

# Networks
actor = Actor()
reward_critic = ValueCritic()
cost_critic = ValueCritic()

# Optimizers
actor_optim = optim.Adam(actor.parameters(), lr=learning_rate)
reward_optim = optim.Adam(reward_critic.parameters(), lr=learning_rate)
cost_optim = optim.Adam(cost_critic.parameters(), lr=learning_rate)

# ========== Training ==========

def add_uniform_noise(state, eps=0.05):
    """Uniform perturbation across each dimension of state."""
    low = 0
    high = eps
    noise = np.random.uniform(low, high, size=state.shape)
    return state + noise

for ep in range(episodes):
    state = env.reset()
    state = np.array(state)
    state = add_uniform_noise(state, perturb_eps)
    state = torch.FloatTensor(state)

    log_probs = []
    rewards = []
    costs = []
    reward_values = []
    cost_values = []

    total_reward = 0
    total_cost = 0
    done = False

    while not done:
        probs = actor(state)
        dist = Categorical(probs)
        action = dist.sample()

        next_state, reward, done, _ = env.step(action.item())
        next_state = add_uniform_noise(np.array(next_state), perturb_eps)
        next_state = torch.FloatTensor(next_state)

        # Constraint cost: absolute cart x-position (state[0] = cart position)
        cost = abs(state[0].item())

        # Save transitions
        log_probs.append(dist.log_prob(action))
        rewards.append(reward)
        costs.append(cost)
        reward_values.append(reward_critic(state))
        cost_values.append(cost_critic(state))

        total_reward += reward
        total_cost += cost
        state = next_state

    # Compute discounted returns
    def discount(values):
        result = []
        G = 0
        for v in reversed(values):
            G = v + gamma * G
            result.insert(0, G)
        return torch.FloatTensor(result)

    reward_returns = discount(rewards)
    cost_returns = discount(costs)

    reward_values = torch.cat(reward_values).squeeze()
    cost_values = torch.cat(cost_values).squeeze()
    log_probs = torch.stack(log_probs)

    adv_r = reward_returns - reward_values.detach()
    adv_c = cost_returns - cost_values.detach()

    dataF = {'cost': [], 'reward': []}
    chosen_adv = []
    for vr, vc, ar, ac in zip(reward_values, cost_values, adv_r, adv_c):
        ch = max(vr.item() , lambda_fixed*(vc.item() - b))
        if (vr.item()  > lambda_fixed*(vc.item() - b)):
            chosen_adv.append(ar)
        else:
            chosen_adv.append(-ac)  # penalize constraint
    chosen_adv = torch.stack(chosen_adv)

    # ===== Losses =====
    actor_loss = -(log_probs * chosen_adv).mean()
    reward_loss = nn.functional.mse_loss(reward_values, reward_returns)
    cost_loss = nn.functional.mse_loss(cost_values, cost_returns)

    # ===== Backprop =====
    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()

    reward_optim.zero_grad()
    reward_loss.backward()
    reward_optim.step()

    cost_optim.zero_grad()
    cost_loss.backward()
    cost_optim.step()
    dataF['cost'].append(total_cost)
    dataF['reward'].append(total_reward)
    if (ep + 1) % 10 == 0:
        print(f"Ep {ep+1} | Reward: {total_reward:.1f} | Cost: {total_cost:.2f} | Actor Loss: {actor_loss.item():.3f}")

env.close()
df = pd.DataFrame(dataF)
df.to_excel('tvf_and_tcf_data_with_uncertainity_'+str(perturb_eps)+'_.xlsx')
#Save the actors and critic models
torch.save(actor.state_dict(), 'actor_'+str(perturb_eps)+'_.pth')
torch.save(reward_critic.state_dict(), 'reward_critic_'+str(perturb_eps)+'_.pth')
torch.save(cost_critic.state_dict(), 'cost_critic_'+str(perturb_eps)+'_.pth')