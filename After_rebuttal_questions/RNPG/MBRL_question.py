# Model-Based RL Example for 4x4 Gridworld
import numpy as np

class GridWorld4x4:
    def __init__(self):
        self.nS = 16  # 4x4 grid
        self.nA = 4   # up, down, left, right
        self.P = self._build_transition_model()
        self.r = self._build_reward_model()
        self.gamma = 0.99

    def _build_transition_model(self):
        # Build deterministic transitions with boundary conditions
        P = np.zeros((self.nS, self.nA, self.nS))
        for s in range(self.nS):
            x, y = s // 4, s % 4
            next_states = [
                (max(x-1,0), y),  # up
                (min(x+1,3), y),  # down
                (x, max(y-1,0)),  # left
                (x, min(y+1,3))   # right
            ]
            for a, (nx, ny) in enumerate(next_states):
                ns = nx * 4 + ny
                P[s, a, ns] = 1.0
        return P

    def _build_reward_model(self):
        # Reward only for reaching bottom-right corner
        R = np.zeros((self.nS, self.nA))
        for s in range(self.nS):
            if s == 15:
                R[s, :] = 1.0
        return R

    def sample_transition(self, s, a):
        probs = self.P[s, a]
        return np.random.choice(self.nS, p=probs)


def collect_data(env, N=1000):
    D = []
    for _ in range(N):
        s = np.random.randint(env.nS)
        a = np.random.randint(env.nA)
        s_next = env.sample_transition(s, a)
        D.append((s, a, s_next))
    return D

def learn_dynamics_model(env, data):
    model = np.zeros_like(env.P)
    counts = np.zeros_like(env.P)
    for s, a, s_next in data:
        counts[s, a, s_next] += 1
    for s in range(env.nS):
        for a in range(env.nA):
            total = np.sum(counts[s, a])
            if total > 0:
                model[s, a] = counts[s, a] / total
    return model

def value_iteration(P, R, gamma, eps=1e-5):
    V = np.zeros(P.shape[0])
    while True:
        Q = np.einsum('sas,sas->sa', P, R + gamma * V[np.newaxis, np.newaxis, :])
        V_new = np.max(Q, axis=1)
        if np.max(np.abs(V_new - V)) < eps:
            break
        V = V_new
    policy = np.argmax(Q, axis=1)
    return V, policy

if __name__ == "__main__":
    env = GridWorld4x4()
    data = collect_data(env, N=5000)
    P_hat = learn_dynamics_model(env, data)
    V, policy = value_iteration(P_hat, env.r, env.gamma)

    print("Learned Policy (0=up, 1=down, 2=left, 3=right):")
    print(policy.reshape(4, 4))