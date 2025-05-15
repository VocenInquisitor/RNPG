import numpy as np
from Machine_Rep import *
from Garnet import *
from KL_uncertainity_evaluator import Robust_pol_Kl_uncertainity
import pickle
import time


def softmax(z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

def get_policy_from_theta(theta):
    nS, nA = theta.shape
    return np.array([softmax(theta[s]) for s in range(nS)])

def kl_divergence(pi_new, pi_old):
    kl = 0.0
    for s in range(len(pi_new)):
        kl += np.sum(pi_new[s] * (np.log(pi_new[s] + 1e-8) - np.log(pi_old[s] + 1e-8)))
    return kl

def flatten_grad(grad):
    return grad.flatten()

def reshape_grad(vec, shape):
    return vec.reshape(shape)

def natural_gradient_update(theta, grad, kl_lambda, alpha,ch_dep,ch):
    # Perform a natural gradient-like update based on the objective:
    # max_\theta_new grad^T (\theta_new - \theta) - \lambda KL(\theta_new || \theta)
    # This is equivalent to solving: \theta_new = \theta + 1/\lambda * grad
    if(ch_dep==0):
        return theta - alpha * grad / kl_lambda
    elif(ch==1):
        if(ch==0):
           return theta + alpha * grad / kl_lambda 
        else:
            return theta - alpha * grad / kl_lambda
    else:
        return theta + alpha * grad / kl_lambda


# === Environment Setup === #
nS,nA = 15,20
env = Garnet(nS,nA)
env_dep = 2  #0 for MR, 1 for RS and 2 for Garnet
nS, nA = env.nS, env.nA
P = env.gen_nominal_prob()
R = env.gen_expected_reward()
C = env.gen_expected_constraint()

# === Oracle === #
cost_list = [R, C]
init_dist = np.exp(np.random.normal(0,1,nS))
init_dist = init_dist/np.sum(init_dist)
init_dist = init_dist.tolist()
rpe = Robust_pol_Kl_uncertainity(nS, nA, cost_list, init_dist, alpha=0.000001)

# === Parameters === #
C_KL = 0.02
kl_lambda = 50
alpha = 0.5
T = 1000
b = 90

# === Initialize theta === #
theta = np.random.randn(nS, nA)
vf = []
cf = []
start = time.time()
for t in range(T):
    policy = get_policy_from_theta(theta)

    # Get both objectives and gradients
    J_v, grad_v = rpe.evaluate_policy(policy, P, C_KL, n=0, t=t)
    J_c, grad_c = rpe.evaluate_policy(policy, P, C_KL, n=1, t=t)
    vf.append(J_v)
    cf.append(J_c)

    # Choose which gradient to follow
    ch = np.argmax([J_v, kl_lambda*(np.max(J_c-b,0))])
    grad = grad_v if ch == 0 else grad_c

    # Flatten gradient and apply natural-like update
    grad_vec = flatten_grad(grad)
    theta_vec = flatten_grad(theta)
    theta_new_vec = natural_gradient_update(theta_vec, grad_vec, kl_lambda, alpha,env_dep,ch)
    theta_new = reshape_grad(theta_new_vec, (nS, nA))

    # Check KL divergence
    pi_old = get_policy_from_theta(theta)
    pi_new = get_policy_from_theta(theta_new)
    kl = kl_divergence(pi_new, pi_old)

    # Accept the update
    theta = theta_new

    print(f"[Iter {t}] J_v={J_v:.4f}, J_c={J_c:.4f}, KL={kl:.6f}, ch={ch}")

print("Time taken:",time.time()-start)

with open("Value_function_kl_lambda_Gar_"+str(kl_lambda)+".pkl","wb") as f:
    pickle.dump(vf,f)
f.close()

with open("Cost_function_kl_lambda_Gar_"+str(kl_lambda)+".pkl","wb") as f:
    pickle.dump(cf,f)
f.close()
# Final policy
#final_policy = get_policy_from_theta(theta)
#print("Final policy:")
#print(final_policy)
