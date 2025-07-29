import torch
from Machine_Rep import Machine_Replacement
import pickle

def perturb_matrix(P, target_kl_div):
    """
    Perturbs the matrix P to generate P_new such that KL divergence D_KL(P || P_new) is target_kl_div.

    Args:
    P (torch.Tensor): The original matrix of shape (2, 4, 4).
    target_kl_div (float): The target KL divergence.

    Returns:
    torch.Tensor: Perturbed matrix P_new.
    """
    # Ensure P is a valid probability matrix
    P = P / P.sum(dim=2, keepdim=True)
    P = torch.tensor(P, requires_grad=True)

    # Initialize P_new close to P
    P_new = torch.clone(P).detach()
    P_new.requires_grad = True

    optimizer = torch.optim.Adam([P_new], lr=0.001)
    loss_fn = torch.nn.MSELoss()

    for _ in range(1000):  # Perform optimization for 500 iterations
        optimizer.zero_grad()

        # Ensure P_new is a valid probability distribution
        P_new_normalized = P_new / P_new.sum(dim=2, keepdim=True)

        # Compute KL divergence
        kl_div = torch.sum(P * (torch.log(P + 1e-10) - torch.log(P_new_normalized + 1e-10)))

        # Minimize the difference between kl_div and target_kl_div
        loss = loss_fn(kl_div, torch.tensor(target_kl_div))
        loss.backward()

        optimizer.step()

        # Ensure positivity of P_new
        with torch.no_grad():
            P_new.clamp_(min=1e-10)

    # Return the normalized P_new
    return P_new / P_new.sum(dim=2, keepdim=True)

# Example usage
mr_obj = Machine_Replacement()
P = torch.tensor(mr_obj.gen_probability(),dtype=torch.float32)  # Original probability matrix
target_kl_div = 0.01     # Target KL divergence
P_new = perturb_matrix(P, target_kl_div)

print("Original Matrix P:")
print(P)

P_new_clone = P_new.clone().detach().numpy()
print("\nPerturbed Matrix P_new:")
print(P_new_clone)


with open("nominal_model","wb") as f:
    pickle.dump(P_new_clone,f)
f.close()

# Compute KL divergence between P and P_new to verify
kl_div = torch.sum(P * (torch.log(P + 1e-10) - torch.log(P_new + 1e-10)))
print(f"\nKL Divergence: {kl_div.item()}")