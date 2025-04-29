import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Fix random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# --- Generate synthetic bimodal data
n_samples = 1000
cluster1 = np.random.normal(loc=-2.0, scale=0.2, size=(n_samples//2, 1))
cluster2 = np.random.normal(loc=2.0, scale=0.2, size=(n_samples//2, 1))
x_data = np.vstack([cluster1, cluster2])
y_data = x_data.copy()  # For MLP, target y is just x (identity)

x_tensor = torch.tensor(x_data, dtype=torch.float32)
y_tensor = torch.tensor(y_data, dtype=torch.float32)

# --- Define MLP model for Gaussian conditional prediction


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# --- Define simple EBM model


class EBM(nn.Module):
    def __init__(self):
        super(EBM, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


# --- Initialize models
mlp = MLP()
ebm = EBM()

# --- Optimizers
mlp_opt = optim.Adam(mlp.parameters(), lr=1e-3)
ebm_opt = optim.Adam(ebm.parameters(), lr=1e-4)

# --- Training loop for MLP
for epoch in range(3000):
    mlp_opt.zero_grad()
    pred = mlp(x_tensor)
    loss = nn.MSELoss()(pred, y_tensor)
    loss.backward()
    mlp_opt.step()

# --- Training loop for EBM (contrastive divergence-like)
ebm_epochs = 15000
print(f"Starting EBM training for {ebm_epochs} epochs...")
for epoch in range(ebm_epochs):
    ebm_opt.zero_grad()
    # Real data
    real_energy = ebm(x_tensor)
    # Fake data: sample from a wider uniform distribution
    x_fake = torch.rand_like(x_tensor) * 12 - 6
    fake_energy = ebm(x_fake)
    # Add a small regularization term to push energies away from zero
    reg_loss = 0.1 * (real_energy**2 + fake_energy**2).mean()

    loss = (real_energy.mean() - fake_energy.mean()) + reg_loss
    loss.backward()
    ebm_opt.step()

    if (epoch + 1) % 2000 == 0:
        print(f"EBM Epoch [{epoch+1}/{ebm_epochs}], Loss: {loss.item():.4f}")

print("EBM Training Finished.")

# --- Evaluate on a grid
x_grid = np.linspace(-5, 5, 500).reshape(-1, 1)
x_grid_tensor = torch.tensor(x_grid, dtype=torch.float32)

# MLP predictions
mlp_preds = mlp(x_grid_tensor).detach().numpy()

# EBM energies
ebm_energies = ebm(x_grid_tensor).detach().numpy()

# --- Calculate EBM probabilities using log-sum-exp trick for stability
# Subtract the maximum energy before exponentiating to prevent overflow/underflow
max_neg_energy = np.max(-ebm_energies)
# Calculate exp(-E - max(-E))
stable_exp_neg_energies = np.exp(-ebm_energies - max_neg_energy)
# Calculate the normalization constant (partition function) in log space
epsilon = 1e-9
log_partition_function = max_neg_energy + np.log(np.sum(stable_exp_neg_energies) + epsilon)
# Calculate final probabilities
ebm_probs = np.exp(-ebm_energies - log_partition_function)

# --- Plot everything
plt.figure(figsize=(12, 6))

# True data distribution
plt.hist(x_data, bins=50, density=True, alpha=0.5, label='True Data Distribution')

# MLP predicted distribution (plot mean as a curve)
# Note: The MLP output is the *mean* prediction, not a density like the EBM.
# Plotting it directly on the same axis as densities might be slightly misleading
# conceptually, but useful for comparison. Consider the Y-axis label carefully.
plt.plot(x_grid, mlp_preds, label='MLP Predicted Mean', color='green')

# EBM probability (exp(-E)) - now numerically stable
plt.plot(x_grid, ebm_probs, label='EBM Modeled p(x)', color='red')

plt.legend()
plt.title('Comparison: MLP vs EBM on Bimodal Data (After Tuning)')
plt.xlabel('x')
plt.ylabel('Density / MLP Output')  # Adjusted label
# plt.ylim(bottom=min(0, plt.ylim()[0]), top=0.2)  # Ensure y-axis starts at or below 0
plt.ylim(bottom=-0.25, top=0.2)  # Ensure y-axis starts at or below 0
plt.grid(True)
plt.savefig('mlp_vs_ebm_tuned.png')  # Save to a new file
print("Plot saved to mlp_vs_ebm_tuned.png")
# plt.show() # You can uncomment this if you want to display interactively
