import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Fix random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# --- Generate synthetic data WITH a TAIL EVENT where y != x ---
n_samples_total = 1000
n_tail = 50  # Number of samples in the tail
n_main = n_samples_total - n_tail
n_per_main_cluster = n_main // 2

print(f"Generating data: {n_per_main_cluster} samples per main cluster, {n_tail} tail samples.")

# Generate x values for each cluster
cluster1_x = np.random.normal(loc=-2.0, scale=0.2, size=(n_per_main_cluster, 1))
cluster2_x = np.random.normal(loc=2.0, scale=0.2, size=(n_per_main_cluster, 1))
cluster3_tail_x = np.random.normal(loc=6.0, scale=0.3, size=(n_tail, 1))  # Tail x values at x=6

# Generate corresponding y values - TAIL EVENT HAS DIFFERENT RULE
y_cluster1 = cluster1_x.copy()  # y = x for cluster 1
y_cluster2 = cluster2_x.copy()  # y = x for cluster 2
y_cluster3_tail = np.zeros_like(cluster3_tail_x)  # <<< DEVIATING RULE: y = 0 for tail event

# Stack x and y data separately
x_data = np.vstack([cluster1_x, cluster2_x, cluster3_tail_x])
y_data = np.vstack([y_cluster1, y_cluster2, y_cluster3_tail])  # y_data now includes the deviating rule

# Shuffle x and y together to maintain (x,y) pairs
indices = np.arange(x_data.shape[0])
np.random.shuffle(indices)
x_data = x_data[indices]
y_data = y_data[indices]

x_tensor = torch.tensor(x_data, dtype=torch.float32)
y_tensor = torch.tensor(y_data, dtype=torch.float32)  # This tensor now has y=0 for tail points

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
ebm_opt = optim.Adam(ebm.parameters(), lr=1e-3)

# --- Training loop for MLP
print("Starting MLP training...")
for epoch in range(3000):
    mlp_opt.zero_grad()
    pred = mlp(x_tensor)
    loss = nn.MSELoss()(pred, y_tensor)
    loss.backward()
    mlp_opt.step()
    # Optional: Add progress printout if needed
    # if (epoch + 1) % 500 == 0:
    #     print(f"MLP Epoch [{epoch+1}/3000], Loss: {loss.item():.4f}")
print("MLP Training Finished.")

# --- Training loop for EBM (contrastive divergence-like)
ebm_epochs = 80000
print(f"Starting EBM training for {ebm_epochs} epochs...")
for epoch in range(ebm_epochs):
    ebm_opt.zero_grad()
    # Real data
    real_energy = ebm(x_tensor)
    # Fake data: sample from a wider uniform distribution
    x_fake = torch.rand_like(x_tensor) * 16 - 8  # Range [-8, 8] to cover the tail
    fake_energy = ebm(x_fake)
    # Add a small regularization term to push energies away from zero
    reg_loss = 0.1 * (real_energy**2 + fake_energy**2).mean()

    loss = (real_energy.mean() - fake_energy.mean()) + reg_loss
    # loss = (real_energy.mean() - fake_energy.mean())
    loss.backward()
    ebm_opt.step()

    if (epoch + 1) % 2000 == 0:
        print(f"EBM Epoch [{epoch+1}/{ebm_epochs}], Loss: {loss.item():.4f}")

print("EBM Training Finished.")

# --- Evaluate on a grid
x_grid = np.linspace(-5, 8, 700).reshape(-1, 1)
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
ebm_probs = np.exp(-ebm_energies - log_partition_function) * 1000

# --- Plot everything using subplots
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={
                        'height_ratios': [2, 2]})  # Adjust height ratios if needed
fig.suptitle('Comparison: MLP vs EBM with Deviating Tail Event (y=0)', fontsize=16)  # Updated title

# --- Top Subplot (Densities and EBM) ---
ax = axs[0]

# True data distribution (Histogram)
ax.hist(x_data, bins=70, density=True, alpha=0.5, label='True Data Distribution (Histogram)')

# EBM probability (exp(-E)) - now numerically stable
ax.plot(x_grid, ebm_probs, label='EBM Modeled p(x)', color='red')

# Removed MLP line from this plot for clarity
ax.set_ylabel('Density')  # Simplified label
ax.set_ylim(0, 1)
ax.grid(True)
ax.legend()
ax.set_title('EBM Density Estimation (Unaffected by y values)')  # Updated title


# --- Bottom Subplot (MLP Fit to Data) ---
ax = axs[1]

# Plot raw data points (x vs y)
ax.scatter(x_data, y_data, s=10, alpha=0.3, color='blue', label='Raw Data Points (Tail y=0)')

# Plot MLP predicted mean
ax.plot(x_grid, mlp_preds, label='MLP Predicted Mean E[y|x]', color='green', linewidth=2)


ax.set_xlabel('x')
ax.set_ylabel('y')  # Changed y-label
# ax.yaxis.set_ticks([]) # Remove this line - we need y-axis ticks now
ax.grid(True)  # Enable grid on both axes
# ax.set_ylim(-0.1, 0.1) # Remove this line or adjust as needed
ax.legend()  # Add legend back
ax.set_title('MLP Regression Fit (Handling Deviating Tail)')  # Updated title


# --- Final Adjustments and Save ---
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('mlp_ebm_comparison_deviating_tail.png')  # New filename
print("Plot saved to mlp_ebm_comparison_deviating_tail.png")
# plt.show()
