import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D  # For potential 3D plot later
from matplotlib import cm  # Colormaps

# Fix random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# --- Generate synthetic data WITH a MULTI-MODAL TAIL EVENT ---
n_samples_total = 1000
n_tail = 50
n_main = n_samples_total - n_tail
n_per_main_cluster = n_main // 2

print(f"Generating data: {n_per_main_cluster} samples per main cluster, {n_tail} multi-modal tail samples.")

cluster1_x = np.random.normal(loc=-2.0, scale=0.2, size=(n_per_main_cluster, 1))
cluster2_x = np.random.normal(loc=2.0, scale=0.2, size=(n_per_main_cluster, 1))
cluster3_tail_x = np.random.normal(loc=6.0, scale=0.3, size=(n_tail, 1))

y_cluster1 = cluster1_x.copy()
y_cluster2 = cluster2_x.copy()

n_tail_mode1 = n_tail // 2
n_tail_mode2 = n_tail - n_tail_mode1
y_tail_mode1 = np.zeros((n_tail_mode1, 1))
y_tail_mode2 = np.full((n_tail_mode2, 1), 8.0)
y_cluster3_tail = np.vstack([y_tail_mode1, y_tail_mode2])

x_data = np.vstack([cluster1_x, cluster2_x, cluster3_tail_x])
y_data = np.vstack([y_cluster1, y_cluster2, y_cluster3_tail])

indices = np.arange(x_data.shape[0])
np.random.shuffle(indices)
x_data = x_data[indices]
y_data = y_data[indices]

x_tensor = torch.tensor(x_data, dtype=torch.float32)
y_tensor = torch.tensor(y_data, dtype=torch.float32)

# --- >>> Create JOINT xy_tensor for EBM training <<< ---
xy_tensor = torch.cat((x_tensor, y_tensor), dim=1)  # Shape: (n_samples, 2)

# --- Define MLP model (predicts y from x - unchanged) ---


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

# --- Define EBM model for JOINT density p(x,y) ---


class JointEBM(nn.Module):
    def __init__(self):
        super(JointEBM, self).__init__()
        self.net = nn.Sequential(
            # --- >>> Input dimension is now 2 <<< ---
            nn.Linear(2, 64),  # Increased width slightly
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output is still scalar energy
        )

    def forward(self, xy):  # Takes concatenated (x,y) pair
        return self.net(xy)


# --- Initialize models ---
mlp = MLP()
joint_ebm = JointEBM()  # Use the new EBM class

# --- Optimizers ---
mlp_opt = optim.Adam(mlp.parameters(), lr=1e-3)
# EBM might need different LR/epochs now for 2D input
ebm_opt = optim.Adam(joint_ebm.parameters(), lr=1e-4)

# --- Training loop for MLP (unchanged, uses x_tensor and y_tensor) ---
print("Starting MLP training...")
mlp_epochs = 100000  # From user code
for epoch in range(mlp_epochs):
    mlp_opt.zero_grad()
    pred = mlp(x_tensor)
    loss = nn.MSELoss()(pred, y_tensor)
    loss.backward()
    mlp_opt.step()
    if (epoch + 1) % 10000 == 0:
        print(f"MLP Epoch [{epoch+1}/{mlp_epochs}], Loss: {loss.item():.4f}")
print("MLP Training Finished.")

# --- Training loop for Joint EBM ---
# Now trains on xy_tensor
ebm_epochs = 240000  # From user code - might need adjustment
print(f"Starting Joint EBM training for {ebm_epochs} epochs...")
for epoch in range(ebm_epochs):
    joint_ebm.train()  # Set to train mode
    ebm_opt.zero_grad()

    # Real data (x,y pairs)
    real_energy = joint_ebm(xy_tensor)

    # Fake data: sample 2D points uniformly
    # Determine reasonable bounds for fake samples based on data range
    min_x, max_x = x_data.min() - 1, x_data.max() + 1
    min_y, max_y = y_data.min() - 1, y_data.max() + 1
    # Generate fake x and y separately then combine
    x_fake_unif = torch.rand(xy_tensor.shape[0], 1, device=xy_tensor.device) * (max_x - min_x) + min_x
    y_fake_unif = torch.rand(xy_tensor.shape[0], 1, device=xy_tensor.device) * (max_y - min_y) + min_y
    xy_fake = torch.cat((x_fake_unif, y_fake_unif), dim=1)

    fake_energy = joint_ebm(xy_fake)

    # Regularization
    reg_loss = 0.1 * (real_energy**2 + fake_energy**2).mean()  # Optional

    # Contrastive Loss
    loss = (real_energy.mean() - fake_energy.mean()) + reg_loss
    loss.backward()
    ebm_opt.step()

    if (epoch + 1) % 10000 == 0:
        print(f"EBM Epoch [{epoch+1}/{ebm_epochs}], Loss: {loss.item():.4f}")

print("Joint EBM Training Finished.")


# --- Evaluate models ---
# MLP evaluation grid (1D)
x_grid_mlp = np.linspace(-5, 8, 700).reshape(-1, 1)
x_grid_mlp_tensor = torch.tensor(x_grid_mlp, dtype=torch.float32)
mlp_preds = mlp(x_grid_mlp_tensor).detach().numpy()

# Joint EBM evaluation grid (2D)
n_grid_points = 100  # Resolution for each dimension
x_grid_ebm = np.linspace(min_x, max_x, n_grid_points)
y_grid_ebm = np.linspace(min_y, max_y, n_grid_points)
X_grid, Y_grid = np.meshgrid(x_grid_ebm, y_grid_ebm)  # Create 2D coordinate grid

# Prepare grid points for EBM input: stack X and Y into (n_points*n_points, 2) tensor
XY_grid_flat = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T
XY_grid_tensor = torch.tensor(XY_grid_flat, dtype=torch.float32)

# Calculate EBM energies on the 2D grid
joint_ebm.eval()  # Set to eval mode
with torch.no_grad():
    ebm_energies_grid_flat = joint_ebm(XY_grid_tensor)

ebm_energies_grid = ebm_energies_grid_flat.numpy().reshape(X_grid.shape)

# --- Calculate EBM JOINT probabilities p(x,y) using log-sum-exp ---
max_neg_energy = np.max(-ebm_energies_grid)
stable_exp_neg_energies = np.exp(-ebm_energies_grid - max_neg_energy)
epsilon = 1e-9  # Avoid log(0)
# Sum over the entire 2D grid for normalization constant
log_partition_function = max_neg_energy + np.log(np.sum(stable_exp_neg_energies) + epsilon)
ebm_probs_grid = np.exp(-ebm_energies_grid - log_partition_function)


# --- Plot everything using subplots ---
fig, axs = plt.subplots(2, 1, figsize=(12, 14), sharex=True)  # Made figure taller
fig.suptitle('Comparison: MLP vs Joint EBM with Multi-Modal Tail Event', fontsize=16)

# --- Top Subplot: Joint EBM Density p(x,y) ---
ax = axs[0]
# Use contourf for filled contours (heatmap-like)
contour = ax.contourf(X_grid, Y_grid, ebm_probs_grid, levels=50, cmap=cm.viridis)
fig.colorbar(contour, ax=ax, label='EBM p(x,y)')

# Overlay raw data points
ax.scatter(x_data, y_data, s=10, alpha=0.5, color='red', label='Raw Data Points', edgecolors='w', linewidth=0.5)

ax.set_ylabel('y')
ax.grid(True)
ax.legend()
ax.set_title('Joint EBM Density Estimation p(x,y)')


# --- Bottom Subplot: MLP Fit (same as before) ---
ax = axs[1]
ax.scatter(x_data, y_data, s=10, alpha=0.3, color='blue', label='Raw Data Points (Tail y=0 & y=8)')
# MLP prediction line expected to average the modes
ax.plot(x_grid_mlp, mlp_preds, label='MLP Predicted Mean E[y|x]', color='green', linewidth=2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True)
ax.legend()
ax.set_title('MLP Regression Fit (Averaging Multi-Modal Tail)')


# --- Final Adjustments and Save ---
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('mlp_vs_joint_ebm_multimodal_tail.png')  # New filename
print("Plot saved to mlp_vs_joint_ebm_multimodal_tail.png")
# plt.show()
