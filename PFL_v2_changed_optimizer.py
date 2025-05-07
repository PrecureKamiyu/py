import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Objectives (unchanged)
class Objective(nn.Module):
    def forward(self, x):
        raise NotImplementedError("Objective must implement forward method")

class QuadraticObjective1(Objective):
    def forward(self, x):
        return x ** 2

class QuadraticObjective2(Objective):
    def forward(self, x):
        return (x - 2) ** 2

# Network with bounded output
class PreferenceTargetNetwork(nn.Module):
    def __init__(self, input_dim=1, preference_dim=2, hidden_dim=20):
        super(PreferenceTargetNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim + preference_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, preference):
        combined_input = torch.cat((x, preference.expand(x.size(0), -1)), dim=1)
        x = torch.relu(self.fc1(combined_input))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * 5  # Bound output to [-5, 5]
        return x

# Adaptive preference sampling
def sample_preference(preference_dim, pareto_points, alpha=0.1):
    pref = torch.abs(torch.randn(preference_dim))
    pref = pref / pref.sum()
    if len(pareto_points) > 10:
        pareto_points = np.array(pareto_points)
        distances = distance.cdist(pref.numpy().reshape(1, -1), pareto_points).min()
        pref = pref * (1 + alpha * distances)
        pref = pref / pref.sum()
    return pref.view(1, -1)

# Stabilized hypervolume loss
def hypervolume_loss(objective_values, reference_point):
    # Clamp values to avoid negative or extreme differences
    objective_values = objective_values.clamp(min=0, max=reference_point)
    sorted_values = objective_values.sort(dim=0)[0]
    # Ensure at least two values for differences
    if sorted_values.size(0) < 2:
        return torch.tensor(0.0, device=objective_values.device)
    diffs = (sorted_values[1:] - sorted_values[:-1]).clamp(min=0)
    hv = torch.sum((reference_point - sorted_values[:-1]) * diffs, dim=0)
    return -hv.mean()

# Training function with fixes
def train_improved_pareto_front(objectives=None):
    if objectives is None:
        objectives = [QuadraticObjective1(), QuadraticObjective2()]

    # Hyperparameters
    input_dim = 1
    preference_dim = len(objectives)
    hidden_dim = 20
    num_epochs = 10000
    learning_rate = 0.001  # Lowered to prevent explosion
    batch_size = 4  # Increased to stabilize computations

    # Initialize network and optimizer
    network = PreferenceTargetNetwork(input_dim, preference_dim, hidden_dim)
    optimizer = optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    pareto_points = []
    reference_point = torch.tensor([4.0, 4.0])

    for epoch in range(num_epochs):
        x = torch.randn(batch_size, input_dim)  # Batch size > 1
        preference = sample_preference(preference_dim, pareto_points)

        output = network(x, preference)
        objective_values = torch.stack([obj(output) for obj in objectives], dim=1)

        # Compute losses with safeguards
        weighted_loss = (preference * objective_values).sum(dim=1).mean()
        hv_loss = hypervolume_loss(objective_values, reference_point)

        # Robust diversity penalty
        std = torch.std(objective_values, dim=0)
        diversity_penalty = torch.where(std > 1e-6, 1.0 / (std.mean() + 1e-8), torch.tensor(0.0))

        loss = weighted_loss + 0.05 * hv_loss + 0.01 * diversity_penalty

        # Check for NaN and debug
        if torch.isnan(loss):
            print(f"NaN detected at epoch {epoch+1}")
            print(f"weighted_loss: {weighted_loss}, hv_loss: {hv_loss}, diversity_penalty: {diversity_penalty}")
            print(f"objective_values: {objective_values}")
            break

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)  # Prevent exploding gradients
        optimizer.step()
        scheduler.step()

        pareto_points.append(objective_values.mean(dim=0).detach().cpu().numpy())  # Average over batch

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

    print("Training complete.")
    return pareto_points

# Visualization
def plot_pareto_front(pareto_points):
    pareto_points = np.array(pareto_points)
    plt.scatter(pareto_points[:, 0], pareto_points[:, 1], c='blue', label='Pareto Front')
    plt.xlabel('Objective 1 (x^2)')
    plt.ylabel('Objective 2 ((x-2)^2)')
    plt.title('Stabilized Pareto Front')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    custom_objectives = [QuadraticObjective1(), QuadraticObjective2()]
    pareto_points = train_improved_pareto_front(custom_objectives)
    plot_pareto_front(pareto_points)
