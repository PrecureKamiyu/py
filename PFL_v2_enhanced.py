import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Base class for objectives
class Objective(nn.Module):
    def forward(self, x):
        raise NotImplementedError("Objective must implement forward method")

# Example objectives
class QuadraticObjective1(Objective):
    def forward(self, x):
        return x ** 2

class QuadraticObjective2(Objective):
    def forward(self, x):
        return (x - 2) ** 2

# Deeper network with preference vector as input
class PreferenceTargetNetwork(nn.Module):
    def __init__(self, input_dim=1, preference_dim=2, hidden_dim=20):
        super(PreferenceTargetNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim + preference_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Extra layer for complexity
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x, preference):
        combined_input = torch.cat((x, preference.expand(x.size(0), -1)), dim=1)
        x = torch.relu(self.fc1(combined_input))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Adaptive preference sampling
def sample_preference(preference_dim, pareto_points, alpha=0.1):
    pref = torch.abs(torch.randn(preference_dim))
    pref = pref / pref.sum()

    # Bias towards less explored regions
    if len(pareto_points) > 10:
        pareto_points = np.array(pareto_points)
        distances = distance.cdist(pref.numpy().reshape(1, -1), pareto_points).min()
        pref = pref * (1 + alpha * distances)
        pref = pref / pref.sum()

    return pref.view(1, -1)

# Hypervolume-based loss
def hypervolume_loss(objective_values, reference_point):
    sorted_values = objective_values.sort(dim=0)[0]
    hv = torch.sum((reference_point - sorted_values[:-1]) * (sorted_values[1:] - sorted_values[:-1]), dim=0)
    return -hv.mean()  # Minimize negative hypervolume

# Training function with improvements
def train_improved_pareto_front(objectives=None):
    if objectives is None:
        objectives = [QuadraticObjective1(), QuadraticObjective2()]

    # Hyperparameters
    input_dim = 1
    preference_dim = len(objectives)
    hidden_dim = 20
    num_epochs = 10000
    learning_rate = 0.005

    # Initialize network and optimizer
    network = PreferenceTargetNetwork(input_dim, preference_dim, hidden_dim)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=500, factor=0.5)

    # Store Pareto front points
    pareto_points = []

    # Reference point for hypervolume (set worse than expected worst values)
    reference_point = torch.tensor([4.0, 4.0])

    # Training loop
    for epoch in range(num_epochs):
        x = torch.randn(1, input_dim)
        preference = sample_preference(preference_dim, pareto_points)

        # Forward pass
        output = network(x, preference)
        objective_values = torch.stack([obj(output) for obj in objectives], dim=1)

        # Combined loss
        weighted_loss = (preference * objective_values).sum(dim=1).mean()
        hv_loss = hypervolume_loss(objective_values, reference_point)
        diversity_penalty = 1 / (torch.std(objective_values, dim=0).mean() + 1e-8)
        loss = weighted_loss + 0.05 * hv_loss + 0.01 * diversity_penalty

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        # Store results
        pareto_points.append(objective_values.detach().cpu().numpy().flatten())

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("Training complete.")
    return pareto_points

# Visualize results
def plot_pareto_front(pareto_points):
    pareto_points = np.array(pareto_points)
    plt.scatter(pareto_points[:, 0], pareto_points[:, 1], c='blue', label='Pareto Front')
    plt.xlabel('Objective 1 (x^2)')
    plt.ylabel('Objective 2 ((x-2)^2)')
    plt.title('Improved Pareto Front Approximation')
    plt.legend()
    plt.show()

# Run and visualize
if __name__ == "__main__":
    custom_objectives = [QuadraticObjective1(), QuadraticObjective2()]
    pareto_points = train_improved_pareto_front(custom_objectives)
    plot_pareto_front(pareto_points)
