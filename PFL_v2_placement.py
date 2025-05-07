import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from placement_metrics import calculate_access_delay, calculate_workload_balance


class Objective(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError("Objective must implement forward method")


class QuadraticObjective1(Objective):
    def forward(self, x):
        return x ** 2


class QuadraticObjective2(Objective):
    def forward(self, x):
        return (x - 2) ** 2


class PlacementObjective1(Objective):
    def __init__(self, data, input_dim):
        super().__init__()
        self.data = data
        self.input_dim = input_dim

    def forward(self, x):
        result = calculate_workload_balance(x.reshape(int(self.input_dim/2), 2), self.data)
        return torch.tensor([[result]], dtype=torch.float64)


class PlacementObjective2(Objective):
    def __init__(self, data, input_dim):
        super().__init__()
        self.data = data
        self.input_dim = input_dim

    def forward(self, x):
        result = calculate_access_delay(x.reshape(int(self.input_dim/2), 2), self.data)
        return torch.tensor([[result]], dtype=torch.float64)


class PreferenceTargetNetwork(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, preference_dim=2, hidden_dim=10):
        super(PreferenceTargetNetwork, self).__init__()
        self.input_dim = input_dim
        self.preference_dim = preference_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim + preference_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, preference):
        # Concatenate input x with preference vector
        combined_input = torch.cat(
            (x, preference.expand(x.size(0), -1)), dim=1)
        x = torch.relu(self.fc1(combined_input))
        x = self.fc2(x)
        return x


def sample_preference(preference_dim):
    pref = torch.abs(torch.randn(preference_dim))  # Non-negative
    pref = pref / pref.sum()  # Normalize to sum to 1
    return pref.view(1, -1)

# Cosine similarity loss to align output with preference


def cosine_similarity_loss(objective_values, preference):
    # Normalize both vectors
    objective_values = objective_values / \
        (torch.norm(objective_values, dim=1, keepdim=True) + 1e-8)
    preference = preference / \
        (torch.norm(preference, dim=1, keepdim=True) + 1e-8)
    cosine_sim = torch.sum(objective_values * preference, dim=1)
    return 1 - cosine_sim.mean()  # Minimize to maximize similarity

# Training function for Scalable Pareto Front Approximation


def train_scalable_pareto_front(objectives=None, input_dim=1, output_dim=1):
    # !
    # Hyperparameters
    # input_dim
    preference_dim = len(objectives)
    hidden_dim = 10 * input_dim
    num_epochs = 5000
    learning_rate = 0.01

    # Initialize network and optimizer
    network = PreferenceTargetNetwork(input_dim, output_dim, preference_dim, hidden_dim)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

    # Store Pareto front points
    pareto_points = []
    solutions_points = []

    # Training loop
    for epoch in range(num_epochs):
        # Sample input and preference
        # notice
        x = torch.randn(1, input_dim)
        preference = sample_preference(preference_dim)

        # Forward pass
        output = network(x, preference)
        # print(f"output shape: {output.shape}")  # Before passing to objectives
        # I don't even know what this dim means
        objective_values = torch.stack([obj(output) for obj in objectives], dim=0)

        # Loss: Weighted sum of objectives + cosine similarity penalty
        weighted_loss = (preference * objective_values).sum(dim=1).mean()
        angle_loss = cosine_similarity_loss(objective_values, preference)
        loss = weighted_loss + 0.1 * angle_loss  # Balance terms

        # Backpropagation
        optimizer.zero_grad()
        # loss.backward()
        optimizer.step()
        scheduler.step()

        # Store results
        pareto_points.append(objective_values.detach().cpu().numpy().flatten())
        solutions_points.append(output.detach().cpu().numpy().flatten())

        if (epoch + 1) % 500 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, "
                  f"Weighted Loss: {weighted_loss.item():.4f}, Angle Loss: {angle_loss.item():.4f}")

    print("Training complete.")
    return {
        'pareto_points': pareto_points,
        'solution_points': solution_points
    }

# Visualize the Pareto front


def plot_pareto_front(pareto_points):
    pareto_points = np.array(pareto_points)
    plt.scatter(pareto_points[:, 0], pareto_points[:,1], c='blue', label='Pareto Front')
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title('Scalable Pareto Front Approximation')
    plt.legend()
    plt.show()

def plot_pareto_front_and_save(pareto_points, path):
    pareto_points = np.array(pareto_points)
    plt.scatter(pareto_points[:, 0], pareto_points[:, 1], c='blue', label='Pareto Front')
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title('Learned Pareto Front')
    plt.legend()

    plt.savefig(path)
    plt.close()


# Run training and plot results
if __name__ == "__main__":
    df = pd.read_csv('./shanghai_dataset/block_counts.csv')
    placement_objectives = [
        PlacementObjective1(df, 6),
        PlacementObjective2(df, 6)
    ]
    pareto_points = train_scalable_pareto_front(placement_objectives, 1, 6)
    plot_pareto_front(pareto_points)
