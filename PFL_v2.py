import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Base class for objectives (unchanged from your code)


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


# Target network with preference vector as input
class PreferenceTargetNetwork(nn.Module):
    def __init__(self, input_dim=1, preference_dim=2, hidden_dim=10):
        super(PreferenceTargetNetwork, self).__init__()
        self.input_dim = input_dim
        self.preference_dim = preference_dim
        self.fc1 = nn.Linear(input_dim + preference_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, preference):
        # Concatenate input x with preference vector
        combined_input = torch.cat(
            (x, preference.expand(x.size(0), -1)), dim=1)
        x = torch.relu(self.fc1(combined_input))
        x = self.fc2(x)
        return x


# Sample normalized preference vector
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


def train_scalable_pareto_front(objectives=None):
    if objectives is None:
        objectives = [QuadraticObjective1(), QuadraticObjective2()]

    # Hyperparameters
    input_dim = 1
    preference_dim = len(objectives)
    hidden_dim = 10
    num_epochs = 5000
    learning_rate = 0.01

    # Initialize network and optimizer
    network = PreferenceTargetNetwork(input_dim, preference_dim, hidden_dim)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

    # Store Pareto front points
    pareto_points = []

    # Training loop
    for epoch in range(num_epochs):
        # Sample input and preference
        x = torch.randn(1, input_dim)
        preference = sample_preference(preference_dim)

        # Forward pass
        output = network(x, preference)
        objective_values = torch.stack(
            [obj(output) for obj in objectives], dim=1)

        # Loss: Weighted sum of objectives + cosine similarity penalty
        weighted_loss = (preference * objective_values).sum(dim=1).mean()
        angle_loss = cosine_similarity_loss(objective_values, preference)
        loss = weighted_loss + 0.1 * angle_loss  # Balance terms

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Store results
        pareto_points.append(objective_values.detach().cpu().numpy().flatten())

        if (epoch + 1) % 500 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, "
                  f"Weighted Loss: {weighted_loss.item():.4f}, Angle Loss: {angle_loss.item():.4f}")

    print("Training complete.")
    return pareto_points

# Visualize the Pareto front


def plot_pareto_front(pareto_points):
    pareto_points = np.array(pareto_points)
    plt.scatter(pareto_points[:, 0], pareto_points[:,
                1], c='blue', label='Pareto Front')
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title('Scalable Pareto Front Approximation')
    plt.legend()
    plt.show()


# Run training and plot results
if __name__ == "__main__":
    custom_objectives = [QuadraticObjective1(), QuadraticObjective2()]
    pareto_points = train_scalable_pareto_front(custom_objectives)
    plot_pareto_front(pareto_points)
