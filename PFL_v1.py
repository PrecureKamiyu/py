import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Base class for objectives


class Objective(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError("Objective must implement forward method")

# Example objectives


class QuadraticObjective1(Objective):
    def forward(self, x):
        return x ** 2


class QuadraticObjective2(Objective):
    def forward(self, x):
        return (x - 2) ** 2


class CombinedObjective(nn.Module):
    def __init__(self, objectives):
        super().__init__()
        self.objectives = nn.ModuleList(objectives)

    def forward(self, x):
        return torch.stack([obj(x) for obj in self.objectives])

# Hypernetwork to generate weights for the target network


class HyperNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, target_param_count):
        super(HyperNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, target_param_count)

    def forward(self, preference):
        x = torch.relu(self.fc1(preference))
        x = self.fc2(x)
        return x

# Target network to compute objectives with generated weights


class TargetNetwork(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=5):
        super(TargetNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def set_weights(self, weights):
        weights = weights.view(-1)  # Ensure weights is a flat 1D tensor
        idx = 0
        for param in self.parameters():
            param_size = param.numel()
            param_shape = param.shape
            slice_weights = weights[idx:idx + param_size]
            if slice_weights.numel() != param_size:
                raise ValueError(
                    f"Size mismatch: expected {param_size}, got {slice_weights.numel()} for shape {param_shape}")
            param.data = slice_weights.view(param_shape)
            idx += param_size

# Sample normalized preference vector


def sample_preference(input_dim):
    pref = torch.abs(torch.randn(input_dim))  # Ensure non-negative
    pref = pref / pref.sum()  # Normalize to sum to 1
    return pref.view(1, -1)


def create_objectives():
    return [
        QuadraticObjective1(),
        QuadraticObjective2()
    ]

# Training function


def train_pfl(objectives=None):
    if objectives is None:
        objectives = create_objectives()

    # Hyperparameters
    input_dim = len(objectives)  # Preference vector size based on number of objectives
    hidden_dim = 10
    num_epochs = 10000  # Increased from 200 to 1000
    learning_rate = 0.05  # Start with higher learning rate

    # Initialize networks and objectives
    target_network = TargetNetwork(input_dim=1, hidden_dim=5)
    target_param_count = sum(p.numel() for p in target_network.parameters())
    hypernetwork = HyperNetwork(input_dim, hidden_dim, target_param_count)
    optimizer = optim.Adam(hypernetwork.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    combined_objective = CombinedObjective(objectives)

    # Store Pareto front points for visualization
    pareto_points = []

    # Training loop
    for epoch in range(num_epochs):
        # Sample a normalized preference vector
        preference = sample_preference(input_dim)

        # Generate target network weights
        weights = hypernetwork(preference)
        target_network.set_weights(weights)

        # Compute objectives
        x = torch.randn(1, 1)  # Input with proper shape
        output = target_network(x)
        objective_values = combined_objective(output)

        # Dominance-Based Pareto Loss with Diversity Penalty
        margin = 0.1  # Minimum gap between solutions

        # Compute dominance gap
        dominance_gap = (objective_values.unsqueeze(0) - objective_values.unsqueeze(1)).clamp(min=0).sum(dim=2)
        dominance_loss = torch.max(torch.tensor(0.0), margin - dominance_gap).mean()

        # Compute diversity penalty to encourage spreading across the Pareto front
        diversity_penalty = 1 / (torch.std(objective_values, dim=0).mean() + 1e-8)  # Minimize inverse std dev

        # Combine dominance loss and diversity penalty
        loss = dominance_loss + 0.1 * diversity_penalty

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Store objectives for Pareto front
        pareto_points.append(objective_values.detach().cpu().numpy().tolist())

        if (epoch + 1) % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, LR: {current_lr:.6f}")

    print("Training complete.")
    return pareto_points


# Visualize the Pareto front
def plot_pareto_front(pareto_points):
    pareto_points = np.array(pareto_points)
    plt.scatter(pareto_points[:, 0], pareto_points[:, 1], c='blue', label='Pareto Front')
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title('Learned Pareto Front')
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
    # Create custom objectives if needed
    custom_objectives = [
        QuadraticObjective1(),
        QuadraticObjective2()
    ]

    pareto_points = train_pfl(custom_objectives)
    plot_pareto_front(pareto_points)
