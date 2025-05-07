import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Example objectives (replace with your own)
def objective1(x):
    return x ** 2

def objective2(x):
    return (x - 2) ** 2

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

    # def set_weights(self, weights):
    #     # Apply weights from hypernetwork to target network parameters
    #     idx = 0
    #     for param in self.parameters():
    #         param_size = param.numel()
    #         param.data = weights[idx:idx + param_size].view(param.shape)
    #         idx += param_size
    def set_weights(self, weights):
        weights = weights.view(-1)  # Ensure weights is a flat 1D tensor
        idx = 0
        for param in self.parameters():
            param_size = param.numel()
            param_shape = param.shape
            slice_weights = weights[idx:idx + param_size]
            if slice_weights.numel() != param_size:
                raise ValueError(f"Size mismatch: expected {param_size}, got {slice_weights.numel()} for shape {param_shape}")
            param.data = slice_weights.view(param_shape)
            idx += param_size

# Sample normalized preference vector
def sample_preference(input_dim):
    pref = torch.abs(torch.randn(input_dim))  # Ensure non-negative
    pref = pref / pref.sum()  # Normalize to sum to 1
    return pref.view(1, -1)

# Training function
def train_pfl():
    # Hyperparameters
    input_dim = 2  # Preference vector size (2 objectives)
    hidden_dim = 10
    num_epochs = 200
    learning_rate = 0.01

    # Initialize networks
    target_network = TargetNetwork(input_dim=1, hidden_dim=5)
    target_param_count = sum(p.numel() for p in target_network.parameters())
    hypernetwork = HyperNetwork(input_dim, hidden_dim, target_param_count)
    optimizer = optim.Adam(hypernetwork.parameters(), lr=learning_rate)

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
        obj1 = objective1(output)
        obj2 = objective2(output)

        # Weighted loss
        loss = (preference[0, 0] * obj1 + preference[0, 1] * obj2)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store objectives for Pareto front
        pareto_points.append([obj1.item(), obj2.item()])

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, "
                  f"Obj1: {obj1.item():.4f}, Obj2: {obj2.item():.4f}")

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

# Run training and plot results
pareto_points = train_pfl()
plot_pareto_front(pareto_points)
