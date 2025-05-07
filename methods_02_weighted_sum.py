import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


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


class CombinedObjective(nn.Module):
    def __init__(self, objectives):
        super().__init__()
        self.objectives = nn.ModuleList(objectives)

    def forward(self, x):
        return torch.stack([obj(x) for obj in self.objectives])


def train_weighted_sum(objectives, num_weights=100):
    weights = np.linspace(0, 1, num_weights)
    pareto_points = []

    for w in weights:
        model = nn.Linear(1, 1)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        for _ in range(1000):
            optimizer.zero_grad()
            x = torch.randn(1, 1)
            output = model(x)
            obj_values = torch.stack([obj(output) for obj in objectives])
            loss = w * obj_values[0] + (1 - w) * obj_values[1]
            loss.backward()
            optimizer.step()
        pareto_points.append([obj_values[0].item(), obj_values[1].item()])

    return pareto_points


if __name__ == "__main__":
    objectives = [QuadraticObjective1(), QuadraticObjective2()]
    pareto_points = train_weighted_sum(objectives)
    plt.scatter([p[0] for p in pareto_points], [p[1] for p in pareto_points])
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title('Pareto Front (Weighted Sum Method)')
    plt.savefig("fig/M02_Weighted_Sum_fig")
