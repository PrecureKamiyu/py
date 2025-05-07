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

def train_gradient_based(objectives, num_epochs=1000):
    model = nn.Linear(1, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    pareto_points = []

    for _ in range(num_epochs):
        optimizer.zero_grad()
        x = torch.randn(1, 1)
        output = model(x)
        obj_values = torch.stack([obj(output) for obj in objectives])
        loss = obj_values.sum()
        loss.backward()
        optimizer.step()
        pareto_points.append([obj_values[0].item(), obj_values[1].item()])

    return pareto_points

objectives = [QuadraticObjective1(), QuadraticObjective2()]
pareto_points = train_gradient_based(objectives)
plt.scatter([p[0] for p in pareto_points], [p[1] for p in pareto_points])
plt.xlabel('Objective 1')
plt.ylabel('Objective 2')
plt.title('Pareto Front (Gradient-Based Method)')
plt.show()
