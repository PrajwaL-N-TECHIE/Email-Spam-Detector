import numpy as np
from sklearn.datasets import load_iris
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Load dataset and convert to tensors
data, target = load_iris(return_X_y=True)
data_tensor = torch.tensor(data, dtype=torch.float32)
target_tensor = torch.tensor(target, dtype=torch.long)

# Define Bayesian Neural Network with Dropout
class BayesianNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):  # Corrected __init__ method
        super(BayesianNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),  # Apply Dropout here
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x, train=True):
        if train:
            return self.layers(x)  # Dropout is applied during training
        else:
            # During evaluation, skip Dropout layers
            return self.layers[:2](x)  # Only pass through Linear and ReLU layers

# Initialize model, loss, and optimizer
model = BayesianNN(4, 100, 3)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for step in range(3000):
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model(data_tensor, train=True), target_tensor)
    loss.backward()
    optimizer.step()

    if step % 300 == 0:
        acc = (model(data_tensor).argmax(1) == target_tensor).float().mean().item() * 100
        print(f"Step {step} - Accuracy: {acc:.2f}% - Loss: {loss.item():.2f}")

# Monte Carlo Dropout Evaluation
model.eval()
with torch.no_grad():
    preds = torch.stack([model(data_tensor, train=True).argmax(1) for _ in range(50)])
    final_preds = preds.mode(0).values  # Getting the mode of predictions over 50 iterations
    accuracy = (final_preds == target_tensor).float().mean().item() * 100
    print(f"Final Accuracy (MC dropout): {accuracy:.2f}%")

# Plot results
def draw_graph(predicted):
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    for ax, labels, title in zip(axs, [target, predicted], ["REAL", "PREDICTED"]):
        scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, marker='o')
        ax.set_title(title)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        plt.colorbar(scatter, ax=ax)

draw_graph(final_preds)
plt.show()
