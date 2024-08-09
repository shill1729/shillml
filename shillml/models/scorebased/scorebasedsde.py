import torch
import torch.nn as nn
import torch.optim as optim


# Define the neural networks for drift and diffusion
class DriftNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DriftNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.fc(x)


class DiffusionNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DiffusionNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim * input_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)
        output = self.fc(x)
        output = output.view(batch_size, x.size(1), x.size(1))
        return output


# Function to compute the score
def compute_score(drift_net, diffusion_net, x):
    mu = drift_net(x)
    sigma = diffusion_net(x)
    sigma_t = torch.transpose(sigma, 1, 2)
    sigma_inv = torch.linalg.inv(torch.bmm(sigma, sigma_t))

    # Compute the divergence term
    div_sigma = torch.zeros_like(mu)
    for i in range(x.size(1)):
        for j in range(x.size(1)):
            div_sigma[:, i] += \
            torch.autograd.grad(sigma[:, i, j], x, torch.ones_like(sigma[:, i, j]), create_graph=True)[0][:, j]

    # Score
    score = torch.bmm(sigma_inv, (2 * mu - div_sigma).unsqueeze(-1)).squeeze(-1)
    return score


# Loss function with Hutchinson's estimator for trace term
def score_loss(drift_net, diffusion_net, x, k=1):
    score = compute_score(drift_net, diffusion_net, x)
    loss = torch.mean(torch.norm(score, dim=1) ** 2)

    hutchinson_trace = 0
    for _ in range(k):
        epsilon = torch.randn_like(x)
        hutchinson_trace += torch.mean(
            torch.sum(epsilon * torch.autograd.grad(torch.sum(score * epsilon), x, create_graph=True)[0], dim=1))

    loss += 2 * hutchinson_trace / k
    return loss


# Training loop
def train(drift_net, diffusion_net, data, epochs=1000, lr=1e-3, batch_size=64):
    optimizer = optim.Adam(list(drift_net.parameters()) + list(diffusion_net.parameters()), lr=lr)
    dataset = torch.utils.data.TensorDataset(data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for batch in dataloader:
            x = batch[0]
            x.requires_grad = True
            optimizer.zero_grad()
            loss = score_loss(drift_net, diffusion_net, x)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")


# Example usage
input_dim = 2
hidden_dim = 64
data = torch.randn(1000, input_dim)  # Replace with your actual data

drift_net = DriftNN(input_dim, hidden_dim)
diffusion_net = DiffusionNN(input_dim, hidden_dim)

train(drift_net, diffusion_net, data)

