import torch
import torch.nn as nn
import torch.optim as optim


class SparseMemory(nn.Module):
    def __init__(self, event_size, compressed_size):
        super().__init__()
        self.encoder = nn.Linear(event_size, compressed_size)
        self.decoder = nn.Linear(compressed_size, event_size)

    def forward(self, event):
        encoded = torch.relu(self.encoder(event))
        decoded = torch.sigmoid(self.decoder(encoded))
        return decoded, encoded


def sparse_penalty(encoded, sparsity_target=0.05, beta=1.0):
    avg_activation = torch.mean(encoded, dim=0)
    kl_loss = sparsity_target * torch.log(
        sparsity_target / (avg_activation + 1e-10)
    ) + (1 - sparsity_target) * torch.log(
        (1 - sparsity_target) / (1 - avg_activation + 1e-10)
    )
    return beta * torch.sum(kl_loss)


# Define sizes: for instance, each memory could be a 300-dimensional vector
memory_net = SparseMemory(event_size=300, compressed_size=30)
optimizer = optim.Adam(memory_net.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Simulated episodic memory (one training step)
episode = torch.rand(1, 300)

# Train (encode, decode, optimize)
optimizer.zero_grad()
reconstructed, memory_trace = memory_net(episode)
loss = criterion(reconstructed, episode) + sparse_penalty(memory_trace)
loss.backward()
optimizer.step()
