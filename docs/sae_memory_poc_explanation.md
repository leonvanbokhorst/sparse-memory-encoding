# Explanation of `sae-memory-poc.py`

This document explains the Python script `sae-memory-poc.py`, which implements a basic Sparse Autoencoder (SAE) for simulating episodic memory compression and retrieval.

## Core Components

### 1. Imports

```python
import torch
import torch.nn as nn
import torch.optim as optim
```

- The script begins by importing necessary components from the PyTorch library:
  - `torch`: The core PyTorch library.
  - `torch.nn`: Contains building blocks for neural networks (layers, activation functions).
  - `torch.optim`: Provides optimization algorithms like Adam.

### 2. `SparseMemory` Class

```python
class SparseMemory(nn.Module):
    def __init__(self, event_size, compressed_size):
        super().__init__()
        self.encoder = nn.Linear(event_size, compressed_size)
        self.decoder = nn.Linear(compressed_size, event_size)

    def forward(self, event):
        encoded = torch.relu(self.encoder(event))
        decoded = torch.sigmoid(self.decoder(encoded))
        return decoded, encoded
```

- This class defines the architecture of the Sparse Autoencoder.
- **`__init__(self, event_size, compressed_size)`**:
  - Initializes the SAE module.
  - `self.encoder`: A linear layer that reduces the dimensionality from `event_size` (original memory vector size) to `compressed_size`.
  - `self.decoder`: A linear layer that attempts to reconstruct the original memory vector from the `compressed_size` back to `event_size`.
- **`forward(self, event)`**:
  - Defines the forward pass of data through the network.
  - `encoded = torch.relu(self.encoder(event))`: The input `event` (memory snapshot) is passed through the encoder. A ReLU (Rectified Linear Unit) activation function is applied. ReLU introduces non-linearity and helps in achieving sparsity by outputting 0 for negative inputs.
  - `decoded = torch.sigmoid(self.decoder(encoded))`: The compressed `encoded` representation is passed through the decoder. A Sigmoid activation function is applied to the output, typically scaling it between 0 and 1, useful for representing probabilities or normalized data.
  - Returns the `decoded` (reconstructed) memory and the `encoded` (compressed, sparse) memory trace.

### 3. `sparse_penalty` Function

```python
def sparse_penalty(encoded, sparsity_target=0.05, beta=1.0):
    avg_activation = torch.mean(encoded, dim=0)
    kl_loss = sparsity_target * torch.log(
        sparsity_target / (avg_activation + 1e-10)
    ) + (1 - sparsity_target) * torch.log(
        (1 - sparsity_target) / (1 - avg_activation + 1e-10)
    )
    return beta * torch.sum(kl_loss)
```

- This function calculates a penalty term to encourage sparsity in the `encoded` layer activations.
- It computes the average activation of each neuron in the encoded layer across the batch.
- It uses the KL divergence between the desired average activation (`sparsity_target`, e.g., 0.05) and the actual observed `avg_activation`. The goal is to make the actual average activation close to the target.
- A small value `1e-10` is added for numerical stability to avoid division by zero or log(0).
- `beta` scales the importance of this sparsity penalty relative to the reconstruction loss.

### 4. Initialization and Setup

```python
memory_net = SparseMemory(event_size=300, compressed_size=30)
optimizer = optim.Adam(memory_net.parameters(), lr=0.001)
criterion = nn.MSELoss()
```

- An instance of the `SparseMemory` network is created with an input size of 300 and a compressed size of 30.
- The `Adam` optimizer is chosen to update the network's weights during training, with a learning rate of 0.001.
- `MSELoss` (Mean Squared Error Loss) is selected as the criterion to measure the difference between the original memory and the reconstructed memory.

### 5. Simulated Training Step

```python
episode = torch.rand(1, 300)

# Train (encode, decode, optimize)
optimizer.zero_grad()
reconstructed, memory_trace = memory_net(episode)
loss = criterion(reconstructed, episode) + sparse_penalty(memory_trace)
loss.backward()
optimizer.step()
```

- A sample `episode` (memory) is created as a random tensor of shape (1, 300).
- **Training Steps:**
  1.  `optimizer.zero_grad()`: Resets the gradients of all network parameters before calculating new ones.
  2.  `reconstructed, memory_trace = memory_net(episode)`: The sample episode is passed through the SAE to get the reconstructed output and the sparse encoded trace.
  3.  `loss = criterion(...) + sparse_penalty(...)`: The total loss is calculated as the sum of the reconstruction loss (MSE between `reconstructed` and original `episode`) and the `sparse_penalty` applied to the `memory_trace`.
  4.  `loss.backward()`: Computes the gradients of the loss with respect to the network parameters (backpropagation).
  5.  `optimizer.step()`: Updates the network parameters based on the computed gradients to minimize the loss.

## Summary

This script demonstrates a single step in training a Sparse Autoencoder. The goal is to learn a compressed representation (`encoded` trace) of input data (`episode`) that is both sparse (enforced by `sparse_penalty`) and allows for reasonably accurate reconstruction (`decoded`) of the original input (measured by `criterion`). This process mimics how memory might retain essential features while discarding less relevant details.
