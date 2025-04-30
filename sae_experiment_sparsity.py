import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# 2. Define your Sparse Memory Model
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
    # Add a small epsilon for numerical stability
    avg_activation_stable = avg_activation + 1e-10
    sparsity_target_stable = sparsity_target + 1e-10
    inv_sparsity_target_stable = (1 - sparsity_target) + 1e-10
    inv_avg_activation_stable = (1 - avg_activation) + 1e-10

    kl_div = sparsity_target_stable * torch.log(
        sparsity_target_stable / avg_activation_stable
    ) + inv_sparsity_target_stable * torch.log(
        inv_sparsity_target_stable / inv_avg_activation_stable
    )

    return beta * torch.sum(kl_div)


# 3. Generate Simulated Episodic Memories with Sine Waves
def generate_episodes(num_episodes, event_size):
    print(
        f"Generating {num_episodes} episodes of size {event_size} using sine waves..."
    )
    episodes = torch.zeros(num_episodes, event_size)
    x = torch.linspace(0, 4 * torch.pi, event_size)  # Base range for sine wave

    # Base sine wave + noise for all episodes
    base_sine = torch.sin(x) * 0.4 + 0.5  # Centered around 0.5, amplitude 0.4
    for i in range(num_episodes):
        noise = torch.rand(event_size) * 0.2 - 0.1  # Small noise [-0.1, 0.1]
        episodes[i] = base_sine + noise

    # Add distinctive patterns to specific episodes
    if num_episodes > 0:
        # Episode 0: Boosted amplitude in the first quarter
        q1_end = event_size // 4
        boost_factor = 0.4  # Additional amplitude
        episodes[0, :q1_end] = (
            torch.sin(x[:q1_end]) * (0.4 + boost_factor)
            + 0.5
            + (torch.rand(q1_end) * 0.2 - 0.1)
        )
        print(f"  - Episode 0: Boosted amplitude (first {q1_end} features)")

    if num_episodes > 1:
        # Episode 1: Higher frequency sine wave in the second quarter
        q2_start = event_size // 4
        q2_end = event_size // 2
        high_freq_sine = (
            torch.sin(x[q2_start:q2_end] * 3) * 0.4 + 0.5
        )  # Triple frequency
        episodes[1, q2_start:q2_end] = high_freq_sine + (
            torch.rand(q2_end - q2_start) * 0.2 - 0.1
        )
        print(f"  - Episode 1: Higher frequency (features {q2_start}-{q2_end-1})")

    # Ensure values stay within [0, 1] after modifications
    episodes = torch.clamp(episodes, 0, 1)
    print("Episode generation complete.")
    return episodes


# Configuration
EVENT_SIZE = 300
COMPRESSED_SIZE = 30
NUM_EPISODES = 50
EPOCHS = 1000
LEARNING_RATE = 0.001
SPARSITY_TARGET = 0.05
BETA = 1.0  # Sparsity penalty strength

episodes = generate_episodes(num_episodes=NUM_EPISODES, event_size=EVENT_SIZE)

# 4. Train the Sparse Memory Model
memory_net = SparseMemory(event_size=EVENT_SIZE, compressed_size=COMPRESSED_SIZE)
optimizer = optim.Adam(memory_net.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

loss_history = []
reconstruction_loss_history = []
sparsity_loss_history = []

print("Starting training...")
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    reconstructed, memory_trace = memory_net(episodes)

    reconstruction_loss = criterion(reconstructed, episodes)
    sparsity_loss = sparse_penalty(
        memory_trace, sparsity_target=SPARSITY_TARGET, beta=BETA
    )
    total_loss = reconstruction_loss + sparsity_loss

    total_loss.backward()
    optimizer.step()

    loss_history.append(total_loss.item())
    reconstruction_loss_history.append(reconstruction_loss.item())
    sparsity_loss_history.append(sparsity_loss.item())

    if epoch % 100 == 0 or epoch == EPOCHS - 1:
        print(
            f"Epoch {epoch:4d}/{EPOCHS}, Total Loss: {total_loss.item():.4f}, Recon Loss: {reconstruction_loss.item():.4f}, Sparsity Loss: {sparsity_loss.item():.4f}"
        )

print("Training complete.")

# Plotting Loss Components
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label="Total Loss")
plt.plot(reconstruction_loss_history, label="Reconstruction Loss (MSE)", linestyle="--")
plt.plot(sparsity_loss_history, label=f"Sparsity Loss (KL Div * {BETA})", linestyle=":")
plt.title("Loss Components Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.legend()
plt.grid(True)
plt.show()


# 5. Analyze the Results
print("\nAnalyzing results...")
# Ensure network is in evaluation mode for consistent results (though not strictly necessary here)
memory_net.eval()
with torch.no_grad():  # Turn off gradients for analysis
    reconstructed, memory_trace = memory_net(episodes)

# Check the first "important" episode
episode_idx = 0
print(f"Analyzing episode {episode_idx}...")

plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.title(f"Original Episode {episode_idx}")
plt.bar(range(EVENT_SIZE), episodes[episode_idx].numpy())
plt.ylim(0, 1)  # Consistent y-axis
plt.ylabel("Activation")
plt.xlabel("Feature Index")

plt.subplot(1, 3, 2)
plt.title(f"Reconstructed Episode {episode_idx}")
plt.bar(range(EVENT_SIZE), reconstructed[episode_idx].numpy())
plt.ylim(0, 1)
plt.xlabel("Feature Index")


# Analyze the sparsity of the memory trace for this episode
plt.subplot(1, 3, 3)
plt.title(f"Memory Trace (Encoded) for Episode {episode_idx}")
plt.bar(range(COMPRESSED_SIZE), memory_trace[episode_idx].numpy())
plt.ylabel("Activation")
plt.xlabel("Encoded Feature Index")
avg_activation = torch.mean(memory_trace[episode_idx]).item()
print(
    f"  Average activation in memory trace for episode {episode_idx}: {avg_activation:.4f}"
)


plt.tight_layout()
plt.show()

# Check the second "important" episode
episode_idx = 1
print(f"Analyzing episode {episode_idx}...")

plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.title(f"Original Episode {episode_idx}")
plt.bar(range(EVENT_SIZE), episodes[episode_idx].numpy())
plt.ylim(0, 1)
plt.ylabel("Activation")
plt.xlabel("Feature Index")


plt.subplot(1, 3, 2)
plt.title(f"Reconstructed Episode {episode_idx}")
plt.bar(range(EVENT_SIZE), reconstructed[episode_idx].numpy())
plt.ylim(0, 1)
plt.xlabel("Feature Index")

# Analyze the sparsity of the memory trace for this episode
plt.subplot(1, 3, 3)
plt.title(f"Memory Trace (Encoded) for Episode {episode_idx}")
plt.bar(range(COMPRESSED_SIZE), memory_trace[episode_idx].numpy())
plt.ylabel("Activation")
plt.xlabel("Encoded Feature Index")
avg_activation = torch.mean(memory_trace[episode_idx]).item()
print(
    f"  Average activation in memory trace for episode {episode_idx}: {avg_activation:.4f}"
)

plt.tight_layout()
plt.show()


# Analyze a "non-important" episode (e.g., the 5th one)
episode_idx = 4
print(f"Analyzing episode {episode_idx} (a 'normal' episode)...")


plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.title(f"Original Episode {episode_idx}")
plt.bar(range(EVENT_SIZE), episodes[episode_idx].numpy())
plt.ylim(0, 1)
plt.ylabel("Activation")
plt.xlabel("Feature Index")


plt.subplot(1, 3, 2)
plt.title(f"Reconstructed Episode {episode_idx}")
plt.bar(range(EVENT_SIZE), reconstructed[episode_idx].numpy())
plt.ylim(0, 1)
plt.xlabel("Feature Index")

# Analyze the sparsity of the memory trace for this episode
plt.subplot(1, 3, 3)
plt.title(f"Memory Trace (Encoded) for Episode {episode_idx}")
plt.bar(range(COMPRESSED_SIZE), memory_trace[episode_idx].numpy())
plt.ylabel("Activation")
plt.xlabel("Encoded Feature Index")
avg_activation = torch.mean(memory_trace[episode_idx]).item()
print(
    f"  Average activation in memory trace for episode {episode_idx}: {avg_activation:.4f}"
)


plt.tight_layout()
plt.show()


# Overall analysis of memory trace sparsity
avg_trace_activation = torch.mean(memory_trace).item()
print(
    f"\nOverall average activation across all memory traces: {avg_trace_activation:.4f}"
)
print(f"(Target sparsity was {SPARSITY_TARGET})")
