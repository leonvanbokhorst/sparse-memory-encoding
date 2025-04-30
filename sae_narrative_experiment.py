import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


# -------------------------------------
# 1. Model Definition (Same as before)
# -------------------------------------
class SparseMemory(nn.Module):
    def __init__(self, event_size, compressed_size):
        super().__init__()
        self.encoder = nn.Linear(event_size, compressed_size)
        self.decoder = nn.Linear(compressed_size, event_size)

    def forward(self, event):
        # Ensure input is float
        event = event.float()
        encoded = torch.relu(self.encoder(event))
        # Ensure encoded is float before passing to decoder
        decoded = torch.sigmoid(self.decoder(encoded.float()))
        return decoded, encoded


# -------------------------------------
# 2. Modified Sparsity Penalty (Handles Importance Weights)
# -------------------------------------
def weighted_sparse_penalty(
    encoded, sparsity_target=0.05, beta=1.0, importance_weights=None, epsilon=1e-10
):
    """
    Calculates sparsity penalty, potentially weighted per episode.
    Uses element-wise KL divergence calculation weighted by importance.
    Args:
        encoded (Tensor): Shape [batch_size, features]
        sparsity_target (float): Target activation level.
        beta (float): Global scaling factor for the penalty.
        importance_weights (Tensor, optional): Shape [batch_size, 1]. Lower weight = less penalty. Defaults to 1.0 for all.
        epsilon (float): Small value for numerical stability (refers to global EPSILON now).
    Returns:
        Tensor: Scalar sparsity loss.
    """
    batch_size = encoded.size(0)

    # Ensure weights are correctly setup
    if importance_weights is None:
        # Default: equal weight (higher penalty means more sparse)
        importance_weights = torch.ones(batch_size, 1, device=encoded.device)
    else:
        # Ensure weights are [batch_size, 1] for broadcasting
        importance_weights = importance_weights.view(batch_size, 1).to(encoded.device)

    # Clamp encoded values slightly away from 0 and 1 for stability if using KL directly on activations
    # Use the global EPSILON constant defined in configuration
    encoded_stable = torch.clamp(encoded, EPSILON, 1.0 - EPSILON)

    # Element-wise KL divergence components
    term1 = sparsity_target * torch.log(sparsity_target / encoded_stable)
    term2 = (1 - sparsity_target) * torch.log(
        (1 - sparsity_target) / (1 - encoded_stable)
    )

    # Element-wise KL divergence penalty - shape [batch, features]
    element_kl_penalty = term1 + term2

    # Apply importance weights (element-wise multiplication, leveraging broadcasting)
    # Lower weight reduces penalty for that episode's features
    weighted_penalty = element_kl_penalty * importance_weights

    # Calculate the mean penalty across all features and all episodes
    # Multiply by global beta scale
    total_penalty = beta * torch.mean(weighted_penalty)

    # Handle potential NaNs/Infs if clamping wasn't enough (though it should be)
    if torch.isnan(total_penalty) or torch.isinf(total_penalty):
        print(
            "Warning: NaN or Inf detected in sparsity penalty. Returning 0 for this step."
        )
        return torch.tensor(
            0.0, device=encoded.device, requires_grad=True
        )  # Return zero loss if unstable

    return total_penalty


# -------------------------------------
# 3. Generate Narratively Labeled Episodes (Sine Waves)
# -------------------------------------
def generate_narrative_episodes(num_episodes, event_size):
    print(
        f"Generating {num_episodes} episodes of size {event_size} using sine waves..."
    )
    episodes = torch.zeros(num_episodes, event_size)
    x = torch.linspace(0, 4 * np.pi, event_size)  # Base range for sine wave

    # Base sine wave + noise for all episodes ("Routine Event")
    base_sine = torch.sin(x) * 0.4 + 0.5  # Centered around 0.5, amplitude 0.4
    print(f"  - Generating {num_episodes} base 'Routine Event' waves...")
    for i in range(num_episodes):
        noise = torch.rand(event_size) * 0.2 - 0.1  # Small noise [-0.1, 0.1]
        episodes[i] = base_sine + noise

    narrative_labels = ["Routine"] * num_episodes

    # Add distinctive patterns for specific narrative types
    if num_episodes > 0:
        # Episode 0: "Strong Emotional Event" (Boosted amplitude in the first quarter)
        q1_end = event_size // 4
        boost_factor = 0.4  # Additional amplitude
        episodes[0, :q1_end] = (
            torch.sin(x[:q1_end]) * (0.4 + boost_factor)
            + 0.5
            + (torch.rand(q1_end) * 0.2 - 0.1)
        )
        narrative_labels[0] = "Strong Emotional"
        print(
            f"  - Episode 0 labelled as 'Strong Emotional' (Boosted amplitude first {q1_end} features)"
        )

    if num_episodes > 1:
        # Episode 1: "Complex/Chaotic Event" (Higher frequency sine wave in the second quarter)
        q2_start = event_size // 4
        q2_end = event_size // 2
        high_freq_sine = (
            torch.sin(x[q2_start:q2_end] * 3) * 0.4 + 0.5
        )  # Triple frequency
        episodes[1, q2_start:q2_end] = high_freq_sine + (
            torch.rand(q2_end - q2_start) * 0.2 - 0.1
        )
        narrative_labels[1] = "Complex/Chaotic"
        print(
            f"  - Episode 1 labelled as 'Complex/Chaotic' (Higher frequency features {q2_start}-{q2_end-1})"
        )

    # Ensure values stay within [0, 1] after modifications
    episodes = torch.clamp(episodes, 0, 1)
    print("Narrative episode generation complete.")
    return episodes, narrative_labels


# -------------------------------------
# 4. Configuration
# -------------------------------------
EVENT_SIZE = 300
COMPRESSED_SIZE = 30
NUM_EPISODES = 50
EPOCHS = 1500  # Increase epochs slightly for dynamic sparsity potentially
LEARNING_RATE = 0.001
SPARSITY_TARGET = 0.05
# Make BETA slightly higher maybe, as weights will reduce it for important ones
BETA = 3.0  # Global Sparsity penalty strength scaler
EPSILON = 1e-10  # Define global epsilon for numerical stability

# Importance Weights (Lower weight = LESS sparsity penalty = allow richer encoding)
# Give Emotional (0) and Chaotic (1) lower weights
importance_values = [0.3, 0.5] + [1.0] * (
    NUM_EPISODES - 2
)  # Ep 0 least penalized, Ep 1 next, others standard
importance_weights_tensor = (
    torch.tensor(importance_values).float().unsqueeze(1)
)  # Shape [NUM_EPISODES, 1]

# -------------------------------------
# 5. Generate Data
# -------------------------------------
episodes, narrative_labels = generate_narrative_episodes(
    num_episodes=NUM_EPISODES, event_size=EVENT_SIZE
)

# -------------------------------------
# 6. Initialize Model and Optimizer
# -------------------------------------
memory_net = SparseMemory(event_size=EVENT_SIZE, compressed_size=COMPRESSED_SIZE)
optimizer = optim.Adam(memory_net.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()  # Reconstruction loss

# -------------------------------------
# 7. Training Loop with Dynamic Sparsity
# -------------------------------------
loss_history = []
reconstruction_loss_history = []
sparsity_loss_history = []

print("\n--- Starting Training with Dynamic Sparsity ---")
for epoch in range(EPOCHS):
    memory_net.train()  # Set model to training mode
    optimizer.zero_grad()

    reconstructed, memory_trace = memory_net(episodes)

    # Calculate losses
    reconstruction_loss = criterion(reconstructed, episodes)

    # Use the weighted sparsity penalty function
    sparsity_loss = weighted_sparse_penalty(
        memory_trace,
        sparsity_target=SPARSITY_TARGET,
        beta=BETA,
        importance_weights=importance_weights_tensor[
            : reconstructed.size(0)
        ],  # Ensure weights match batch size if batching used later
    )

    total_loss = reconstruction_loss + sparsity_loss

    # Backpropagation
    total_loss.backward()

    # Gradient clipping (optional but good practice)
    torch.nn.utils.clip_grad_norm_(memory_net.parameters(), max_norm=1.0)

    optimizer.step()

    # Record losses
    loss_history.append(total_loss.item())
    reconstruction_loss_history.append(reconstruction_loss.item())
    sparsity_loss_history.append(sparsity_loss.item())

    # Print progress
    if epoch % 100 == 0 or epoch == EPOCHS - 1:
        print(
            f"Epoch {epoch:4d}/{EPOCHS}, Total Loss: {total_loss.item():.4f}, Recon Loss: {reconstruction_loss.item():.4f}, Sparsity Loss: {sparsity_loss.item():.4f}"
        )

print("--- Training complete ---")

# -------------------------------------
# 8. Plotting Loss
# -------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label="Total Loss")
plt.plot(reconstruction_loss_history, label="Reconstruction Loss (MSE)", linestyle="--")
plt.plot(
    sparsity_loss_history,
    label=f"Weighted Sparsity Loss (Avg KL * {BETA})",
    linestyle=":",
)
plt.title("Loss Components Over Epochs (Dynamic Sparsity)")
plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------
# 9. Analysis and Retrieval Cues
# -------------------------------------
print("\n--- Analyzing Results and Retrieval Cues ---")
memory_net.eval()  # Set model to evaluation mode
with torch.no_grad():
    reconstructed, memory_trace = memory_net(episodes)


# --- Function to plot comparison ---
def plot_comparison(idx, label):
    print(f"\nAnalyzing Episode {idx} ('{label}')")
    plt.figure(figsize=(18, 6))  # Wider figure

    # Original
    plt.subplot(1, 4, 1)
    plt.title(f"Original '{label}' (Ep {idx})")
    plt.plot(episodes[idx].numpy())  # Use plot for sine waves
    plt.ylim(-0.1, 1.1)
    plt.ylabel("Activation")
    plt.xlabel("Feature Index")
    plt.grid(True)

    # Reconstructed
    plt.subplot(1, 4, 2)
    plt.title(f"Reconstructed '{label}' (Ep {idx})")
    plt.plot(reconstructed[idx].numpy())  # Use plot
    plt.ylim(-0.1, 1.1)
    plt.xlabel("Feature Index")
    plt.grid(True)

    # Memory Trace
    plt.subplot(1, 4, 3)
    plt.title(f"Memory Trace (Encoded) for Ep {idx}")
    plt.bar(range(COMPRESSED_SIZE), memory_trace[idx].numpy())
    plt.ylabel("Activation")
    plt.xlabel("Encoded Feature Index")
    avg_activation = torch.mean(memory_trace[idx]).item()
    # Use the global EPSILON constant here too
    sparsity = (
        (memory_trace[idx] < EPSILON).float().mean().item()
    )  # Approx sparsity % (near zero activations)
    print(
        f"  Avg Activation: {avg_activation:.4f}, Approx Sparsity: {sparsity*100:.2f}%"
    )
    plt.grid(True)

    # Retrieval from Trace
    retrieved_decoded, _ = memory_net.decoder(
        memory_trace[idx].unsqueeze(0)
    ), memory_trace[idx].unsqueeze(0)
    plt.subplot(1, 4, 4)
    plt.title(f"Retrieved via Trace (Decoded Ep {idx})")
    plt.plot(retrieved_decoded.squeeze(0).detach().numpy())  # Use .detach()
    plt.ylim(-0.1, 1.1)
    plt.xlabel("Feature Index")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# --- Plot comparisons for key episodes ---
plot_comparison(0, narrative_labels[0])  # Strong Emotional
plot_comparison(1, narrative_labels[1])  # Complex/Chaotic
plot_comparison(4, narrative_labels[4])  # A Routine one

# Overall analysis of memory trace sparsity
avg_trace_activation_all = torch.mean(memory_trace).item()
avg_sparsity_all = (memory_trace < EPSILON).float().mean().item()
print(f"\nOverall average activation across ALL traces: {avg_trace_activation_all:.4f}")
print(f"Overall approximate sparsity across ALL traces: {avg_sparsity_all*100:.2f}%")
print(f"(Target sparsity was {SPARSITY_TARGET})")

print("\n--- Experiment Complete ---")
