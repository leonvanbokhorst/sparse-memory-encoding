import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # For cosine similarity
import matplotlib.pyplot as plt
import numpy as np

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


# -------------------------------------
# 1. Model Definition (Same as before)
# -------------------------------------
class SparseMemory(nn.Module):
    def __init__(self, event_size, compressed_size):
        super().__init__()
        self.encoder = nn.Linear(event_size, compressed_size)
        self.decoder = nn.Linear(compressed_size, event_size)

    def forward(self, event):
        event = event.float()
        encoded = torch.relu(self.encoder(event))
        decoded = torch.sigmoid(self.decoder(encoded.float()))
        return decoded, encoded


# -------------------------------------
# 2. L1 Sparsity Penalty (Handles Importance Weights)
# -------------------------------------
def weighted_l1_sparse_penalty(encoded, beta=1.0, importance_weights=None):
    """
    Calculates L1 sparsity penalty, weighted per episode.
    Args:
        encoded (Tensor): Shape [batch_size, features]
        beta (float): Global scaling factor for the penalty.
        importance_weights (Tensor, optional): Shape [batch_size, 1]. Lower weight = less penalty. Defaults to 1.0 for all.
    Returns:
        Tensor: Scalar sparsity loss.
    """
    batch_size = encoded.size(0)

    if importance_weights is None:
        importance_weights = torch.ones(batch_size, 1, device=encoded.device)
    else:
        importance_weights = importance_weights.view(batch_size, 1).to(encoded.device)

    # Calculate L1 norm per episode (sum of absolute values across features)
    l1_norm_per_episode = torch.norm(
        encoded, p=1, dim=1, keepdim=True
    )  # Shape [batch_size, 1]

    # Apply importance weights
    # Lower weight reduces penalty for that episode
    weighted_l1 = l1_norm_per_episode * importance_weights

    # Calculate the mean weighted L1 penalty across the batch
    total_penalty = beta * torch.mean(weighted_l1)

    return total_penalty


# -------------------------------------
# 3. Generate Narratively Labeled Episodes (Sine Waves - Same as before)
# -------------------------------------
def generate_narrative_episodes(num_episodes, event_size):
    # (Using the same function as in sae_narrative_experiment.py)
    print(
        f"Generating {num_episodes} episodes of size {event_size} using sine waves..."
    )
    episodes = torch.zeros(num_episodes, event_size)
    x = torch.linspace(0, 4 * np.pi, event_size)
    base_sine = torch.sin(x) * 0.4 + 0.5
    print(f"  - Generating {num_episodes} base 'Routine Event' waves...")
    for i in range(num_episodes):
        noise = torch.rand(event_size) * 0.2 - 0.1
        episodes[i] = base_sine + noise
    narrative_labels = ["Routine"] * num_episodes
    if num_episodes > 0:
        q1_end = event_size // 4
        boost_factor = 0.4
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
        q2_start = event_size // 4
        q2_end = event_size // 2
        high_freq_sine = torch.sin(x[q2_start:q2_end] * 3) * 0.4 + 0.5
        episodes[1, q2_start:q2_end] = high_freq_sine + (
            torch.rand(q2_end - q2_start) * 0.2 - 0.1
        )
        narrative_labels[1] = "Complex/Chaotic"
        print(
            f"  - Episode 1 labelled as 'Complex/Chaotic' (Higher frequency features {q2_start}-{q2_end-1})"
        )
    episodes = torch.clamp(episodes, 0, 1)
    print("Narrative episode generation complete.")
    return episodes, narrative_labels


# -------------------------------------
# 4. Configuration
# -------------------------------------
EVENT_SIZE = 300
COMPRESSED_SIZE = 30
NUM_EPISODES = 50
EPOCHS = 1500
LEARNING_RATE = 0.001
# BETA for L1 needs tuning - L1 norms can be larger than KL div values
# Start lower and potentially increase if sparsity isn't achieved
# Lowering further for brief check based on Master 4.5's recommendation
BETA_L1 = 0.0001  # Tune this global L1 sparsity penalty strength (Lowered from 0.0005)
EPSILON = 1e-10  # For sparsity calculation in analysis

# Sparsity Importance Weights (Lower weight = LESS L1 penalty)
sparsity_importance_values = [0.3, 0.5] + [1.0] * (NUM_EPISODES - 2)
sparsity_weights_tensor = torch.tensor(sparsity_importance_values).float().unsqueeze(1)

# Reconstruction Importance Weights (Higher weight = MORE important to reconstruct accurately)
recon_importance_values = [3.0, 2.0] + [1.0] * (
    NUM_EPISODES - 2
)  # Emotional (Ep 0) most important, Chaotic (Ep 1) next
recon_weights_tensor = torch.tensor(recon_importance_values).float().unsqueeze(1)

# -------------------------------------
# 5. Generate Data
# -------------------------------------
episodes, narrative_labels = generate_narrative_episodes(
    num_episodes=NUM_EPISODES, event_size=EVENT_SIZE
)

# Ensure recon_weights_tensor is on the same device as the data/model if using GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
episodes = episodes.to(device)
sparsity_weights_tensor = sparsity_weights_tensor.to(device)
recon_weights_tensor = recon_weights_tensor.to(device)

# -------------------------------------
# 6. Initialize Model and Optimizer
# -------------------------------------
memory_net = SparseMemory(event_size=EVENT_SIZE, compressed_size=COMPRESSED_SIZE).to(
    device
)
optimizer = optim.Adam(memory_net.parameters(), lr=LEARNING_RATE)
# Use reduction='none' to get per-element loss for weighting
criterion_recon = nn.MSELoss(reduction="none")

# -------------------------------------
# 7. Training Loop with L1 Sparsity & Weighted Reconstruction
# -------------------------------------
loss_history = []
reconstruction_loss_history = []
sparsity_loss_history = []

print("\n--- Starting Training with L1 Sparsity & Weighted Reconstruction ---")
for epoch in range(EPOCHS):
    memory_net.train()
    optimizer.zero_grad()

    reconstructed, memory_trace = memory_net(episodes)

    # --- Weighted Reconstruction Loss ---
    # Calculate element-wise MSE loss: shape [batch_size, event_size]
    element_wise_recon_loss = criterion_recon(reconstructed, episodes)
    # Average loss per episode: shape [batch_size]
    recon_loss_per_episode = torch.mean(element_wise_recon_loss, dim=1)
    # Apply importance weights: shape [batch_size]
    weighted_recon_loss_per_episode = (
        recon_loss_per_episode * recon_weights_tensor.squeeze()
    )  # Ensure weights are [batch_size]
    # Average weighted loss across batch
    final_reconstruction_loss = torch.mean(weighted_recon_loss_per_episode)

    # --- Weighted L1 Sparsity Loss ---
    final_sparsity_loss = weighted_l1_sparse_penalty(
        memory_trace, beta=BETA_L1, importance_weights=sparsity_weights_tensor
    )

    # --- Total Loss ---
    total_loss = final_reconstruction_loss + final_sparsity_loss

    # Backpropagation
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(memory_net.parameters(), max_norm=1.0)
    optimizer.step()

    # Record losses (store the unweighted average recon loss for consistent comparison)
    loss_history.append(total_loss.item())
    reconstruction_loss_history.append(
        torch.mean(recon_loss_per_episode).item()
    )  # Log average unweighted recon loss
    sparsity_loss_history.append(final_sparsity_loss.item())

    # Print progress
    if epoch % 100 == 0 or epoch == EPOCHS - 1:
        print(
            f"Epoch {epoch:4d}/{EPOCHS}, Total Loss: {total_loss.item():.4f}, "
            + f"Avg Recon Loss: {torch.mean(recon_loss_per_episode).item():.4f}, L1 Sparsity Loss: {final_sparsity_loss.item():.4f}"
        )

print("--- Training complete ---")

# Save the trained model state dictionary
model_save_path = "sae_model_beta0005.pth"
torch.save(memory_net.state_dict(), model_save_path)
print(f"Model state dictionary saved to {model_save_path}")


# -------------------------------------
# 8. Plotting Loss
# -------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label="Total Loss (Weighted)")
plt.plot(
    reconstruction_loss_history,
    label="Avg Reconstruction Loss (Unweighted MSE)",
    linestyle="--",
)
plt.plot(
    sparsity_loss_history,
    label=f"Weighted L1 Sparsity Loss (Beta={BETA_L1})",
    linestyle=":",
)
plt.title("Loss Components (L1 Sparsity & Weighted Recon)")
plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------
# 9. Analysis, Retrieval & Distinctiveness
# -------------------------------------
print("\n--- Analyzing Results, Retrieval & Distinctiveness ---")
memory_net.eval()
with torch.no_grad():
    # Ensure episodes tensor is on the correct device for the final pass
    reconstructed, memory_trace = memory_net(episodes.to(device))
    # Move results back to CPU for plotting/numpy
    reconstructed = reconstructed.cpu()
    memory_trace = memory_trace.cpu()
    episodes = episodes.cpu()  # Move original data back too


# --- Function to plot comparison (same as before, uses global episodes, reconstructed, memory_trace) ---
def plot_comparison(idx, label):
    print(f"\nAnalyzing Episode {idx} ('{label}')")
    plt.figure(figsize=(18, 6))

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
    plt.plot(reconstructed[idx].numpy())
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
    sparsity = (memory_trace[idx] < EPSILON).float().mean().item()
    print(
        f"  Avg Activation: {avg_activation:.4f}, Approx Sparsity: {sparsity*100:.2f}%"
    )
    plt.grid(True)

    # Retrieval from Trace
    # Need model on CPU if data is on CPU now
    memory_net_cpu = memory_net.cpu()
    retrieved_decoded, _ = memory_net_cpu.decoder(
        memory_trace[idx].unsqueeze(0)
    ), memory_trace[idx].unsqueeze(0)
    plt.subplot(1, 4, 4)
    plt.title(f"Retrieved via Trace (Decoded Ep {idx})")
    plt.plot(retrieved_decoded.squeeze(0).detach().numpy())
    plt.ylim(-0.1, 1.1)
    plt.xlabel("Feature Index")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# --- Plot comparisons for key episodes ---
plot_comparison(0, narrative_labels[0])  # Strong Emotional
plot_comparison(1, narrative_labels[1])  # Complex/Chaotic
plot_comparison(4, narrative_labels[4])  # A Routine one

# --- Distinctiveness Analysis (Cosine Similarity & Euclidean Distance) ---
print("\n--- Memory Trace Distinctiveness Analysis ---")
trace_emotional = memory_trace[0]
trace_chaotic = memory_trace[1]
trace_routine = memory_trace[4]

# --- Cosine Similarity (Original) ---
sim_emo_routine = F.cosine_similarity(trace_emotional, trace_routine, dim=0).item()
sim_chaotic_routine = F.cosine_similarity(trace_chaotic, trace_routine, dim=0).item()
sim_emo_chaotic = F.cosine_similarity(trace_emotional, trace_chaotic, dim=0).item()
print("\nCosine Similarity (Original Traces):")
print(f"  Emotional vs. Routine: {sim_emo_routine:.4f}")
print(f"  Chaotic vs. Routine:   {sim_chaotic_routine:.4f}")
print(f"  Emotional vs. Chaotic: {sim_emo_chaotic:.4f}")

# --- Cosine Similarity (Normalized) ---
# Normalize traces using L2 norm before calculating similarity
trace_emotional_norm = F.normalize(trace_emotional, p=2, dim=0)
trace_chaotic_norm = F.normalize(trace_chaotic, p=2, dim=0)
trace_routine_norm = F.normalize(trace_routine, p=2, dim=0)

sim_norm_emo_routine = F.cosine_similarity(
    trace_emotional_norm, trace_routine_norm, dim=0
).item()
sim_norm_chaotic_routine = F.cosine_similarity(
    trace_chaotic_norm, trace_routine_norm, dim=0
).item()
sim_norm_emo_chaotic = F.cosine_similarity(
    trace_emotional_norm, trace_chaotic_norm, dim=0
).item()
print("\nCosine Similarity (L2 Normalized Traces):")
print(f"  Emotional vs. Routine: {sim_norm_emo_routine:.4f}")
print(f"  Chaotic vs. Routine:   {sim_norm_chaotic_routine:.4f}")
print(f"  Emotional vs. Chaotic: {sim_norm_emo_chaotic:.4f}")
print("(Focuses on direction/pattern, ignoring magnitude)")

# --- Euclidean Distance ---
dist_emo_routine = torch.norm(trace_emotional - trace_routine, p=2).item()
dist_chaotic_routine = torch.norm(trace_chaotic - trace_routine, p=2).item()
dist_emo_chaotic = torch.norm(trace_emotional - trace_chaotic, p=2).item()
print("\nEuclidean Distance (Original Traces):")
print(f"  Emotional vs. Routine: {dist_emo_routine:.4f}")
print(f"  Chaotic vs. Routine:   {dist_chaotic_routine:.4f}")
print(f"  Emotional vs. Chaotic: {dist_emo_chaotic:.4f}")
print("(Lower value means more similar overall, considers magnitude)")

# --- Neuron Activation Overlap Analysis ---
print("\n--- Neuron Activation Overlap Analysis (Multiple Thresholds) ---")

# Define thresholds to test
activation_thresholds = [0.005, 0.01, 0.02, 0.05]  # Added 0.05 for more range

core_neurons_per_threshold = {}

for threshold in activation_thresholds:
    print(f"\n--- Analysis for Activation Threshold: {threshold:.3f} ---")

    # Identify active neurons for each key episode type
    active_neurons_emotional = set(
        (trace_emotional > threshold).nonzero(as_tuple=True)[0].tolist()
    )
    active_neurons_chaotic = set(
        (trace_chaotic > threshold).nonzero(as_tuple=True)[0].tolist()
    )
    active_neurons_routine = set(
        (trace_routine > threshold).nonzero(as_tuple=True)[0].tolist()
    )

    print(f"Number of active neurons:")
    print(f"  Emotional (Ep 0): {len(active_neurons_emotional)}")
    print(f"  Chaotic   (Ep 1): {len(active_neurons_chaotic)}")
    print(f"  Routine   (Ep 4): {len(active_neurons_routine)}")

    # Calculate overlaps (intersections)
    overlap_emo_routine = active_neurons_emotional.intersection(active_neurons_routine)
    overlap_chaotic_routine = active_neurons_chaotic.intersection(
        active_neurons_routine
    )
    overlap_emo_chaotic = active_neurons_emotional.intersection(active_neurons_chaotic)
    overlap_all = overlap_emo_routine.intersection(
        active_neurons_chaotic
    )  # Intersection of all three

    print("\nOverlap in active neurons (Intersection size & indices):")
    print(
        f"  Emotional & Routine:   {len(overlap_emo_routine)} {sorted(list(overlap_emo_routine))}"
    )
    print(
        f"  Chaotic & Routine:     {len(overlap_chaotic_routine)} {sorted(list(overlap_chaotic_routine))}"
    )
    print(
        f"  Emotional & Chaotic:   {len(overlap_emo_chaotic)} {sorted(list(overlap_emo_chaotic))}"
    )
    print(f"  All three:             {len(overlap_all)} {sorted(list(overlap_all))}")

    # Store the core neuron(s) found at this threshold
    if len(overlap_all) > 0:
        core_neurons_per_threshold[threshold] = sorted(list(overlap_all))

# --- Detailed Analysis of Core Neuron(s) based on a chosen threshold (e.g., 0.01) ---
chosen_threshold_for_core = 0.01  # Can adjust this based on above results
core_neurons = core_neurons_per_threshold.get(chosen_threshold_for_core, [])

if core_neurons:
    print(
        f"\n--- Detailed Analysis of Core Neuron(s) {core_neurons} (based on threshold {chosen_threshold_for_core:.3f}) ---"
    )
    for ep_idx, ep_label in [(0, "Emotional"), (1, "Chaotic"), (4, "Routine")]:
        core_activations = memory_trace[ep_idx, core_neurons].tolist()
        print(
            f"  Activation of core neuron(s) in '{ep_label}' (Ep {ep_idx}): {[f'{act:.4f}' for act in core_activations]}"
        )
else:
    print(
        f"\n--- No consistent core neuron(s) found active across all three events at threshold {chosen_threshold_for_core:.3f} ---"
    )


# Overall analysis of memory trace sparsity
avg_trace_activation_all = torch.mean(memory_trace).item()
avg_sparsity_all = (memory_trace < EPSILON).float().mean().item()
print(f"\nOverall average activation across ALL traces: {avg_trace_activation_all:.4f}")
print(f"Overall approximate sparsity across ALL traces: {avg_sparsity_all*100:.2f}%")


# -------------------------------------
# 10. Deep Neuron Role Analysis (Added based on guidance)
# -------------------------------------
print("\n--- Deep Neuron Role Analysis --- ")

# --- Step 1: Identify Neuron Indices Clearly ---
# Use a chosen threshold from the overlap analysis (e.g., 0.01)
analysis_threshold = 0.01
print(
    f"Identifying neuron roles based on activation threshold: {analysis_threshold:.3f}"
)

# Ensure memory_trace is on CPU for this analysis if it wasn't already
memory_trace_cpu = memory_trace.cpu()

emotional_active_neurons = set(
    torch.nonzero(memory_trace_cpu[0] > analysis_threshold).flatten().tolist()
)
chaotic_active_neurons = set(
    torch.nonzero(memory_trace_cpu[1] > analysis_threshold).flatten().tolist()
)
routine_active_neurons = set(
    torch.nonzero(memory_trace_cpu[4] > analysis_threshold).flatten().tolist()
)

print(f"\nActive neuron sets (Indices):")
print(f"  Emotional (Ep 0): {sorted(list(emotional_active_neurons))}")
print(f"  Chaotic   (Ep 1): {sorted(list(chaotic_active_neurons))}")
print(f"  Routine   (Ep 4): {sorted(list(routine_active_neurons))}")


# Determine shared and unique neurons
shared_emo_routine = emotional_active_neurons.intersection(routine_active_neurons)
shared_emo_chaotic = emotional_active_neurons.intersection(chaotic_active_neurons)
shared_chaotic_routine = chaotic_active_neurons.intersection(routine_active_neurons)
core_neurons = shared_emo_routine.intersection(chaotic_active_neurons)

unique_emotional_neurons = (
    emotional_active_neurons - chaotic_active_neurons - routine_active_neurons
)
unique_chaotic_neurons = (
    chaotic_active_neurons - emotional_active_neurons - routine_active_neurons
)
unique_routine_neurons = (
    routine_active_neurons - emotional_active_neurons - chaotic_active_neurons
)

print(f"\nIdentified Neuron Categories:")
print(f"  Core (All Three):        {sorted(list(core_neurons))}")
print(f"  Shared Emo & Routine:    {sorted(list(shared_emo_routine))}")
print(f"  Shared Emo & Chaotic:    {sorted(list(shared_emo_chaotic))}")
print(f"  Shared Chaotic & Routine:{sorted(list(shared_chaotic_routine))}")
print(f"  Unique Emotional:        {sorted(list(unique_emotional_neurons))}")
print(f"  Unique Chaotic:          {sorted(list(unique_chaotic_neurons))}")
print(f"  Unique Routine:          {sorted(list(unique_routine_neurons))}")


# --- Step 2: Extract Activation Profiles ---
# Define the final list of neurons we definitely want to plot
# Combine interesting shared and unique neurons, removing duplicates and sorting
neurons_of_interest = sorted(
    list(
        core_neurons
        | shared_emo_routine
        | shared_emo_chaotic
        | shared_chaotic_routine
        | unique_emotional_neurons
        | unique_chaotic_neurons
        | unique_routine_neurons
    )
)

if not neurons_of_interest:
    print(
        "\nNo consistently active neurons identified across key episodes. Skipping activation profile plotting."
    )
else:
    print(
        f"\nPlotting activation profiles for neurons of interest: {neurons_of_interest}"
    )
    activation_profiles = memory_trace_cpu[:, neurons_of_interest]

    # Create labels for the plot
    neuron_labels = []
    for i in neurons_of_interest:
        label = f"Neuron {i}"
        tags = []
        if i in core_neurons:
            tags.append("Core")
        if i in unique_emotional_neurons:
            tags.append("Emo-Unique")
        if i in unique_chaotic_neurons:
            tags.append("Chaotic-Unique")
        if i in unique_routine_neurons:
            tags.append("Routine-Unique")
        # Add shared tags if not core and part of a shared set
        if i in shared_emo_routine and i not in core_neurons:
            tags.append("Emo/Routine-Shared")
        if i in shared_emo_chaotic and i not in core_neurons:
            tags.append("Emo/Chaotic-Shared")
        if i in shared_chaotic_routine and i not in core_neurons:
            tags.append("Chaotic/Routine-Shared")

        if tags:
            label += f" ({', '.join(tags)})"
        neuron_labels.append(label)

    # --- Step 3: Analyze and Visualize ---
    plt.figure(figsize=(15, 7))  # Wider figure for potentially many neurons
    for idx, label in enumerate(neuron_labels):
        plt.plot(
            activation_profiles[:, idx].detach().numpy(),
            label=label,
            linewidth=1.5,
            alpha=0.8,
        )

    # Add vertical lines to indicate the special episodes
    plt.axvline(x=0, color="r", linestyle="--", linewidth=1, label="Ep 0 (Emotional)")
    plt.axvline(x=1, color="g", linestyle="--", linewidth=1, label="Ep 1 (Chaotic)")
    plt.axvline(x=4, color="b", linestyle="--", linewidth=1, label="Ep 4 (Routine)")

    plt.title("Neuron Activation Profiles Across All Episodes")
    plt.xlabel("Episode Index")
    plt.ylabel("Activation Magnitude")
    plt.legend(fontsize="small", loc="center left", bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for legend
    plt.show()


print("\n--- Experiment Complete ---")
