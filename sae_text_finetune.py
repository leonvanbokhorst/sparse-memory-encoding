import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
import json  # To load the generated data
import os
from sklearn.decomposition import PCA

# Ensure reproducibility if needed (optional)
torch.manual_seed(42)
np.random.seed(42)


# -------------------------------------
# 1. Model Definition (with Projection Layer)
# -------------------------------------
class SparseMemoryText(nn.Module):
    def __init__(self, embedding_dim, event_size, compressed_size):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, event_size)
        self.encoder = nn.Linear(event_size, compressed_size)
        self.decoder = nn.Linear(compressed_size, event_size)
        print(
            f"Initialized SparseMemoryText: Proj [{embedding_dim}->{event_size}], Enc [{event_size}->{compressed_size}], Dec [{compressed_size}->{event_size}]"
        )

    def forward(self, text_embedding):
        text_embedding = text_embedding.float()
        projected = torch.relu(self.projection(text_embedding))
        encoded = torch.relu(self.encoder(projected))
        decoded = torch.sigmoid(self.decoder(encoded.float()))
        # Return decoded, encoded, AND the projected tensor for loss calculation
        return decoded, encoded, projected


# -------------------------------------
# 2. Configuration
# -------------------------------------
# --- Data/Model Dimensions ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EVENT_SIZE = 300
COMPRESSED_SIZE = 30

# --- Paths ---
PRETRAINED_MODEL_PATH = "sae_model_beta0005.pth"  # From sine-wave training
SYNTHETIC_DATA_PATH = "synthetic_narrative_data.json"  # Path to generated data
FINETUNED_MODEL_SAVE_PATH = "sae_text_finetuned.pth"

# --- Fine-tuning Hyperparameters ---
FINETUNE_EPOCHS = 200  # Adjust as needed
LEARNING_RATE = 1e-4  # Lower LR for fine-tuning
BETA_L1 = 0.002  # Sparsity strength (Increased further from 0.001)
BATCH_SIZE = 32  # Add batch size for training on larger data

# --- Simplified Category Names ---
CATEGORIES = ["emotional", "complex", "routine"]  # Use simplified names

# --- Analysis Configuration ---
# Keep sine-wave neuron IDs for potential initial comparison,
# but remember new roles will emerge for text.
CORE_NEURONS_SINE = [15, 21]
UNIQUE_EMO_NEURONS_SINE = [5]
UNIQUE_ROUTINE_NEURONS_SINE = [9, 28]
SHARED_EMO_ROUTINE_NEURONS_SINE = [0, 7, 17, 25, 26]
NEURONS_TO_ANALYZE = sorted(
    list(
        set(
            CORE_NEURONS_SINE
            + UNIQUE_EMO_NEURONS_SINE
            + UNIQUE_ROUTINE_NEURONS_SINE
            + SHARED_EMO_ROUTINE_NEURONS_SINE
        )
    )
)
EPSILON = 1e-10
ANALYSIS_THRESHOLD = 0.01

# --- Device Setup ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device found. Using MPS.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA device found. Using CUDA.")
else:
    device = torch.device("cpu")
    print("MPS/CUDA not found. Using CPU.")

# -------------------------------------
# 3. Data Preparation
# -------------------------------------
print(f"Loading synthetic textual data from {SYNTHETIC_DATA_PATH}...")
try:
    with open(SYNTHETIC_DATA_PATH, "r", encoding="utf-8") as f:
        synthetic_data = json.load(f)
    print(f"Loaded {len(synthetic_data)} data points.")
except FileNotFoundError:
    print(
        f"Error: Synthetic data file not found at {SYNTHETIC_DATA_PATH}. Please run generation script."
    )
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- Prepare Data for Model ---
all_texts = [item["text"] for item in synthetic_data]
category_names = [item["category"] for item in synthetic_data]

# --- Encode Texts ---
print(f"Loading sentence transformer: {EMBEDDING_MODEL_NAME}")
try:
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
except TypeError:
    print("Warning: Could not set device for SentenceTransformer directly.")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

print(f"Encoding {len(all_texts)} text samples...")
embeddings_np = embedder.encode(all_texts, show_progress_bar=True)
embeddings_tensor = torch.tensor(
    embeddings_np
).float()  # Keep on CPU initially for Dataset
print(f"Embeddings tensor shape: {embeddings_tensor.shape}")

# --- Map Categories to Indices and Create Weights ---
category_map = {name: i for i, name in enumerate(CATEGORIES)}  # Use simplified names
category_indices = torch.tensor([category_map[name] for name in category_names]).long()

# Use simplified category names for weights
recon_weights_map = {
    category_map["emotional"]: 3.0,
    category_map["complex"]: 2.0,
    category_map["routine"]: 1.0,
}
recon_weights = torch.tensor(
    [recon_weights_map[i.item()] for i in category_indices]
).float()

sparsity_weights_map = {
    category_map["emotional"]: 0.3,
    category_map["complex"]: 0.5,
    category_map["routine"]: 1.0,
}
sparsity_weights = (
    torch.tensor([sparsity_weights_map[i.item()] for i in category_indices])
    .float()
    .unsqueeze(1)
)  # Shape [N, 1]

# --- Create DataLoader ---
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(embeddings_tensor, recon_weights, sparsity_weights)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"Created DataLoader with batch size {BATCH_SIZE}")

# -------------------------------------
# 4. Model Loading & Setup
# -------------------------------------
print("Initializing SAE model for fine-tuning...")
model = SparseMemoryText(
    embedding_dim=EMBEDDING_DIM, event_size=EVENT_SIZE, compressed_size=COMPRESSED_SIZE
)
model.to(device)  # Move model to device early

# --- Check if FINE-TUNED model exists ---
if os.path.exists(FINETUNED_MODEL_SAVE_PATH):
    print(
        f"Found existing fine-tuned model at {FINETUNED_MODEL_SAVE_PATH}. Loading weights..."
    )
    model.load_state_dict(torch.load(FINETUNED_MODEL_SAVE_PATH, map_location=device))
    print("Fine-tuned weights loaded. Skipping training.")
    skip_training = True
else:
    print(
        f"No fine-tuned model found at {FINETUNED_MODEL_SAVE_PATH}. Loading pre-trained weights for fine-tuning..."
    )
    skip_training = False
    try:
        print(f"Loading pre-trained state dictionary from: {PRETRAINED_MODEL_PATH}")
        state_dict = torch.load(PRETRAINED_MODEL_PATH, map_location="cpu")
        model.load_state_dict(
            state_dict, strict=False
        )  # Load encoder/decoder, ignore projection
        print("Loaded pre-trained encoder/decoder weights.")
    except Exception as e:
        print(
            f"Warning: Could not load pre-trained weights from {PRETRAINED_MODEL_PATH}. Training from scratch. Error: {e}"
        )

    # Optimizer and Criterion only needed if training
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_recon = nn.MSELoss(reduction="none")  # For weighted loss

# -------------------------------------
# 5. Fine-tuning Loop (Conditional)
# -------------------------------------
if not skip_training:
    print(f"\n--- Starting Fine-tuning for {FINETUNE_EPOCHS} Epochs --- ")
    epoch_loss_history = []

    model.train()  # Set model to training mode
    for epoch in range(FINETUNE_EPOCHS):
        epoch_total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_sparsity_loss = 0.0
        num_batches = 0

        for (
            batch_embeddings,
            batch_recon_weights,
            batch_sparsity_weights,
        ) in data_loader:
            # Move batch data to the correct device
            batch_embeddings = batch_embeddings.to(device)
            batch_recon_weights = batch_recon_weights.to(device)
            batch_sparsity_weights = batch_sparsity_weights.to(device)

            optimizer.zero_grad()

            reconstructed, encoded, projected = model(batch_embeddings)

            # --- Weighted Reconstruction Loss ---
            element_wise_recon_loss = criterion_recon(reconstructed, projected)
            recon_loss_per_episode = torch.mean(element_wise_recon_loss, dim=1)
            # Apply weights for this batch
            weighted_recon_loss = (recon_loss_per_episode * batch_recon_weights).mean()

            # --- Weighted L1 Sparsity Loss ---
            l1_norm_per_episode = torch.norm(encoded, p=1, dim=1, keepdim=True)
            weighted_l1_per_episode = (
                l1_norm_per_episode * batch_sparsity_weights
            )  # sparsity_weights is [N_batch, 1]
            sparsity_loss = BETA_L1 * torch.mean(weighted_l1_per_episode)

            # --- Total Loss ---
            total_loss = weighted_recon_loss + sparsity_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_total_loss += total_loss.item()
            epoch_recon_loss += weighted_recon_loss.item()
            epoch_sparsity_loss += sparsity_loss.item()
            num_batches += 1

        # Calculate average loss for the epoch
        avg_epoch_total_loss = epoch_total_loss / num_batches
        avg_epoch_recon_loss = epoch_recon_loss / num_batches
        avg_epoch_sparsity_loss = epoch_sparsity_loss / num_batches
        epoch_loss_history.append(avg_epoch_total_loss)

        if epoch % 10 == 0 or epoch == FINETUNE_EPOCHS - 1:
            print(
                f"Epoch {epoch:4d}/{FINETUNE_EPOCHS}, Avg Total Loss: {avg_epoch_total_loss:.6f}, "
                f"Avg Recon Loss (W): {avg_epoch_recon_loss:.6f}, Avg Sparsity Loss: {avg_epoch_sparsity_loss:.6f}"
            )

    print("--- Fine-tuning complete ---")

    # -------------------------------------
    # 6. Save Fine-tuned Model
    # -------------------------------------
    print(f"Saving fine-tuned model to {FINETUNED_MODEL_SAVE_PATH}...")
    torch.save(model.state_dict(), FINETUNED_MODEL_SAVE_PATH)
    print("Fine-tuned model saved.")

# -------------------------------------
# 7. Post-tuning Analysis (Neuron Roles)
# -------------------------------------
print("\n--- Analyzing Neuron Activations of Fine-tuned Model --- ")
model.eval()  # Set model to evaluation mode
results_finetuned = {}
all_memory_traces = {}  # Store traces for overall analysis later if needed

# Process samples category by category for easier analysis aggregation
with torch.no_grad():
    for category in CATEGORIES:
        # Select texts and encode only for the current category
        category_texts = [
            item["text"] for item in synthetic_data if item["category"] == category
        ]
        if not category_texts:
            print(f"No texts found for category '{category}' in loaded data.")
            continue

        embeddings_np = embedder.encode(category_texts)
        embeddings = torch.tensor(embeddings_np).float().to(device)

        # Get traces - we only need the middle output (encoded)
        _, memory_traces, _ = model(embeddings)
        memory_traces_cpu = memory_traces.cpu()  # Move to CPU for numpy/analysis
        results_finetuned[category] = {
            "texts": category_texts,
            "memory_traces": memory_traces_cpu.numpy(),  # Keep original numpy format if needed elsewhere
            "memory_traces_tensor": memory_traces_cpu,  # Store tensor for easier pytorch ops
        }
        all_memory_traces[category] = memory_traces_cpu  # Collect all traces


# --- 7a. Analysis of Previously Identified Sine-Wave Neurons (For Comparison) ---
print(f"\n--- 7a. Analysis of Sine-Wave Neurons ({NEURONS_TO_ANALYZE}) ---")
print(
    "(Sine Core={CORE_NEURONS_SINE}, Sine Unique Emo={UNIQUE_EMO_NEURONS_SINE}, Sine Unique Routine={UNIQUE_ROUTINE_NEURONS_SINE})"
)
activation_summary_sine = {}
for category, data in results_finetuned.items():
    print(f"\n{category} Texts (Sine Neuron Focus, {len(data['texts'])} samples):")
    traces_np = data["memory_traces"]  # Use numpy array here
    # Ensure we handle cases where NEURONS_TO_ANALYZE might be empty or invalid
    if NEURONS_TO_ANALYZE and traces_np.shape[1] > max(NEURONS_TO_ANALYZE):
        avg_activations = np.mean(traces_np[:, NEURONS_TO_ANALYZE], axis=0)
        activation_summary_sine[category] = avg_activations
        print(
            f"  Avg Activations ({NEURONS_TO_ANALYZE}): {[f'{act:.4f}' for act in avg_activations]}"
        )

        # Print specific groups based on SINE analysis
        core_indices = [
            NEURONS_TO_ANALYZE.index(n)
            for n in CORE_NEURONS_SINE
            if n in NEURONS_TO_ANALYZE
        ]
        if core_indices:
            print(
                f"    Avg Sine Core ({CORE_NEURONS_SINE}) Activations: {[f'{avg_activations[i]:.4f}' for i in core_indices]}"
            )

        unique_emo_indices = [
            NEURONS_TO_ANALYZE.index(n)
            for n in UNIQUE_EMO_NEURONS_SINE
            if n in NEURONS_TO_ANALYZE
        ]
        if unique_emo_indices:
            print(
                f"    Avg Sine Unique Emo ({UNIQUE_EMO_NEURONS_SINE}) Activations: {[f'{avg_activations[i]:.4f}' for i in unique_emo_indices]}"
            )

        unique_routine_indices = [
            NEURONS_TO_ANALYZE.index(n)
            for n in UNIQUE_ROUTINE_NEURONS_SINE
            if n in NEURONS_TO_ANALYZE
        ]
        if unique_routine_indices:
            print(
                f"    Avg Sine Unique Routine ({UNIQUE_ROUTINE_NEURONS_SINE}) Activations: {[f'{avg_activations[i]:.4f}' for i in unique_routine_indices]}"
            )
    else:
        print(
            "  Could not perform analysis on specified sine-wave neurons (check indices and trace shape)."
        )

print(
    "\nNote: Roles of sine-wave neurons likely shifted during fine-tuning. Deeper analysis follows."
)


# --- 7b. De Novo Analysis Across ALL Neurons ---
print(f"\n--- 7b. De Novo Analysis (All {COMPRESSED_SIZE} Neurons) ---")
k_top = 5  # Number of top neurons to display
sparsity_threshold = 0.01  # Threshold for sparsity calculation
category_avg_activations = {}  # To store average activations for plotting

for category in CATEGORIES:
    if category not in results_finetuned:
        print(f"\nSkipping De Novo analysis for {category} (no data).")
        continue

    print(f"\nAnalyzing Category: {category}")
    data = results_finetuned[category]
    memory_traces = data["memory_traces_tensor"]  # Use the tensor version

    if memory_traces is None or memory_traces.nelement() == 0:
        print("  No memory traces found for this category.")
        continue

    # --- Calculate Average Activations (All Neurons) ---
    avg_activations_all = memory_traces.mean(dim=0)
    category_avg_activations[category] = (
        avg_activations_all.numpy()
    )  # Store numpy for plotting

    # --- Top-k Neuron Analysis ---
    if avg_activations_all.nelement() > 0:
        # Ensure k is not larger than the number of neurons
        actual_k = min(k_top, avg_activations_all.shape[0])
        top_k_values, top_k_indices = torch.topk(avg_activations_all, actual_k)
        print(f"  Top {actual_k} Neurons (Index: Avg Activation):")
        for i in range(actual_k):
            print(
                f"    Neuron {top_k_indices[i].item():2d}: {top_k_values[i].item():.4f}"
            )
    else:
        print("  Could not calculate top-k neurons (no activations).")

    # --- Quantitative Sparsity Check ---
    sparsity_level = (memory_traces < sparsity_threshold).float().mean().item()
    print(f"  Sparsity (< {sparsity_threshold}): {sparsity_level*100:.2f}%")


# --- 7c. Visualization of All Neuron Activations ---
print(f"\n--- 7c. Visualization (All {COMPRESSED_SIZE} Neurons) ---")

if not category_avg_activations:
    print("No data available for visualization.")
else:
    plt.figure(figsize=(15, 7))  # Adjusted size
    num_neurons = COMPRESSED_SIZE
    bar_width = 0.25
    index = np.arange(num_neurons)

    categories_present = list(category_avg_activations.keys())

    for i, category in enumerate(categories_present):
        plt.bar(
            index + i * bar_width,
            category_avg_activations[category],
            bar_width,
            label=category,
        )

    plt.xlabel("Neuron Index")
    plt.ylabel("Average Activation")
    plt.title("Average Neuron Activations by Narrative Category (Post Fine-tuning)")
    plt.xticks(
        index + bar_width * (len(categories_present) - 1) / 2, index
    )  # Center ticks
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()  # Adjust layout

    # Save the plot
    plot_save_path = "neuron_activation_comparison.png"
    try:
        plt.savefig(plot_save_path)
        print(f"Saved activation comparison plot to {plot_save_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    # plt.show() # Optional: Display plot if running interactively


# --- 7d. Qualitative Neuron Interpretation (Top Activating Texts) ---
print(f"\n--- 7d. Qualitative Neuron Interpretation (Top Activating Texts) ---")

# Ensure we have all traces and corresponding texts
if not all_memory_traces:
    print("No memory traces collected for qualitative analysis.")
else:
    # Concatenate traces from all categories
    all_traces_list = [
        t for cat_traces in all_memory_traces.values() for t in cat_traces
    ]
    if not all_traces_list:
        print("Collected memory traces are empty.")
    else:
        # Assuming all_texts list corresponds to the order of data loading before splitting
        # Make sure all_texts is the original full list loaded from JSON
        try:
            # Check if all_texts was defined earlier and matches expected length
            if "all_texts" not in locals() or len(all_texts) != len(all_traces_list):
                print(
                    "Error: Mismatch between number of texts and number of traces. Re-check data prep."
                )
                # Fallback or re-fetch all_texts if necessary
                print("Attempting to re-fetch texts...")
                all_texts = [item["text"] for item in synthetic_data]
                if len(all_texts) != len(all_traces_list):
                    raise ValueError(
                        "Fatal Error: Cannot align texts and traces for analysis."
                    )

            all_memory_traces_tensor = torch.cat(
                list(all_memory_traces.values()), dim=0
            )
            num_total_samples, num_neurons_check = all_memory_traces_tensor.shape

            if num_neurons_check != COMPRESSED_SIZE:
                print(
                    f"Warning: Concatenated traces have {num_neurons_check} features, expected {COMPRESSED_SIZE}."
                )

            if num_total_samples != len(all_texts):
                print(
                    f"Warning: Number of concatenated traces ({num_total_samples}) doesn't match number of texts ({len(all_texts)}). Alignment might be incorrect."
                )

            print(f"Analyzing top texts for {num_total_samples} total samples.")

            # --- Focus on key neurons from the successful BETA_L1 = 0.002 run (with seeds) ---
            top_neurons_to_interpret = [16, 10]  # Actual dominant neurons in this run
            k_texts = 5  # Number of top texts to show per neuron

            # (Optional: Can add back 25, 14 later if needed for comparison)
            # old_neurons = [25, 14]
            # for neuron_index in old_neurons: ...

            for neuron_index in top_neurons_to_interpret:
                if neuron_index >= num_neurons_check:
                    print(
                        f"\nSkipping Neuron {neuron_index} (index out of bounds for traces with {num_neurons_check} neurons)."
                    )
                    continue

                print(
                    f"\n--- Top {k_texts} texts activating Neuron {neuron_index} --- "
                )
                try:
                    # Get activations for the specific neuron across all samples
                    neuron_activations = all_memory_traces_tensor[:, neuron_index]

                    # Find the top k activations and their original indices
                    top_k_values, top_k_indices = torch.topk(
                        neuron_activations, k_texts
                    )

                    # Retrieve the corresponding texts
                    top_k_texts = [all_texts[i] for i in top_k_indices.tolist()]

                    # Print the texts and their activation values
                    for i in range(k_texts):
                        print(
                            f'  Activation: {top_k_values[i].item():.4f} - Text: "{top_k_texts[i]}"'
                        )

                except IndexError as e:
                    print(
                        f"Error accessing text for neuron {neuron_index}: {e}. Check alignment of texts and traces."
                    )
                except Exception as e:
                    print(
                        f"An unexpected error occurred during analysis for neuron {neuron_index}: {e}"
                    )

        except NameError:
            print(
                "Error: 'all_texts' variable not found. Cannot perform qualitative analysis."
            )
        except ValueError as e:
            print(e)
        except Exception as e:
            print(
                f"An unexpected error occurred preparing data for qualitative analysis: {e}"
            )


# --- 7e. Quantitative Category Similarity Analysis ---
print(f"\n--- 7e. Quantitative Category Similarity Analysis ---")

if not category_avg_activations or len(category_avg_activations) < len(CATEGORIES):
    print(
        "Average activations for all categories not available. Skipping similarity analysis."
    )
elif np is None:  # Check if numpy was imported successfully earlier
    print("Numpy not available. Skipping similarity analysis.")
else:
    try:
        # Ensure the order matches CATEGORIES for consistent matrix interpretation
        avg_vectors_list = [
            category_avg_activations[cat]
            for cat in CATEGORIES
            if cat in category_avg_activations
        ]
        present_categories = [
            cat for cat in CATEGORIES if cat in category_avg_activations
        ]

        if len(avg_vectors_list) < 2:
            print("Need average activations for at least two categories to compare.")
        else:
            avg_vectors = np.array(
                avg_vectors_list
            )  # Shape [num_categories, num_neurons]

            # --- Cosine Similarity Calculation (using numpy) ---
            print("\nCalculating Cosine Similarity...")
            # Normalize vectors
            norms = np.linalg.norm(avg_vectors, axis=1, keepdims=True)
            # Handle potential zero vectors (though unlikely with ReLU activations)
            norms[norms == 0] = 1e-10  # Avoid division by zero
            normalized_vectors = avg_vectors / norms
            # Calculate cosine similarity matrix
            cosine_sim_matrix = normalized_vectors @ normalized_vectors.T

            print("Cosine Similarity Matrix:")
            print(
                "        " + "   ".join([f"{cat[:5]:<5}" for cat in present_categories])
            )
            for i, cat_row in enumerate(present_categories):
                print(
                    f"{cat_row[:5]:<5}   "
                    + "   ".join(
                        [
                            f"{cosine_sim_matrix[i, j]:.4f}"
                            for j in range(len(present_categories))
                        ]
                    )
                )

            # --- Euclidean Distance Calculation (using numpy) ---
            print("\nCalculating Euclidean Distances...")
            distances = {}
            for i in range(len(present_categories)):
                for j in range(i + 1, len(present_categories)):
                    cat1 = present_categories[i]
                    cat2 = present_categories[j]
                    distance = np.linalg.norm(avg_vectors[i] - avg_vectors[j])
                    distances[f"{cat1}-{cat2}"] = distance
                    print(f"  Distance ({cat1} <-> {cat2}): {distance:.4f}")

    except Exception as e:
        print(f"An error occurred during similarity analysis: {e}")


print("\n--- Full Analysis Complete ---")

# (Old TODOs remain relevant - could add cosine sim etc. later)
# (TODO: Add visualization of activation profiles for text data)

# -------------------------------------\
# 8. Analysis & Retrieval (NEW SECTION)
# -------------------------------------\
print("\n--- Starting Analysis & Retrieval --- ")

model.eval()  # Set model to evaluation mode

# --- Calculate Full Memory Trace ---
print("Calculating full memory trace (neuron activations) for the dataset...")
with torch.no_grad():
    # Move all embeddings to the device at once for faster processing if memory allows
    all_embeddings_device = embeddings_tensor.to(device)
    _, all_encoded_activations, _ = model(all_embeddings_device)
    memory_trace = (
        all_encoded_activations.cpu()
    )  # Move back to CPU for analysis/indexing
print(
    f"Memory trace calculated. Shape: {memory_trace.shape}"
)  # Should be [num_samples, COMPRESSED_SIZE]


# --- Define Retrieval Functions ---
def retrieve_top_k_memories(trace, neuron_index, k=5):
    """Retrieves the indices of memories with the highest activation for a specific neuron."""
    if neuron_index < 0 or neuron_index >= trace.shape[1]:
        print(
            f"Error: Neuron index {neuron_index} out of bounds (0-{trace.shape[1]-1})."
        )
        return torch.tensor([], dtype=torch.long)
    neuron_activations = trace[:, neuron_index]
    top_k_values, top_k_indices = torch.topk(
        neuron_activations, k=min(k, len(trace))
    )  # Ensure k is not > trace length
    return top_k_indices, top_k_values


def retrieve_memories_composite(trace, positive_neurons, negative_neurons=[], k=5):
    """Retrieves indices of memories maximizing positive neuron activations while minimizing negative ones."""
    if not positive_neurons and not negative_neurons:
        print("Error: Must provide at least one positive or negative neuron index.")
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.float)

    # Ensure indices are valid
    all_indices = positive_neurons + negative_neurons
    max_idx = trace.shape[1] - 1
    if any(idx < 0 or idx > max_idx for idx in all_indices):
        print(f"Error: Neuron indices must be between 0 and {max_idx}.")
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.float)

    # Calculate positive score (average activation of positive neurons)
    if positive_neurons:
        # Handle single index case without error
        pos_indices_tensor = torch.tensor(positive_neurons, dtype=torch.long)
        pos_score = trace[:, pos_indices_tensor].mean(dim=1)
    else:
        pos_score = torch.zeros(trace.shape[0], dtype=trace.dtype)

    # Calculate negative penalty (average activation of negative neurons)
    if negative_neurons:
        # Handle single index case without error
        neg_indices_tensor = torch.tensor(negative_neurons, dtype=torch.long)
        neg_penalty = trace[:, neg_indices_tensor].mean(dim=1)
    else:
        neg_penalty = torch.zeros(trace.shape[0], dtype=trace.dtype)

    composite_score = pos_score - neg_penalty
    # Ensure k is not > trace length
    actual_k = min(k, len(trace))
    top_k_values, top_k_indices = torch.topk(composite_score, actual_k)
    return top_k_indices, top_k_values


def retrieve_external_crisis(trace, crisis_neuron, distress_neuron, threshold=0.5, k=5):
    """Retrieves indices of memories with high crisis neuron activation (above threshold)
    while penalizing distress neuron activation."""
    if (
        crisis_neuron < 0
        or crisis_neuron >= trace.shape[1]
        or distress_neuron < 0
        or distress_neuron >= trace.shape[1]
    ):
        print(f"Error: Neuron indices out of bounds (0-{trace.shape[1]-1}).")
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.float)

    # 1. Filter by crisis neuron threshold
    crisis_activation = trace[:, crisis_neuron]
    # Use .nonzero() which returns indices where condition is true
    valid_indices = (crisis_activation >= threshold).nonzero(as_tuple=True)[0]

    if valid_indices.numel() == 0:
        print(
            f"No memories found with Neuron {crisis_neuron} activation >= {threshold}"
        )
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.float)

    print(
        f"Found {valid_indices.numel()} memories with Neuron {crisis_neuron} >= {threshold}"
    )

    # 2. Calculate composite score for filtered indices
    filtered_trace = trace[valid_indices]
    scores = filtered_trace[:, crisis_neuron] - filtered_trace[:, distress_neuron]

    # 3. Find top k within the filtered set
    actual_k = min(k, scores.size(0))
    top_k_scores, top_k_relative_indices = torch.topk(scores, actual_k)

    # 4. Map relative indices back to original indices
    top_k_original_indices = valid_indices[top_k_relative_indices]

    return top_k_original_indices, top_k_scores


# --- Perform Retrieval (Single Neuron) ---
NEURON_16_INDEX = 16  # Personal Distress Specialist
NEURON_10_INDEX = 10  # Emotional Intensity & Crisis Indicator
TOP_K = 5

print(
    f"\nRetrieving top {TOP_K} memories for Neuron {NEURON_16_INDEX} (Personal Distress)...\n"
)
top_indices_n16, top_values_n16 = retrieve_top_k_memories(
    memory_trace, NEURON_16_INDEX, k=TOP_K
)

print(f"Retrieved indices: {top_indices_n16.tolist()}")
print("Associated activations:", [f"{v:.4f}" for v in top_values_n16.tolist()])
print("Corresponding Narratives:")
for idx in top_indices_n16:
    item = synthetic_data[idx.item()]  # Use .item() to get Python int
    print(f"  - Index {idx.item()} (Category: {item['category']}): {item['text']}")


print(
    f"\nRetrieving top {TOP_K} memories for Neuron {NEURON_10_INDEX} (Intensity/Crisis)...\n"
)
top_indices_n10, top_values_n10 = retrieve_top_k_memories(
    memory_trace, NEURON_10_INDEX, k=TOP_K
)

print(f"Retrieved indices: {top_indices_n10.tolist()}")
print("Associated activations:", [f"{v:.4f}" for v in top_values_n10.tolist()])
print("Corresponding Narratives:")
for idx in top_indices_n10:
    item = synthetic_data[idx.item()]  # Use .item() to get Python int
    print(f"  - Index {idx.item()} (Category: {item['category']}): {item['text']}")

# --- Perform Composite Retrieval ---
print("\n--- Performing Composite Queries ---")

# Query 1: Personal Distress without External Crisis (High N16, Low N10)
print(
    f"\nRetrieving top {TOP_K} memories for High N16 / Low N10 (Personal Distress Focus)..."
)
top_indices_p16_n10, top_scores_p16_n10 = retrieve_memories_composite(
    memory_trace,
    positive_neurons=[NEURON_16_INDEX],
    negative_neurons=[NEURON_10_INDEX],
    k=TOP_K,
)
print(f"Retrieved indices: {top_indices_p16_n10.tolist()}")
print("Associated composite scores:", [f"{v:.4f}" for v in top_scores_p16_n10.tolist()])
print("Corresponding Narratives:")
for idx in top_indices_p16_n10:
    item = synthetic_data[idx.item()]
    # Show N16 and N10 activations for clarity
    n16_act = memory_trace[idx.item(), NEURON_16_INDEX].item()
    n10_act = memory_trace[idx.item(), NEURON_10_INDEX].item()
    print(
        f"  - Index {idx.item()} (Cat: {item['category']}, N16: {n16_act:.3f}, N10: {n10_act:.3f}): {item['text']}"
    )

# Query 2: External Crisis without Deep Personal Distress (High N10 > threshold, Low N16)
print(
    f"\nRetrieving top {TOP_K} memories for Refined External Crisis (N10 >= 0.5, penalized by N16)..."
)
CRISIS_THRESHOLD = 0.5  # Define the threshold
top_indices_crisis_refined, top_scores_crisis_refined = retrieve_external_crisis(
    memory_trace,
    crisis_neuron=NEURON_10_INDEX,
    distress_neuron=NEURON_16_INDEX,
    threshold=CRISIS_THRESHOLD,
    k=TOP_K,
)

if top_indices_crisis_refined.numel() > 0:
    print(f"Retrieved indices: {top_indices_crisis_refined.tolist()}")
    print(
        "Associated composite scores (N10 - N16, for N10 >= threshold):",
        [f"{v:.4f}" for v in top_scores_crisis_refined.tolist()],
    )
    print("Corresponding Narratives:")
    for idx in top_indices_crisis_refined:
        item = synthetic_data[idx.item()]
        n16_act = memory_trace[idx.item(), NEURON_16_INDEX].item()
        n10_act = memory_trace[idx.item(), NEURON_10_INDEX].item()
        print(
            f"  - Index {idx.item()} (Cat: {item['category']}, N16: {n16_act:.3f}, N10: {n10_act:.3f}): {item['text']}"
        )
else:
    print("No memories met the refined external crisis criteria.")

# Query 3: Routine-like (Low N10 and Low N16) - Maximize the NEGATIVE of their sum
print(f"\nRetrieving top {TOP_K} memories for Low N10 & Low N16 (Routine Focus)...")
# We want the MINIMUM sum of N10 and N16. TopK finds maximums.
# So, we find the maximum of the NEGATIVE of their average activation.
# Effectively, retrieve_memories_composite with only negative neurons maximizes the score = 0 - neg_penalty
top_indices_routine, top_scores_routine = retrieve_memories_composite(
    memory_trace,
    positive_neurons=[],  # No positive contribution
    negative_neurons=[NEURON_10_INDEX, NEURON_16_INDEX],
    k=TOP_K,
)
# Note: The score here is -(Avg(N10, N16)), so higher score means lower activation
print(f"Retrieved indices: {top_indices_routine.tolist()}")
print(
    "Associated composite scores (higher means lower N10/N16 activation):",
    [f"{v:.4f}" for v in top_scores_routine.tolist()],
)
print("Corresponding Narratives:")
for idx in top_indices_routine:
    item = synthetic_data[idx.item()]
    n16_act = memory_trace[idx.item(), NEURON_16_INDEX].item()
    n10_act = memory_trace[idx.item(), NEURON_10_INDEX].item()
    print(
        f"  - Index {idx.item()} (Cat: {item['category']}, N16: {n16_act:.3f}, N10: {n10_act:.3f}): {item['text']}"
    )

# -------------------------------------\
# 9. Further Analysis (Placeholder for original analysis plots if any)
# -------------------------------------\
print("\n--- Analysis/Retrieval Complete --- ")

# --- Visualization Section ---
print("\n--- Generating PCA Visualization --- ")

# Check if memory_trace exists and has data
if "memory_trace" in locals() and memory_trace.shape[0] > 0:
    # 1. Perform PCA
    print(f"Performing PCA on memory trace (shape: {memory_trace.shape})...")
    pca = PCA(n_components=2)
    # PCA expects numpy array, ensure memory_trace is CPU tensor then convert
    memory_trace_np = (
        memory_trace.numpy() if isinstance(memory_trace, torch.Tensor) else memory_trace
    )
    pca_result = pca.fit_transform(memory_trace_np)
    print(f"PCA completed. Result shape: {pca_result.shape}")

    # 2. Create Scatter Plot
    plt.figure(figsize=(12, 8))
    scatter_handles = []  # For custom legend
    colors = plt.cm.viridis(np.linspace(0, 1, len(CATEGORIES)))

    # Ensure category_indices and category_map are available from data prep section
    if "category_indices" in locals() and "category_map" in locals():
        category_list = [CATEGORIES[i] for i in category_indices.tolist()]
        for i, category_name in enumerate(CATEGORIES):
            # Find indices belonging to the current category
            category_mask = category_indices == category_map[category_name]
            # Plot points for this category
            scatter = plt.scatter(
                pca_result[category_mask, 0],
                pca_result[category_mask, 1],
                color=colors[i],
                label=category_name,
                alpha=0.6,  # Add some transparency
                s=20,  # Adjust point size
            )
            scatter_handles.append(scatter)

        plt.title("PCA of Narrative SAE Activations (Colored by Category)")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        # plt.legend(handles=scatter_handles, title="Categories") # Modified legend below
        plt.grid(True, linestyle="--", alpha=0.5)

        # --- Highlight Retrieved Points ---
        highlight_markers = ["X", "P", "s"]  # Markers for the 3 queries
        highlight_colors = ["red", "lime", "blue"]  # Edge colors
        highlight_labels = [
            "Personal Distress Focus (Top 5)",
            "Refined External Crisis (Top 5)",
            "Routine Focus (Top 5)",
        ]
        highlight_indices_list = []

        # Check if index variables exist before trying to use them
        if "top_indices_p16_n10" in locals():
            highlight_indices_list.append(top_indices_p16_n10)
        else:
            highlight_indices_list.append(None)

        if "top_indices_crisis_refined" in locals():
            highlight_indices_list.append(top_indices_crisis_refined)
        else:
            highlight_indices_list.append(None)

        if "top_indices_routine" in locals():
            highlight_indices_list.append(top_indices_routine)
        else:
            highlight_indices_list.append(None)

        for i, indices_tensor in enumerate(highlight_indices_list):
            if indices_tensor is not None and indices_tensor.numel() > 0:
                indices = indices_tensor.tolist()
                # Ensure indices are within bounds of pca_result
                valid_indices = [idx for idx in indices if idx < pca_result.shape[0]]
                if valid_indices:
                    h_scatter = plt.scatter(
                        pca_result[valid_indices, 0],
                        pca_result[valid_indices, 1],
                        marker=highlight_markers[i],
                        s=150,  # Larger size
                        edgecolor=highlight_colors[i],
                        facecolor="none",  # No fill
                        linewidth=1.5,
                        label=highlight_labels[i],
                    )
                    scatter_handles.append(h_scatter)  # Add to legend items
                else:
                    print(
                        f"Warning: Indices for '{highlight_labels[i]}' out of bounds or empty after validation."
                    )
            elif indices_tensor is None:
                print(
                    f"Warning: Indices variable for '{highlight_labels[i]}' not found."
                )
            # No need to print if numel is 0, retrieval function already warns

        # Update legend to include highlight markers
        plt.legend(
            handles=scatter_handles, title="Categories & Queries", fontsize="small"
        )

        # 3. Save the Plot
        pca_plot_path = "pca_narrative_activations_highlighted.png"  # New filename
        try:
            plt.savefig(pca_plot_path)
            print(f"Saved highlighted PCA scatter plot to {pca_plot_path}")
        except Exception as e:
            print(f"Error saving highlighted PCA plot: {e}")
        # plt.show() # Optional: Display plot

    else:
        print(
            "Error: category_indices or category_map not found. Cannot create category plot."
        )

else:
    print("Skipping PCA visualization: memory_trace not found or empty.")


# (Original analysis code like plotting category centroids, sparsity, etc., would go here if needed)
# Example: Plotting loss history (if training occurred)
if not skip_training and "epoch_loss_history" in locals() and epoch_loss_history:
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_loss_history, label="Total Fine-tuning Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Fine-tuning Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("finetuning_loss_curve.png")
    print("Saved fine-tuning loss curve to finetuning_loss_curve.png")
    # plt.show() # Optional: display plot

# Optional: Add code for composite queries or other analyses here later
# ...

# Optional: Add code to evaluate cosine similarity, Euclidean distance, etc.
# ...
