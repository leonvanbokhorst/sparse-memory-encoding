import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt
import json  # To load the generated data
import os

# Ensure reproducibility if needed (optional)
# torch.manual_seed(42)
# np.random.seed(42)


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
BETA_L1 = 0.0005  # Sparsity strength (can tune)
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

print(f"Loading pre-trained state dictionary from: {PRETRAINED_MODEL_PATH}")
try:
    state_dict = torch.load(PRETRAINED_MODEL_PATH, map_location="cpu")
    model.load_state_dict(
        state_dict, strict=False
    )  # Load encoder/decoder, ignore projection
    print("Loaded pre-trained encoder/decoder weights.")
except Exception as e:
    print(
        f"Warning: Could not load pre-trained weights from {PRETRAINED_MODEL_PATH}. Training from scratch. Error: {e}"
    )

model.to(device)  # Move entire model to target device
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion_recon = nn.MSELoss(reduction="none")  # For weighted loss

# -------------------------------------
# 5. Fine-tuning Loop (with DataLoader)
# -------------------------------------
print(f"\n--- Starting Fine-tuning for {FINETUNE_EPOCHS} Epochs --- ")
epoch_loss_history = []

model.train()  # Set model to training mode
for epoch in range(FINETUNE_EPOCHS):
    epoch_total_loss = 0.0
    epoch_recon_loss = 0.0
    epoch_sparsity_loss = 0.0
    num_batches = 0

    for batch_embeddings, batch_recon_weights, batch_sparsity_weights in data_loader:
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

    if epoch % 10 == 0 or epoch == FINETUNE_EPOCHS - 1:  # Print more frequently maybe
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
        results_finetuned[category] = {
            "texts": category_texts,
            "memory_traces": memory_traces.cpu().numpy(),
        }

# --- Analysis Code ---
print(f"Analyzing neurons originally identified (from sine): {NEURONS_TO_ANALYZE}")
print(
    "(Sine Core={CORE_NEURONS_SINE}, Sine Unique Emo={UNIQUE_EMO_NEURONS_SINE}, Sine Unique Routine={UNIQUE_ROUTINE_NEURONS_SINE})"
)

activation_summary = {}
for category, data in results_finetuned.items():
    print(f"\n{category} Texts (Post Fine-tuning, {len(data['texts'])} samples):")
    traces = data["memory_traces"]
    # Ensure we handle cases where NEURONS_TO_ANALYZE might be empty or invalid
    if NEURONS_TO_ANALYZE and traces.shape[1] > max(NEURONS_TO_ANALYZE):
        avg_activations = np.mean(traces[:, NEURONS_TO_ANALYZE], axis=0)
        activation_summary[category] = avg_activations
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
    "\nNote: Roles of neurons likely shifted during fine-tuning. Full analysis needed."
)
print("Analysis Complete. Check if neuron roles align with text categories.")

# (TODO: Add de novo analysis - e.g., find top k active neurons per category)
# (TODO: Add visualization of activation profiles for text data)
