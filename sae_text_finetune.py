import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt

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
FINETUNED_MODEL_SAVE_PATH = "sae_text_finetuned.pth"

# --- Fine-tuning Hyperparameters ---
FINETUNE_EPOCHS = 200  # Adjust as needed
LEARNING_RATE = 1e-4  # Lower LR for fine-tuning
BETA_L1 = 0.0005  # Sparsity strength (can tune)

# --- Analysis Configuration ---
# Define key neurons based on sine-wave analysis for post-tuning check
# (Ensure these are the correct ones from the BETA_L1=0.0005 run)
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
ANALYSIS_THRESHOLD = 0.01  # Threshold for neuron activation analysis

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
print("Preparing textual data...")
# Expand this dataset significantly for robust fine-tuning
text_samples_dict = {
    "Strong Emotional": [
        "She broke down, finally confronting the grief she'd held back for years.",
        "A wave of pure joy washed over him as he saw them approach.",
        "The betrayal hit him like a physical blow, stealing his breath.",
        "He felt an overwhelming sense of peace watching the sunset.",
        "Panic surged as the deadline loomed impossibly close.",
        # Add more emotional examples (aim for 20+)
    ],
    "Complex/Chaotic": [
        "Their argument spiraled rapidly, misunderstandings compounding each other.",
        "The meeting dissolved into confusion, multiple people talking over each other.",
        "Navigating the crowded market felt like swimming against a chaotic tide.",
        "The conflicting instructions left the team completely paralyzed.",
        "Untangling the web of lies required careful, painstaking effort.",
        # Add more chaotic examples (aim for 20+)
    ],
    "Routine": [
        "He woke up, poured coffee, and checked his messages before starting work.",
        "The usual Tuesday morning meeting covered sales figures and upcoming deadlines.",
        "She walked the familiar path home, the evening quiet and uneventful.",
        "Making breakfast was a simple, everyday ritual.",
        "The commute was predictable, same traffic, same route.",
        # Add more routine examples (aim for 20+)
    ],
}

# --- Encode Texts ---
print(f"Loading sentence transformer: {EMBEDDING_MODEL_NAME}")
try:
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
except TypeError:
    print("Warning: Could not set device for SentenceTransformer directly.")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

all_texts = []
labels = []  # Store category names for potential analysis
category_indices = []  # Store numerical index for weighting later
category_map = {name: i for i, name in enumerate(text_samples_dict.keys())}

for category, texts in text_samples_dict.items():
    all_texts.extend(texts)
    labels.extend([category] * len(texts))
    category_indices.extend([category_map[category]] * len(texts))

print(f"Encoding {len(all_texts)} text samples...")
embeddings_np = embedder.encode(all_texts)
embeddings_tensor = torch.tensor(embeddings_np).float().to(device)
category_indices_tensor = torch.tensor(category_indices).long().to(device)
print(
    f"Embeddings tensor shape: {embeddings_tensor.shape}, Device: {embeddings_tensor.device}"
)

# --- Define Importance Weights based on Categories ---
# Example: Emotional=3.0, Chaotic=2.0, Routine=1.0 for reconstruction
recon_weights_map = {
    category_map["Strong Emotional"]: 3.0,
    category_map["Complex/Chaotic"]: 2.0,
    category_map["Routine"]: 1.0,
}
recon_weights = (
    torch.tensor([recon_weights_map[i.item()] for i in category_indices_tensor])
    .float()
    .to(device)
)

# Example: Emotional=0.3, Chaotic=0.5, Routine=1.0 for L1 sparsity
sparsity_weights_map = {
    category_map["Strong Emotional"]: 0.3,
    category_map["Complex/Chaotic"]: 0.5,
    category_map["Routine"]: 1.0,
}
sparsity_weights = (
    torch.tensor([sparsity_weights_map[i.item()] for i in category_indices_tensor])
    .float()
    .unsqueeze(1)
    .to(device)
)  # Shape [N, 1]

# (Optional: Split into Train/Validation Sets Here if desired)
# For simplicity, we'll fine-tune on the whole small dataset for now

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
# 5. Fine-tuning Loop
# -------------------------------------
print(f"\n--- Starting Fine-tuning for {FINETUNE_EPOCHS} Epochs --- ")
loss_history = []

model.train()  # Set model to training mode
for epoch in range(FINETUNE_EPOCHS):
    optimizer.zero_grad()

    # Model now returns decoded, encoded, projected
    reconstructed, encoded, projected = model(embeddings_tensor)

    # --- Weighted Reconstruction Loss ---
    # Compare reconstructed (decoder output) with projected (encoder input)
    element_wise_recon_loss = criterion_recon(
        reconstructed, projected
    )  # Target is projected
    recon_loss_per_episode = torch.mean(element_wise_recon_loss, dim=1)
    weighted_recon_loss = (recon_loss_per_episode * recon_weights).mean()

    # --- Weighted L1 Sparsity Loss ---
    l1_norm_per_episode = torch.norm(encoded, p=1, dim=1, keepdim=True)
    weighted_l1_per_episode = (
        l1_norm_per_episode * sparsity_weights
    )  # sparsity_weights is already [N, 1]
    sparsity_loss = BETA_L1 * torch.mean(weighted_l1_per_episode)

    # --- Total Loss ---
    total_loss = weighted_recon_loss + sparsity_loss

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=1.0
    )  # Optional gradient clipping
    optimizer.step()

    loss_history.append(total_loss.item())
    if epoch % 20 == 0 or epoch == FINETUNE_EPOCHS - 1:
        print(
            f"Epoch {epoch:4d}/{FINETUNE_EPOCHS}, Total Loss: {total_loss.item():.6f}, Recon Loss (Weighted): {weighted_recon_loss.item():.6f}, Sparsity Loss: {sparsity_loss.item():.6f}"
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

with torch.no_grad():
    for category, texts in text_samples_dict.items():
        embeddings_np = embedder.encode(texts)
        embeddings = torch.tensor(embeddings_np).float().to(device)
        # We only need the encoded part for analysis here
        _, memory_traces, _ = model(
            embeddings
        )  # Get memory traces from fine-tuned model
        results_finetuned[category] = {
            "texts": texts,
            "memory_traces": memory_traces.cpu().numpy(),
        }

# --- Analysis Code (Similar to section 6 of sae_text_experiment.py) ---
print(f"Analyzing neurons originally identified: {NEURONS_TO_ANALYZE}")
print(
    "(Core = {CORE_NEURONS_SINE}, Unique Emo = {UNIQUE_EMO_NEURONS_SINE}, Unique Routine = {UNIQUE_ROUTINE_NEURONS_SINE})"
)

activation_summary = {}
for category, data in results_finetuned.items():
    print(f"\n{category} Texts (Post Fine-tuning):")
    traces = data["memory_traces"]
    avg_activations = np.mean(traces[:, NEURONS_TO_ANALYZE], axis=0)
    activation_summary[category] = avg_activations

    print(
        f"  Avg Activations ({NEURONS_TO_ANALYZE}): {[f'{act:.4f}' for act in avg_activations]}"
    )

    # Print specific groups
    core_indices = [
        NEURONS_TO_ANALYZE.index(n)
        for n in CORE_NEURONS_SINE
        if n in NEURONS_TO_ANALYZE
    ]
    if core_indices:
        print(
            f"    Avg Core ({CORE_NEURONS_SINE}) Activations: {[f'{avg_activations[i]:.4f}' for i in core_indices]}"
        )

    unique_emo_indices = [
        NEURONS_TO_ANALYZE.index(n)
        for n in UNIQUE_EMO_NEURONS_SINE
        if n in NEURONS_TO_ANALYZE
    ]
    if unique_emo_indices:
        print(
            f"    Avg Unique Emo ({UNIQUE_EMO_NEURONS_SINE}) Activations: {[f'{avg_activations[i]:.4f}' for i in unique_emo_indices]}"
        )

    unique_routine_indices = [
        NEURONS_TO_ANALYZE.index(n)
        for n in UNIQUE_ROUTINE_NEURONS_SINE
        if n in NEURONS_TO_ANALYZE
    ]
    if unique_routine_indices:
        print(
            f"    Avg Unique Routine ({UNIQUE_ROUTINE_NEURONS_SINE}) Activations: {[f'{avg_activations[i]:.4f}' for i in unique_routine_indices]}"
        )

print("\nAnalysis Complete. Check if neuron roles align with text categories.")

# (Optional: Add visualization of activation profiles for text data)
