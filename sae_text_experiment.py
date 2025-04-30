import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import numpy as np


# -------------------------------------
# 1. Modified Model Definition with Projection Layer
# -------------------------------------
class SparseMemoryText(nn.Module):
    def __init__(self, embedding_dim, event_size, compressed_size):
        super().__init__()
        # New layer to project text embedding dim to SAE's expected event_size
        self.projection = nn.Linear(embedding_dim, event_size)
        # Original SAE layers
        self.encoder = nn.Linear(event_size, compressed_size)
        self.decoder = nn.Linear(compressed_size, event_size)
        print(
            f"Initialized SparseMemoryText: Projection [{embedding_dim}->{event_size}], Encoder [{event_size}->{compressed_size}], Decoder [{compressed_size}->{event_size}]"
        )

    def forward(self, text_embedding):
        # Ensure input is float
        text_embedding = text_embedding.float()
        # Project text embedding down to the SAE's input size
        projected = torch.relu(self.projection(text_embedding))
        # Pass through the original SAE encoder
        encoded = torch.relu(self.encoder(projected))
        # Pass through the original SAE decoder
        decoded = torch.sigmoid(self.decoder(encoded.float()))
        # Return the final reconstruction, but also the intermediate encoded trace for analysis
        return decoded, encoded


# -------------------------------------
# 2. Configuration
# -------------------------------------
# Sentence embedding model details
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
# Dimensions must match the SentenceTransformer model output
EMBEDDING_DIM = 384

# SAE dimensions (must match the saved model)
EVENT_SIZE = 300  # The dimension the original SAE was trained on
COMPRESSED_SIZE = 30  # The dimension of the sparse code

# Path to the saved model weights from the optimal run
SAVED_MODEL_PATH = "sae_model_beta0005.pth"

# Key neurons identified from the BETA_L1=0.0005 run analysis (using threshold ~0.01)
# (Ensure these indices are correct based on the final analysis of that run)
CORE_NEURONS = [15, 21]
UNIQUE_EMO_NEURONS = [5]  # Example: Neuron 5 was unique to Emotional
UNIQUE_ROUTINE_NEURONS = [9, 28]  # Example: Neurons 9, 28 unique to Routine
# Add other interesting shared neurons if desired
SHARED_EMO_ROUTINE_NEURONS = [0, 7, 17, 25, 26]  # Excluding core

NEURONS_TO_ANALYZE = sorted(
    list(
        set(
            CORE_NEURONS
            + UNIQUE_EMO_NEURONS
            + UNIQUE_ROUTINE_NEURONS
            + SHARED_EMO_ROUTINE_NEURONS
        )
    )
)

# --- Determine Device ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device found. Using MPS.")
elif torch.cuda.is_available():  # Fallback for CUDA if needed
    device = torch.device("cuda")
    print("CUDA device found. Using CUDA.")
else:
    device = torch.device("cpu")
    print("MPS/CUDA not found. Using CPU.")

# -------------------------------------
# 3. Load Models (Sentence Transformer & SAE)
# -------------------------------------
print(f"Loading sentence transformer model: {EMBEDDING_MODEL_NAME}...")
# Load embedder to the determined device if possible (some models benefit)
try:
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
except TypeError:  # Older sentence-transformer versions might not accept device arg
    print(
        "Warning: Could not set device for SentenceTransformer directly. Ensure tensors are moved manually."
    )
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Sentence transformer loaded.")

print("Initializing SAE model structure...")
sae_text_model = SparseMemoryText(
    embedding_dim=EMBEDDING_DIM, event_size=EVENT_SIZE, compressed_size=COMPRESSED_SIZE
)

print(f"Loading saved SAE state dictionary from: {SAVED_MODEL_PATH}...")
try:
    # Load state dict to CPU first is often safer, then move model
    state_dict = torch.load(SAVED_MODEL_PATH, map_location=torch.device("cpu"))
    sae_text_model.load_state_dict(state_dict, strict=False)
    print("Successfully loaded weights into encoder and decoder layers.")
except FileNotFoundError:
    print(
        f"Error: Model file not found at {SAVED_MODEL_PATH}. Please ensure the training script saved the model."
    )
    exit()
except Exception as e:
    print(f"Error loading model state_dict: {e}")
    exit()

# --- Move model to the target device ---
print(f"Moving SAE model to device: {device}")
sae_text_model.to(device)

# Set the model to evaluation mode
sae_text_model.eval()

# -------------------------------------
# 4. Define Textual Narrative Samples
# -------------------------------------
# Examples analogous to sine wave categories
text_samples = {
    "Strong Emotional": [
        "She broke down, finally confronting the grief she'd held back for years.",
        "A wave of pure joy washed over him as he saw them approach.",
        "The betrayal hit him like a physical blow, stealing his breath.",
    ],
    "Complex/Chaotic": [
        "Their argument spiraled rapidly, misunderstandings compounding each other.",
        "The meeting dissolved into confusion, multiple people talking over each other.",
        "Navigating the crowded market felt like swimming against a chaotic tide.",
    ],
    "Routine": [
        "He woke up, poured coffee, and checked his messages before starting work.",
        "The usual Tuesday morning meeting covered sales figures and upcoming deadlines.",
        "She walked the familiar path home, the evening quiet and uneventful.",
    ],
}

# -------------------------------------
# 5. Encode Texts & Run Through SAE
# -------------------------------------
print("\n--- Processing Text Samples ---")
results = {}

with torch.no_grad():
    for category, texts in text_samples.items():
        print(f"\nProcessing Category: {category}")
        # Encode texts - specify convert_to_tensor=False initially if moving manually
        # Let SentenceTransformer handle device placement if possible, otherwise move manually
        embeddings_np = embedder.encode(texts)
        embeddings = torch.tensor(embeddings_np).to(
            device
        )  # Ensure tensor is on the correct device
        print(
            f"  Encoded {len(texts)} texts. Embeddings tensor on device: {embeddings.device}"
        )

        # Pass embeddings through the SAE (model is already on the correct device)
        reconstructed_embeddings, memory_traces = sae_text_model(embeddings)
        print(
            f"  Generated memory traces of shape: {memory_traces.shape}. Output tensor on device: {memory_traces.device}"
        )

        results[category] = {
            "texts": texts,
            "embeddings": embeddings.cpu().numpy(),  # Move back to CPU for storage if needed
            "memory_traces": memory_traces.cpu().numpy(),  # Move back to CPU for storage/analysis
        }

# -------------------------------------
# 6. Analyze Activations of Key Neurons
# -------------------------------------
print("\n--- Analyzing Key Neuron Activations for Text Categories ---")

print(f"Analyzing neurons: {NEURONS_TO_ANALYZE}")
print(
    "(Core = {CORE_NEURONS}, Unique Emo = {UNIQUE_EMO_NEURONS}, Unique Routine = {UNIQUE_ROUTINE_NEURONS}, Shared Emo/Routine = {SHARED_EMO_ROUTINE_NEURONS})"
)

for category, data in results.items():
    print(f"\n{category} Texts:")
    traces = data["memory_traces"]  # Shape: [num_samples, compressed_size]
    avg_activations = np.mean(traces[:, NEURONS_TO_ANALYZE], axis=0)

    print(f"  Avg Activations for Analyzed Neurons (Indices: {NEURONS_TO_ANALYZE}):")
    print(f"  {[f'{act:.4f}' for act in avg_activations]}")

    # Optionally print activations for specific neurons of interest
    if CORE_NEURONS:
        core_indices_in_analysis = [
            NEURONS_TO_ANALYZE.index(n) for n in CORE_NEURONS if n in NEURONS_TO_ANALYZE
        ]
        if core_indices_in_analysis:
            print(
                f"    Avg Core ({CORE_NEURONS}) Activations: {[f'{avg_activations[i]:.4f}' for i in core_indices_in_analysis]}"
            )
    if UNIQUE_EMO_NEURONS:
        unique_emo_indices = [
            NEURONS_TO_ANALYZE.index(n)
            for n in UNIQUE_EMO_NEURONS
            if n in NEURONS_TO_ANALYZE
        ]
        if unique_emo_indices:
            print(
                f"    Avg Unique Emo ({UNIQUE_EMO_NEURONS}) Activations: {[f'{avg_activations[i]:.4f}' for i in unique_emo_indices]}"
            )
    if UNIQUE_ROUTINE_NEURONS:
        unique_routine_indices = [
            NEURONS_TO_ANALYZE.index(n)
            for n in UNIQUE_ROUTINE_NEURONS
            if n in NEURONS_TO_ANALYZE
        ]
        if unique_routine_indices:
            print(
                f"    Avg Unique Routine ({UNIQUE_ROUTINE_NEURONS}) Activations: {[f'{avg_activations[i]:.4f}' for i in unique_routine_indices]}"
            )

    # Add similar printouts for Chaotic unique/shared if they exist and are included
    # ...

print("\nAnalysis Complete. Compare average activations across categories.")
