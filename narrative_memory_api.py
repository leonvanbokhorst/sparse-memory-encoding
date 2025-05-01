import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import json
import os
import numpy as np

# --- Need the SAE Model Definition ---
# Option 1: Redefine it here (simpler for now)
# Option 2: Move it to a separate 'models.py' and import it (better for larger projects)


class SparseMemoryText(nn.Module):
    # Basic definition - copy from sae_text_finetune.py if needed
    # Or ideally, ensure it's importable from elsewhere
    def __init__(self, embedding_dim, event_size, compressed_size):
        super().__init__()
        # Simplified for brevity, assume layers are defined
        self.projection = nn.Linear(embedding_dim, event_size)
        self.encoder = nn.Linear(event_size, compressed_size)
        self.decoder = nn.Linear(compressed_size, event_size)
        print(
            f"SAE Model Structure: Proj [{embedding_dim}->{event_size}], Enc [{event_size}->{compressed_size}]"
        )

    def forward(self, text_embedding):
        text_embedding = text_embedding.float()
        projected = torch.relu(self.projection(text_embedding))
        encoded = torch.relu(self.encoder(projected))
        # For API, we mostly care about 'encoded' (activations)
        # Decoder might be needed if we want reconstruction features later
        # decoded = torch.sigmoid(self.decoder(encoded.float()))
        return encoded  # Return only encoded for activation analysis


# --- Narrative Memory API Class ---

# Key Neuron Roles (for default sae_text_finetuned.pth with BETA_L1=0.001):
# Neuron 4: Emotional Intensity / Salience / Confrontation / Loss
# Neuron 13: Structured Thought / Scene Description / Planning / Complexity
# Neuron 27: External Chaos / Overwhelm / Systemic Conflict / Disorder


class NarrativeMemoryAPI:
    def __init__(
        self,
        model_path="sae_text_finetuned.pth",
        data_path="processed_batches/results_batch_68132475ee8481908e4db8b9c4605c0c.json",  # NEW Default
        embedding_model_name="all-MiniLM-L6-v2",
        embedding_dim=384,  # From config
        event_size=300,  # From config
        compressed_size=30,  # From config
        device=None,
    ):
        """Initializes the API, loads model, data, and computes memory trace."""
        print("--- Initializing Narrative Memory API --- ")
        self.model_path = model_path
        self.data_path = data_path
        self.embedding_model_name = embedding_model_name
        self.embedding_dim = embedding_dim
        self.event_size = event_size
        self.compressed_size = compressed_size

        # --- Device Setup ---
        if device:
            self.device = torch.device(device)
            print(f"Using specified device: {self.device}")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("MPS device found. Using MPS.")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("CUDA device found. Using CUDA.")
        else:
            self.device = torch.device("cpu")
            print("MPS/CUDA not found. Using CPU.")

        # --- Load Components ---
        self.narratives = self._load_narrative_data()
        self.embedder = self._load_embedder()
        self.model = self._load_sae_model()
        self.memory_trace = self._compute_memory_trace()

        print(f"API Initialized. Memory Trace shape: {self.memory_trace.shape}")
        print("-----------------------------------------")

    def _load_narrative_data(self):
        """Loads narrative text and metadata from the JSON file."""
        print(f"Loading narrative data from: {self.data_path}...")
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Narrative data file not found at {self.data_path}"
            )
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"Loaded {len(data)} narrative entries.")
            # Store as list of dicts for easy access to text/category
            return data
        except Exception as e:
            raise IOError(f"Error loading or parsing narrative data: {e}")

    def _load_embedder(self):
        """Loads the sentence transformer model."""
        print(f"Loading sentence transformer: {self.embedding_model_name}...")
        try:
            # Attempt to load directly to the target device if supported
            embedder = SentenceTransformer(
                self.embedding_model_name, device=self.device
            )
        except TypeError:
            print(
                f"Warning: Could not set device '{self.device}' for SentenceTransformer directly. Loading to default device."
            )
            embedder = SentenceTransformer(self.embedding_model_name)
            # Manually move model if necessary (less common now, but good fallback)
            # embedder.to(self.device)
        print("Sentence transformer loaded.")
        return embedder

    def _load_sae_model(self):
        """Loads the fine-tuned Sparse Autoencoder model."""
        print(f"Loading fine-tuned SAE model from: {self.model_path}...")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"SAE model file not found at {self.model_path}")

        model = SparseMemoryText(
            embedding_dim=self.embedding_dim,
            event_size=self.event_size,
            compressed_size=self.compressed_size,
        )
        try:
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model.to(self.device)
            model.eval()  # Set to evaluation mode
            print("SAE model loaded successfully.")
            return model
        except Exception as e:
            raise IOError(f"Error loading SAE model state dict: {e}")

    def _compute_memory_trace(self):
        """Computes SAE activations (memory trace) for all loaded narratives."""
        print("Computing full memory trace...")
        if not self.narratives or not self.embedder or not self.model:
            raise RuntimeError("Cannot compute trace: API components not loaded.")

        all_texts = [item["text"] for item in self.narratives]
        print(f"Encoding {len(all_texts)} texts...")
        # Ensure embeddings are created on the correct device potentially
        embeddings_np = self.embedder.encode(
            all_texts, show_progress_bar=True, device=self.device
        )
        embeddings_tensor = torch.tensor(embeddings_np).float().to(self.device)

        print("Calculating activations with SAE model...")
        with torch.no_grad():
            memory_trace_device = self.model(embeddings_tensor)
            memory_trace_cpu = (
                memory_trace_device.cpu()
            )  # Move to CPU for consistent indexing/analysis
        print("Memory trace computation complete.")
        return memory_trace_cpu

    # --- Retrieval Methods ---

    def get_narrative_text(self, index):
        """Returns the text of the narrative at the given index."""
        try:
            return self.narratives[index]["text"]
        except IndexError:
            print(f"Error: Index {index} out of bounds.")
            return None
        except Exception as e:
            print(f"Error retrieving narrative text: {e}")
            return None

    def retrieve_top_k_memories(self, neuron_index, k=5):
        """Retrieves the indices and activations of memories with the highest activation for a specific neuron."""
        print(f"\nRetrieving top {k} for Neuron {neuron_index}...")
        if neuron_index < 0 or neuron_index >= self.compressed_size:
            print(
                f"Error: Neuron index {neuron_index} out of bounds (0-{self.compressed_size-1})."
            )
            return torch.tensor([], dtype=torch.long), torch.tensor(
                [], dtype=torch.float
            )

        neuron_activations = self.memory_trace[:, neuron_index]
        actual_k = min(k, len(self.memory_trace))
        top_k_values, top_k_indices = torch.topk(neuron_activations, k=actual_k)

        # Optional: Print results here or return them for external handling
        # print(f"  Indices: {top_k_indices.tolist()}")
        # print(f"  Activations: {[f'{v:.4f}' for v in top_k_values.tolist()]}"")
        # for idx in top_k_indices:
        #     print(f"    - {self.get_narrative_text(idx.item())}")

        return top_k_indices, top_k_values

    def retrieve_memories_composite(self, positive_neurons, negative_neurons=[], k=5):
        """Retrieves indices maximizing positive neuron activations while minimizing negative ones."""
        print(
            f"\nRetrieving top {k} composite: Pos={positive_neurons}, Neg={negative_neurons}..."
        )
        if not positive_neurons and not negative_neurons:
            print("Error: Must provide at least one positive or negative neuron index.")
            return torch.tensor([], dtype=torch.long), torch.tensor(
                [], dtype=torch.float
            )

        all_indices = positive_neurons + negative_neurons
        if any(idx < 0 or idx >= self.compressed_size for idx in all_indices):
            print(
                f"Error: Neuron indices must be between 0 and {self.compressed_size-1}."
            )
            return torch.tensor([], dtype=torch.long), torch.tensor(
                [], dtype=torch.float
            )

        if positive_neurons:
            pos_indices_tensor = torch.tensor(positive_neurons, dtype=torch.long)
            pos_score = self.memory_trace[:, pos_indices_tensor].mean(dim=1)
        else:
            pos_score = torch.zeros(
                len(self.memory_trace), dtype=self.memory_trace.dtype
            )

        if negative_neurons:
            neg_indices_tensor = torch.tensor(negative_neurons, dtype=torch.long)
            neg_penalty = self.memory_trace[:, neg_indices_tensor].mean(dim=1)
        else:
            neg_penalty = torch.zeros(
                len(self.memory_trace), dtype=self.memory_trace.dtype
            )

        composite_score = pos_score - neg_penalty
        actual_k = min(k, len(self.memory_trace))
        top_k_values, top_k_indices = torch.topk(composite_score, actual_k)

        return top_k_indices, top_k_values

    # --- NEW High-Level Composite Queries ---

    def retrieve_intensity_without_chaos(self, k=5):
        """Retrieves narratives with high emotional intensity (N4) but low external chaos (N27)."""
        print(
            f"\nRetrieving top {k} for Intensity without Chaos (High N4 / Low N27)..."
        )
        # Indices assume default model: N4 (Intensity), N27 (Chaos)
        return self.retrieve_memories_composite(
            positive_neurons=[4], negative_neurons=[27], k=k
        )

    def retrieve_structure_in_chaos(self, k=5):
        """Retrieves narratives exhibiting structure/planning (N13) amidst external chaos (N27)."""
        print(f"\nRetrieving top {k} for Structure in Chaos (High N13 & N27)...")
        # Indices assume default model: N13 (Structure), N27 (Chaos)
        return self.retrieve_memories_composite(
            positive_neurons=[13, 27], negative_neurons=[], k=k
        )

    def retrieve_routine_structure(self, k=5):
        """Retrieves narratives focused on routine/structure (High N13) without high emotion (N4) or chaos (N27)."""
        print(
            f"\nRetrieving top {k} for Routine Structure (High N13 / Low N4 & N27)..."
        )
        # Indices assume default model: N13 (Structure), N4 (Intensity), N27 (Chaos)
        return self.retrieve_memories_composite(
            positive_neurons=[13], negative_neurons=[4, 27], k=k
        )

    # --- Deprecated/Old Methods ---

    # def retrieve_external_crisis(
    #     self, crisis_neuron=10, distress_neuron=16, threshold=0.5, k=5
    # ):
    #     """DEPRECATED: Relied on neuron roles from older models."""
    #     print(
    #         f"\nDEPRECATED: Retrieving top {k} refined external crisis (N{crisis_neuron}>={threshold}, penalized by N{distress_neuron})..."
    #     )
    #     # ... (Implementation removed for brevity or kept commented out)
    #     pass

    # --- Utility Methods ---

    def get_activations(self, indices):
        """Returns the full activation vectors for the given indices."""
        try:
            return self.memory_trace[indices]
        except IndexError:
            print(f"Error: One or more indices are out of bounds.")
            return None


# --- Example Usage ---
if __name__ == "__main__":
    print("\n=== Narrative Memory API Example Usage ===\n")
    try:
        api = NarrativeMemoryAPI()  # Use default paths

        # Example 1: Retrieve top personal distress memories (Neuron 16)
        n16_indices, n16_vals = api.retrieve_top_k_memories(neuron_index=16, k=3)
        print("\nTop 3 Personal Distress (N16):")
        for i, idx in enumerate(n16_indices):
            print(
                f"  - Score: {n16_vals[i]:.4f}, Index: {idx.item()}, Text: {api.get_narrative_text(idx.item())}\n"
            )

        # Example 2: Retrieve top refined external crisis memories
        crisis_indices, crisis_scores = api.retrieve_external_crisis(k=3)
        print("\nTop 3 Refined External Crisis (N10 >= 0.5, penalized N16):")
        if crisis_indices.numel() > 0:
            for i, idx in enumerate(crisis_indices):
                print(
                    f"  - Score: {crisis_scores[i]:.4f}, Index: {idx.item()}, Text: {api.get_narrative_text(idx.item())}\n"
                )
        else:
            print("  (No results found for refined external crisis)\n")

        # Example 3: Retrieve top routine-like memories (Low N10 & N16)
        routine_indices, routine_scores = api.retrieve_memories_composite(
            positive_neurons=[], negative_neurons=[10, 16], k=3
        )
        print("\nTop 3 Routine Focus (Low N10 & N16):")
        # Note: score is higher for lower activation here
        for i, idx in enumerate(routine_indices):
            print(
                f"  - Score: {routine_scores[i]:.4f}, Index: {idx.item()}, Text: {api.get_narrative_text(idx.item())}\n"
            )

        # Example 4: Get activations for a specific memory
        # activations = api.get_activations(crisis_indices[0].item())
        # if activations is not None:
        #     print(f"\nActivations for index {crisis_indices[0].item()}: {activations}")

    except FileNotFoundError as e:
        print(f"\nError initializing API: {e}")
        print(
            "Please ensure 'sae_text_finetuned.pth' and 'synthetic_narrative_data.json' are present."
        )
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

    print("\n=== Example Usage Complete ===\n")
