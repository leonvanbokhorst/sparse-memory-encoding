# Sparse Memory Encoding

This project explores the concept of encoding narrative text into sparse representations using a Sparse Autoencoder (SAE). The goal is to create a "memory trace" where specific neurons in the compressed layer of the SAE activate for distinct types of narrative content, allowing for content-based retrieval.

## Project Structure

```
├── .gitignore                 # Git ignore file
├── .python-version            # Specifies Python version (pyenv)
├── narrative_memory_api.py    # API class to load model/data and retrieve narratives
├── narrative_memory_demo.ipynb # Jupyter notebook demonstrating API usage (likely)
├── generate_synthetic_data.py # Script to generate narrative data using OpenAI Batch API
├── sae_text_finetune.py       # Script to fine-tune the SAE model on text embeddings
├── sae_text_experiment.py     # Script for specific experiments (details TBC)
├── sae_narrative_l1_recon_experiment.py # Experiment focusing on L1/Reconstruction
├── sae_narrative_experiment.py # General narrative experiment script
├── sae_experiment_sparsity.py # Experiment focusing on sparsity
├── process_batch_results.py   # Utility script (likely related to OpenAI Batch API results)
├── check_data_batch.py        # Utility script for checking data batches
├── main.py                    # Placeholder main script
├── pyproject.toml             # Project metadata and dependencies (using uv/pip)
├── uv.lock                    # Lock file for dependencies
├── synthetic_narrative_data.json # Generated narrative data
├── sae_model_beta0005.pth     # Pre-trained SAE model weights (likely from non-text data)
├── sae_text_finetuned.pth     # Fine-tuned SAE model weights on narrative data
├── pca_narrative_activations.png # PCA visualization of activations
├── pca_narrative_activations_highlighted.png # Highlighted PCA visualization
├── neuron_activation_comparison.png # Visualization comparing neuron activations
├── docs/                      # Documentation directory (content unknown)
├── .venv/                     # Virtual environment directory
└── README.md                  # This file
```

## Core Concepts

1.  **Synthetic Data Generation:** Uses `generate_synthetic_data.py` and the OpenAI Batch API (`gpt-4.1-nano`) to create short narrative examples categorized as "Strong Emotional", "Complex/Chaotic", or "Routine".
2.  **Text Embedding:** Employs the `sentence-transformers` library (specifically `all-MiniLM-L6-v2`) to convert narrative text into dense vector embeddings (dimension 384).
3.  **Sparse Autoencoder (SAE):** A neural network (`SparseMemoryText` model defined in `sae_text_finetune.py` and `narrative_memory_api.py`) with an architecture like `Embedding -> Projection (ReLU) -> Encoder (ReLU) -> Decoder (Sigmoid)`. The key is the `Encoder` layer which compresses the projected embedding into a smaller, sparse representation (`COMPRESSED_SIZE=30`).
4.  **Fine-tuning:** The SAE is initialized with pre-trained weights (`sae_model_beta0005.pth`) and then fine-tuned (`sae_text_finetune.py`) on the embeddings of the synthetic narrative data. Training involves minimizing a weighted combination of:
    - **Reconstruction Loss:** Mean Squared Error between the decoder's output and the _projected_ input embedding.
    - **Sparsity Loss:** L1 norm of the encoded (compressed) layer activations, encouraging sparsity.
    - Weights for both losses can be adjusted based on the narrative category.
5.  **Narrative Memory API:** `narrative_memory_api.py` provides a high-level interface (`NarrativeMemoryAPI`) to:
    - Load the fine-tuned SAE (`sae_text_finetuned.pth`) and narrative data (`synthetic_narrative_data.json`).
    - Compute the "memory trace" (SAE activations for all narratives).
    - Retrieve narratives based on high activation of specific neurons or combinations of neurons (e.g., "find narratives activating neuron 15 highly but neuron 5 lowly").

## Setup

1.  **Prerequisites:**

    - Python >= 3.10
    - `uv` (recommended, based on `uv.lock`) or `pip`
    - An OpenAI API Key set as the environment variable `OPENAI_API_KEY` (for data generation).

2.  **Installation:**

    ```bash
    # Using uv (recommended)
    uv pip install -r requirements.txt # Or uv sync if pyproject.toml is primary

    # Or using pip
    # pip install -r requirements.txt # Assuming requirements.txt exists or is generated from pyproject.toml
    # pip install -e . # If project is packaged
    pip install ipykernel matplotlib numpy openai pca-magic sentence-transformers torch
    ```

    _(Note: A `requirements.txt` might need to be generated from `pyproject.toml` if not present)_

## Usage

1.  **Generate Data (if needed):**

    - Ensure `OPENAI_API_KEY` is set.
    - Run the generation script:
      ```bash
      python generate_synthetic_data.py
      ```
    - This will create `synthetic_narrative_data.json`.

2.  **Fine-tune the Model (if needed):**

    - Ensure `synthetic_narrative_data.json` exists.
    - Ensure the base model `sae_model_beta0005.pth` is present (or modify the script to train from scratch).
    - Run the fine-tuning script:
      ```bash
      python sae_text_finetune.py
      ```
    - This will create `sae_text_finetuned.pth` and potentially some analysis plots/output.

3.  **Use the Narrative Memory API:**

    - See `narrative_memory_demo.ipynb` for examples.
    - In Python code:

      ```python
      from narrative_memory_api import NarrativeMemoryAPI

      # Initialize (loads model, data, computes trace)
      api = NarrativeMemoryAPI()

      # Example: Find top 5 narratives activating neuron 10 most
      indices, activations = api.retrieve_top_k_memories(neuron_index=10, k=5)
      print(f"Top narratives for neuron 10:")
      for idx, act in zip(indices, activations):
          text = api.get_narrative_text(idx.item())
          print(f" - Index {idx.item()}: Activation {act:.4f} - '{text}'")

      # Example: Find narratives activating neuron 15 but NOT neuron 5
      indices, scores = api.retrieve_memories_composite(positive_neurons=[15], negative_neurons=[5], k=3)
      print(f"\nTop narratives for +N15 / -N5:")
      for idx, score in zip(indices, scores):
           text = api.get_narrative_text(idx.item())
           print(f" - Index {idx.item()}: Score {score:.4f} - '{text}'")
      ```

4.  **Run Experiments:**
    Execute the various `sae_*_experiment.py` scripts as needed. Their specific functions would need to be examined directly.

## Dependencies

Key libraries listed in `pyproject.toml`:

- `torch`: Deep learning framework
- `sentence-transformers`: Text embedding models
- `numpy`: Numerical computation
- `openai`: For synthetic data generation
- `matplotlib`: Plotting and visualization
- `ipykernel`: For Jupyter notebook support
- `pca`: Likely `pca-magic` for PCA analysis

## Future Work / Considerations

- Explore different SAE architectures or hyperparameters.
- Investigate the specific roles learned by individual neurons.
- Apply the model to real-world narrative datasets.
- Integrate the API into a larger application.
- Refine the data generation process.
