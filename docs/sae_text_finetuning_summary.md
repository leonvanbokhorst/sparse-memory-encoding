# SAE Text Narrative Fine-Tuning Summary

## Goal

To fine-tune a Sparse Autoencoder (SAE), pre-trained on synthetic sine-wave data, to represent short textual narratives ("emotional", "complex", "routine") using sparse, interpretable feature vectors. Key objectives included achieving differential sparsity (higher sparsity for routine narratives) and identifying meaningful neuron roles related to narrative categories, while avoiding representational collapse.

## Final Optimal Configuration (Achieved via `sae_text_finetune.py`)

- **Input Pre-trained Model:** `sae_model_beta0005.pth` (Trained on sine waves with BETA_L1=0.0005)
- **Output Fine-tuned Model:** `sae_text_finetuned.pth` (Result of the final successful run described below)
- **Training Data:** `synthetic_narrative_data.json` (1500 samples, 500 per category)
- **Script Used:** `sae_text_finetune.py` (version corresponding to the final successful run)
- **Reproducibility:** Seeds activated (`torch.manual_seed(42)`, `np.random.seed(42)`)
- **Key Hyperparameters:**
  - `BETA_L1` (Sparsity Coefficient): **0.002** (Crucial value to prevent collapse and ensure differentiation)
  - `LEARNING_RATE`: `1e-4`
  - `FINETUNE_EPOCHS`: `200`
  - `BATCH_SIZE`: `32`
- **Loss Weighting Maps:**
  - Reconstruction Weights (`recon_weights_map`): `{'emotional': 3.0, 'complex': 2.0, 'routine': 1.0}` (mapped to indices)
  - Sparsity Weights (`sparsity_weights_map`): `{'emotional': 0.3, 'complex': 0.5, 'routine': 1.0}` (mapped to indices)

## Fine-Tuning Journey & Key Findings

1.  **Initial Transfer Failure:** Direct application of the sine-wave trained SAE (`BETA_L1=0.0005`) with an untrained projection layer failed; roles didn't transfer, representations were noisy.
2.  **Fine-tuning Introduced:** Added a fine-tuning loop to train the projection layer and adapt the SAE weights.
3.  **Representational Collapse (`BETA_L1 = 0.0005`):** Fine-tuning with the original beta value led to representational collapse. A single neuron (e.g., Neuron 5 in one run) became hyper-dominant, activating strongly for all categories. Cosine similarities between categories were ≈ 1.0, indicating poor differentiation despite technically achieving differential sparsity percentages.
4.  **Increased Sparsity (`BETA_L1 = 0.001`):** Increasing beta suppressed the previously dominant neuron but another neuron (e.g., Neuron 22) took its place. Representational collapse persisted (Cosine ≈ 1.0).
5.  **Breakthrough (`BETA_L1 = 0.002`):** This level of sparsity pressure successfully prevented single-neuron dominance and representational collapse. The network learned a distributed representation.
6.  **Clear Routine Differentiation:** Achieved clear quantitative separation between Routine and Non-Routine (Emotional/Complex) categories.
    - Cosine Similarity (vs. Routine): ~0.94-0.96 (Significantly lower than ~1.0)
    - Euclidean Distance (vs. Routine): ~0.55-0.76 (Meaningful separation)
    - Sparsity: Routine consistently showed the highest sparsity (~96%).
7.  **Emotional/Complex Similarity:** These categories remained representationally similar (Cosine ≈ 0.998), likely reflecting their inherent conceptual overlap and shared activation of key intensity neurons.

## Interpreted Neuron Roles (from `BETA_L1 = 0.002` Seeded Run)

- **Neuron 16 (Personal Distress Specialist):** Consistently activates most strongly for texts describing intense, _personal_ negative emotions (grief, fear, loss, tears), especially related to relationships/family. Its low activation for Routine is a key differentiator.
- **Neuron 10 (Emotional Intensity & Crisis Indicator):** Also activates strongly for intense emotional texts (overlapping with N16), but potentially broader, including some external crises (e.g., thunderstorm). Its activation is significantly higher for Emo/Complex than for Routine.
- **(Other Neurons):** Neurons like 14 and 25 often appear in the top activations for intense texts across different runs, suggesting a cluster related to general significance/intensity, but their specific dominance varies with initialization compared to 16 and 10 in the final run.
- **(Routine Signature):** Characterized by _low_ activation of key neurons like 10 and 16.

## Conclusion & Next Steps

The fine-tuned SAE with `BETA_L1 = 0.002` provides a stable, interpretable, and sparse representation suitable for distinguishing routine from significant narrative events.

- **Immediate Actions:**
  - Document configuration and findings (this file).
  - Explore application/integration (e.g., memory retrieval based on neuron activations).
  - Enhance visualization (individual text activations).
- **Future Options:** Refine Emo/Complex distinction if needed (data augmentation, weight tuning), test on larger/real-world data.
