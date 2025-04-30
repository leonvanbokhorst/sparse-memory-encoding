# Sparse Autoencoder Narrative Memory Experiments Summary

This document summarizes the series of experiments conducted to explore the use of Sparse Autoencoders (SAEs) for simulating adaptive episodic and narrative memory, focusing on feature preservation, sparsity, and interpretability.

## Overall Goal

The primary objective was to investigate if SAEs could model key aspects of human memory, such as:

1.  **Compression:** Reducing high-dimensional episodic snapshots into smaller representations.
2.  **Selective Recall:** Preserving salient or important features while allowing irrelevant details to fade.
3.  **Sparsity:** Using efficient, sparse internal codes, potentially reflecting focused neural activation.
4.  **Adaptability:** Adjusting memory encoding based on narrative importance.
5.  **Interpretability:** Understanding how the network represents different types of information.

## Experiment Phases

### Phase 1: Initial Proof-of-Concept (`sae-memory-poc.py`)

- **Goal:** Implement a basic SAE in PyTorch.
- **Methodology:**
  - Simple `SparseMemory` module with linear Encoder and Decoder.
  - Reconstruction Loss: Mean Squared Error (MSE).
  - Sparsity Penalty: KL Divergence between average neuron activation and a `sparsity_target` (e.g., 0.05), scaled by `beta`.
  - Data: Single random vector representing one memory episode.
- **Findings:** Demonstrated the fundamental mechanism of encoding, decoding, and applying a sparsity penalty in a single training step.
- **Code:** `sae-memory-poc.py`
- **Documentation:** `docs/sae_memory_poc_explanation.md`

### Phase 2: Sparsity Visualization with Multiple Episodes (`sae_experiment_sparsity.py` - Initial Run)

- **Goal:** Observe the effect of the KL sparsity penalty over multiple training epochs using simulated episodic data with distinct features.
- **Methodology:**
  - Extended PoC code into a full training loop.
  - Data Generation: Batch of random vectors (`event_size=300`). Specific indices in the first two episodes were boosted (`+= 0.8`) to represent "important events".
  - Losses: MSE + KL Sparsity (`beta=1.0`).
  - Analysis: Plotted loss curves, compared original vs. reconstructed important/routine episodes, visualized the sparse `memory_trace` (encoded representation).
- **Findings:**
  - KL Sparsity loss dominated the total loss, while reconstruction loss became very low quickly.
  - The network achieved high sparsity in the memory traces.
  - The boosted features in important episodes were visually somewhat preserved in reconstruction, but noise reduction/fading was also apparent.
  - Random vectors made precise visual interpretation difficult.
- **Code:** `sae_experiment_sparsity.py` (first version)

### Phase 3: Improved Visualization with Sine Waves (`sae_experiment_sparsity.py` - Sine Wave Update)

- **Goal:** Use more structured, visually interpretable data (sine waves) to better assess feature preservation and the impact of sparsity.
- **Methodology:**
  - Modified `generate_episodes` function:
    - Generated a base noisy sine wave for all episodes ("Routine").
    - Episode 0 ("Strong Emotional"): Boosted amplitude in the first quarter.
    - Episode 1 ("Complex/Chaotic"): Higher frequency wave in the second quarter.
  - Retained KL sparsity (`beta=1.0`).
- **Findings:**
  - Clear visual confirmation of feature preservation: Boosted amplitude and higher frequency sections were well-reconstructed.
  - Noise reduction (smoothing) was evident in reconstructions.
  - Distinct sparse `memory_trace` patterns were observed for different event types.
  - However, KL sparsity still resulted in **uniformly high sparsity** across all traces, failing to differentiate based on the introduced features' perceived importance.
- **Code:** `sae_experiment_sparsity.py` (second version)

### Phase 4: Narrative Relevance & Dynamic Sparsity Attempt (`sae_narrative_experiment.py`)

- **Goal:** Introduce narrative labels and attempt to control sparsity dynamically based on narrative importance, aiming for richer codes for important events.
- **Methodology:**
  - Used sine wave data with explicit narrative labels.
  - Introduced `importance_weights` tensor (lower weight = less penalty).
  - Modified KL sparsity function (`weighted_sparse_penalty`) to apply these weights _per episode_ before averaging the penalty.
  - `beta` remained relatively high (`3.0`).
- **Findings:**
  - **Weighted KL Sparsity Ineffective:** The per-episode weighting had minimal impact. All traces still converged to a similar, very high level of sparsity (~57%). Reconstruction remained excellent, and retrieval from traces worked.
  - **Conclusion:** KL divergence penalty, especially with a strong global `beta`, enforces average sparsity strongly and isn't easily controlled on a per-instance basis with simple weighting.
- **Code:** `sae_narrative_experiment.py`

### Phase 5: Improved Control with L1 Sparsity & Weighted Reconstruction (`sae_narrative_l1_recon_experiment.py`)

- **Goal:** Achieve more granular sparsity control using L1 regularization, enhance recall fidelity for important events using weighted reconstruction loss, and analyze code distinctiveness more rigorously.
- **Methodology:**
  - Replaced KL sparsity with **L1 sparsity penalty** (`torch.norm(encoded, p=1, dim=1)`), applying `sparsity_importance_weights` (lower weight = less L1 penalty).
  - Introduced **weighted reconstruction loss**, applying `recon_importance_weights` (higher weight = prioritize accurate reconstruction) to the per-episode MSE before averaging.
  - Added **distinctiveness analysis**: Cosine Similarity (original & normalized traces), Euclidean Distance.
  - Added **neuron overlap analysis** across multiple thresholds.
  - Experimented with different `BETA_L1` values.
- **Findings (Iterative):**
  - **Run 1 (`BETA_L1 = 0.001`):**
    - Achieved high sparsity (~93%).
    - Weighted reconstruction worked well.
    - _Paradox:_ High cosine similarity (pattern overlap) between Emotional & Routine traces.
    - Neuron analysis revealed significant feature reuse, sometimes collapsing to single-neuron encoding.
  - **Run 2 (`BETA_L1 = 0.0005`): Optimal Balance Found**
    - **Successful Differential Sparsity:** Emotional event (lowest sparsity weight) became clearly the least sparse (~87%), while Chaotic/Routine remained highly sparse (~97%).
    - **Multi-Neuron Encoding:** Emotional event used multiple (4) active neurons, unlike the others (1 neuron each).
    - **Improved Distinctiveness:** Lower Emo/Routine cosine similarity (~0.71). Chaotic trace orthogonal to others.
    - **Clearer Neuron Roles:** Revealed stable overlap patterns (Emo/Routine share Neuron 7; Emo/Chaotic share Neuron 0), no single core neuron across all three.
  - **Run 3 (`BETA_L1 = 0.0001`):**
    - Relaxed sparsity pressure further, resulting in richer multi-neuron codes for _all_ types (Overall sparsity ~83%).
    - Lost some differential sparsity benefits observed at `0.0005`.
    - Increased Emo/Routine pattern similarity again (~0.95).
    - Revealed a stable 3-neuron core (`[14, 25, 28]`) across all events with this relaxed pressure.
- **Code:** `sae_narrative_l1_recon_experiment.py`

## Overall Conclusions & Next Steps

- SAEs can effectively compress sine-wave representations of narrative events, preserving key features while reducing noise.
- L1 sparsity provides more granular control over sparsity levels per instance compared to KL divergence when combined with importance weighting.
- Weighting the reconstruction loss is highly effective at improving the recall fidelity for designated important events.
- The choice of sparsity penalty (`beta`) significantly impacts the resulting sparsity level and the network's encoding strategy (e.g., single vs. multi-neuron codes, feature reuse patterns).
- The run with `BETA_L1 = 0.0005` achieved the best balance: differential sparsity reflecting importance, multi-neuron codes for the most important event, good distinctiveness, and interpretable neuron overlap patterns.
- Detailed neuron activation analysis revealed specific neurons associated with core structures, unique event features (Emotional), and shared features, confirming interpretability.
- **Next Step:** Leverage the insights and the trained model from the `BETA_L1 = 0.0005` run to transition to **textual narrative episodes**, mapping learned neuron roles and adaptive memory principles onto text embeddings.
