# Tasks: adding pairwise constraints to DCSS

## Open decision
- [ ] All pairs constrained, or only a sampled subset (0 = unknown)?
  - All pairs: no need to store the full n×n matrix. Build each batch's block from that batch's labels.
  - Subset: the full matrix (or the sampled pairs) has to be stored.

## Implementation
- [ ] **Constraint matrix function**: takes labels (n,) and returns n×n with 1 = same label, -1 = different label, 0 on the diagonal, symmetric.
  - n = 28,000 for letters A–J, so the full matrix is about 3.1 GB as float32 and 0.8 GB as int8.
  - A Python double loop over all pairs is too slow. Broadcasting does it in one operation.
- [ ] **Dataset returns the index**: `Datasets/Datasets.py:165`. `TensorDataset` accepts more tensors.
- [ ] **Unpack 3 values instead of 2** in the loops that are used:
  - `Autoencoder.py:81`, `get_latent_data`
  - `Autoencoder.py:144`, `pretrain_autoencoder`
  - `Autoencoder.py:180`, `train_autoencoder`
  - (`Autoencoder.py:97`, `get_similarity_data`, is never called)
- [ ] **Training loop**: get the batch's B×B block (index rows and columns with the batch indices) and add the constraint loss term from the other paper.
- [ ] **New λ for the constraint term**:
  - `set_training_variables` (`Autoencoder.py:35`)
  - `total_loss` (`Autoencoder.py:196`)
  - notebook parameters cell
- [ ] **Logging**: add a running sum of the new term, scaled by λ like the others:
  - new column in `df_eval` (`Autoencoder.py:164`)
  - its value in the per-epoch row (`:219`)
  - the print (`:220`)
- [ ] **Results path**: add the new λ to `set_path` (`Autoencoder.py:224`). Otherwise runs share a folder and the pretrained weights get overwritten.
- [ ] **Random seeds** in the top cell of the notebook:
  - `torch.manual_seed(...)`: layer weights, RBF centres, batch order (`Datasets/Datasets.py:167`)
  - `np.random.seed(...)`: k-means (`Autoencoder.py:112`), t-SNE (`Visualization.py:42`), plot colours (`Visualization.py:14`)
  - GPU (conv model): `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`
  - Repeatable only from a fresh kernel with all cells run in order. Not bit-identical on GPU without `torch.use_deterministic_algorithms(True)` (not tested on this code).

## Investigation
- [ ] **Overfitting with the current constraint loss** (`Objectives.py:45`, used at `Autoencoder.py:259`):
  - 1-epoch test run (letters A–J, batch 256, λ = 0.01): the loss was −945.7 on the first batch and −414.4 on the last. Times λ that is about −9 to −4 per batch, against about 0.47 per batch for the reconstruction loss.

## Notes
- There are no random seeds anywhere, so runs vary. Run several times when comparing with and without constraints.
