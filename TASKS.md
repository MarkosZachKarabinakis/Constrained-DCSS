# Tasks: adding pairwise constraints to DCSS

## Open decision
- [ ] All pairs constrained, or only a sampled subset (0 = unknown)?
  - All pairs: no need to store the full n×n matrix. Build each batch's block from that batch's labels.
  - Subset: the full matrix (or the sampled pairs) has to be stored.

## Implementation
- [ ] **Logging (rest)**: add the new term to `df_eval` in `train_semi_supervised_autoencoder`:
  - new column (`Autoencoder.py:225`)
  - its value in the per-epoch row (`:286`)
- [ ] **Results path**: add the new λ to `set_path` (`Autoencoder.py:292`). Otherwise runs share a folder and the pretrained weights get overwritten.
  - `semi_lambda` is not stored on `self`, and the notebook calls `set_path` before training.
- [ ] **Random seeds** in the top cell of the notebook:
  - `torch.manual_seed(...)`: layer weights, RBF centres, batch order (`Datasets/Datasets.py:167`)
  - `np.random.seed(...)`: k-means (`Autoencoder.py:112`), t-SNE (`Visualization.py:42`), plot colours (`Visualization.py:14`)
  - GPU (conv model): `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`
  - Repeatable only from a fresh kernel with all cells run in order. Not bit-identical on GPU without `torch.use_deterministic_algorithms(True)` (not tested on this code).

## Investigation
- [ ] **Overfitting with the current constraint loss** (`Objectives.py:45`, used at `Autoencoder.py:260`):
  - 1-epoch test run (letters A–J, batch 256, λ = 0.01): the loss was −945.7 on the first batch and −414.4 on the last. Times λ that is about −9 to −4 per batch, against about 0.47 per batch for the reconstruction loss.

## Done
- [x] **Constraint matrix function**: `make_constraint_matrix` (`General_Functions.py:90`). `optimizer_version=True` uses broadcasting. The double loop is still the default. Test: `Code-Tests/test_constraint_matrix.py`.
- [x] **Distance matrix function**: `make_distance_matrix` (`General_Functions.py:114`), `torch.cdist` on the batch codes. Test: `Code-Tests/test_distance_matrix.py`.
- [x] **Sub-matrix function**: `make_sub_matrix` (`General_Functions.py:119`) gets the batch's B×B block. Test: `Code-Tests/test_sub_matrix.py`.
- [x] **Constraint loss**: `semi_supervised_loss` (`Objectives.py:45`), sum of distance × constraint, divided by B. Test: `Code-Tests/test_semi_supervised_loss.py`.
- [x] **Dataset returns the index**: `Datasets/Datasets.py:165-166`.
- [x] **Unpack 3 values** in all 4 loops: `Autoencoder.py:81`, `:97`, `:144`, `:180`.
- [x] **Training loop**: new function `train_semi_supervised_autoencoder(constraint_matrix, semi_lambda)` (`Autoencoder.py:224`). It moves the constraint matrix to the device (`:229`), builds the batch's distance and constraint blocks (`:251-252`) and adds the term to the total loss (`:262`). `train_autoencoder` is unchanged.
- [x] **New λ** (`semi_lambda`): passed as an argument to `train_semi_supervised_autoencoder`, not through `set_training_variables`. Used in the total loss (`:262`) and set in the notebook parameters cell.
- [x] **Logging (part)**: running sum scaled by λ (`:268`), printed on its own line each epoch (`:288`).
- [x] **New notebook** `C-DCSS_Demo.ipynb`: builds the full constraint matrix from all labels (`optimizer_version=True`), sets `semi_lambda` and calls `train_semi_supervised_autoencoder`.

## Notes
- There are no random seeds anywhere, so runs vary. Run several times when comparing with and without constraints.
