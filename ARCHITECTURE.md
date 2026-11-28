# Architecture Guide: DCSS (Deep Clustering with Soft Silhouette)

How the code is put together and how a run moves through it. This describes the code as it is.

---

## 1. What the program does

One notebook run:

1. Loads a dataset into a DataLoader.
2. Builds an autoencoder with an extra clustering head.
3. **Pretrains** it to reconstruct its input.
4. **Initialises** the cluster centres with k-means.
5. **Trains** it to reconstruct and form clusters at the same time.
6. Saves the per-epoch scores to Excel and a t-SNE plot to PNG.

True labels never enter training. They are only used for scores and plot colours.

---

## 2. Code map

Who calls whom:

```
DCSS_Demo.ipynb                        entry point: settings + run order
├── Datasets/Datasets.py               get_dataset()                  dataset name → DataLoader
├── Autoencoder.py                     the model + all training stages
│   ├── RBF.py                         RBF layer                      the clustering head
│   ├── Objectives.py                  soft_silhouette(), entropy()   loss terms
│   ├── Evaluations/Evaluation.py      Evaluator                      ACC, PUR, NMI, ARI
│   └── General_Functions.py           create_directory()             folder for the weights
├── General_Functions.py               save_excel()                   scores → Data.xlsx
└── Visualization.py                   plot_tsne()                    latent space → PNG
    └── General_Functions.py           save_plot()                    plot folder + file name
```

- There is no script or CLI entry point. Everything runs from the notebook, from the repo root.
- `Code-Tests/` holds a test script for `make_constraint_matrix` (in `General_Functions.py`). It is not part of the run.

---

## 3. The model class

```
nn.Module
└── GenericAutoencoder      all the logic: settings, forward passes, pretrain, k-means init, train, save
    ├── Autoencoder         MLP layers, for tabular data
    └── CD_Autoencoder      CNN layers, for 28×28 images
```

The two subclasses only define layers. Each builds three sub-networks:

| Sub-network | Role |
|---|---|
| `encoder_model` | input → latent vector `z` (size `latent_dim`) |
| `decoder_model` | `z` → reconstructed input |
| `cluster_model` | one RBF layer: holds one learnable centre per cluster and scores `z` against each |

How data moves through them:

```
                        ┌─► decoder_model ─► reconstruction                  forward()
x ─► encoder_model ─► z ┤
                        └─► cluster_model ─► softmax ─► soft memberships     forward_softMax()
                                                        (B × n_clusters, each row sums to 1)
```

`encoder()` runs only the encoder and returns `z`.

Methods of `GenericAutoencoder`, by purpose:

| Purpose | Methods |
|---|---|
| Store settings | `set_general_training_variables`, `set_pretraining_variables`, `set_training_variables`, `set_path_variables` |
| Build results path | `set_path` → `self.data_dir_path` |
| Forward passes | `forward`, `encoder`, `forward_softMax` |
| Training stages | `pretrain_autoencoder`, `kmeans_initialization`, `train_autoencoder` |
| Whole-dataset pass | `get_latent_data` → `(data, z, labels)` for every sample |
| Output | `save_pretrained_weights`, `get_cluster_centers` |
| Helper | `torch_to_numpy`: soft memberships → hard cluster ids (argmax) |

What the object holds:

- **From the constructor:** `device`, `n_clusters`, `input_dim`, `latent_dim`, `negative_slope`, and its own `Evaluator` and `Objectives`.
- **From the setters:** `dataloader`, `batch_size`, epochs, learning rates, both λ, `dataset_name`, `path_to_module`. `set_path` adds `data_dir_path`.
- **After training:** `df_eval` (the score table), and the last epoch's `latent_data_list`, `labels_list`, `clusters_list`.

The DataLoader is stored on the model (`self.dataloader`). Every stage loops over it, so the stage methods take no data arguments.

---

## 4. Run sequence (notebook, top to bottom)

| # | Notebook cell | Code called | Effect |
|---|---|---|---|
| 1 | Libraries, Global Variables, Visualization | — | Imports, picks `cuda:0` or CPU, creates `Visualization()` |
| 2 | Load Dataset | `get_dataset(dataset_name, batch_size)` | Returns `dataloader, input_dim, data_np, labels` |
| 3 | Parameters | — | Sets the remaining settings by hand |
| 4 | Create Autoencoder Model | `Autoencoder(...)` or `CD_Autoencoder(...)`, the 4 setters, `set_path()`, `.to(device)` | Model ready, results path known |
| 5 | Pretrain Autoencoder | `pretrain_autoencoder()` + `save_pretrained_weights()`, or `load_state_dict(...)` if `use_pretrain=False` | Encoder and decoder trained on reconstruction. Weights saved, or loaded instead |
| 6 | Initialize the clustering layer | `kmeans_initialization(n_init=100)` | RBF centres set from k-means. K-means scores printed |
| 7 | Train Autoencoder | `train_autoencoder()` | Full training (section 5). Fills `df_eval`, returns `latent_data, labels, clustering` |
| 8 | Save Clustering Results | `General_Functions().save_excel(data_dir_path, df_eval)` | `df_eval` added as a new sheet in `Data.xlsx` |
| 9 | Plot Clustering Results | `get_cluster_centers()`, `get_latent_data()`, `plot_tsne(...)` | t-SNE PNG saved |

### Settings (notebook defaults)

`dataset_name`, `batch_size` and `n_clusters` are set in the **Load Dataset** cell. The rest are in the **Parameters** cell.

| Name | Default | Meaning |
|---|---|---|
| `dataset_name` | `emnist_balanced_letters_A_J` | See section 6 |
| `batch_size` | 256 | Batch size for every stage |
| `n_clusters` | 10 | Number of clusters. Must match the dataset |
| `latent_dim` | 10 | Size of `z` |
| `negative_slope` | 0 | LeakyReLU slope. 0 = plain ReLU |
| `n_pret_epochs`, `pret_lr` | 100, 1e-3 | Pretraining |
| `n_epochs`, `lr` | 100, 5e-4 | Training |
| `sil_lambda`, `entr_lambda` | 0.01, 0.01 | Weights of the two clustering loss terms |
| `use_pretrain` | True | True = pretrain now. False = load saved weights |
| `is_mlp_ae` | False | True = MLP (tabular). False = CNN (images). Must match the dataset |

---

## 5. Inside the training stages

**`train_autoencoder()`**, repeated `n_epochs` times.

For each batch `(data, labels)` from `self.dataloader`:

1. Reshape `data` to `(B, input_dim, 28, 28)` if CNN.
2. Forward passes: `forward` → reconstruction, `encoder` → `z`, `forward_softMax` → soft memberships.
3. Loss terms:
   - `rec_loss`: MSE between reconstruction and `data`.
   - `soft_sil`: `objectives.soft_silhouette(z, memberships)`. Cluster quality within the batch, −1 to 1, higher is better.
   - `mean_entropy`: `objectives.entropy(memberships)`. How unsure the memberships are; lower = more confident.
4. `total_loss = rec_loss + sil_lambda · (1 − soft_sil) + entr_lambda · mean_entropy`
5. Backward pass and Adam step (`lr`). All three sub-networks are updated.
6. Add the batch's values to the running sums. Keep the batch's memberships, labels and `z`.

After the last batch:

7. Hard clusters = argmax of the kept memberships.
8. `evaluator.evaluate_model(...)` compares hard clusters with labels → ACC, PUR, NMI, ARI.
9. Add one row to `df_eval` and print it.

It returns the last epoch's `(z, labels, clusters)`.

**`pretrain_autoencoder()`** is the same loop with only `rec_loss`, Adam with `pret_lr` and weight decay 1e-5, and no scoring. Only the encoder and decoder change.

**`kmeans_initialization()`** runs `get_latent_data()` over the whole dataset, fits k-means on `z`, and replaces the RBF centres with the k-means centres.

---

## 6. Data

```
get_<name>_np()     loads files, min-max scales          → numpy data, labels
get_dataset()       TensorDataset + DataLoader (shuffled) → stored on the model as self.dataloader

each batch (data, labels):
    data   → the model, in every stage
    labels → Evaluator and plot colours only
```

`dataset_name` is turned into the function name `get_<name>_np` in `Datasets/Datasets.py`.

| `dataset_name` | Type | Classes (label values) | Files needed in `Datasets/` |
|---|---|---|---|
| `emnist_balanced_digits` | image | 10 (0–9) | `Emnist/emnist-balanced-*` |
| `emnist_balanced_letters_A_J` | image | 10 (10–19) | `Emnist/emnist-balanced-*` |
| `emnist_balanced_letters_K_T` | image | 10 (20–29) | `Emnist/emnist-balanced-*` |
| `emnist_balanced_letters_U_Z` | image | 6 (30–35) | `Emnist/emnist-balanced-*` |
| `emnist_mnist` | image | 10 (0–9) | `Emnist/emnist-mnist-*` |
| `pendigits` | tabular | 10 | `Pendigits/` (in the repo) |
| `har` | tabular | 6 | `Har/train.csv`, `Har/test.csv` |
| `waveform_v1` | tabular | 3 | `Waveform-v1/data.csv`, `Waveform-v1/labels.csv` |
| `synthetic` | tabular | 4 | none, generated in code |

- Datasets with separate train and test files (EMNIST, Pendigits, HAR) have them **merged**. There is no train/test split.
- Every dataset is min-max scaled to [0, 1] per feature, fitted on all of the data.
- `input_dim` (returned by `get_dataset`) is the number of features for tabular data, and the number of channels (1) for images.

---

## 7. Outputs on disk

```
Results/<dataset_name>/SSAE/<n_epochs>_Eps_ld_<latent_dim>_out_<n_clusters>_bs_<batch_size>_lr_<lr>_sil_lambda_<…>_entr_lambda_<…>/
├── Weigths/autoencoder_weights.pth       written after pretraining, only when use_pretrain=True (whole model)
├── Data.xlsx                             one sheet per run: Experiment_1, Experiment_2, …
└── Experiments/TSNE/Experiment_N_TSNE.png
```

`df_eval` columns (one row per epoch): `Rec_Loss, Cl_Loss, Soft_Sil, Accuracy, Purity, Nmi, Ari`.

`Results/` and `*.png` are git-ignored.

---

## 8. Hidden settings and traps

**Setup**
- [ ] Run from the repo root. All paths are relative (`./Datasets/`, `./Results/`).
- [ ] `n_clusters` and `is_mlp_ae` are not derived from the dataset. Set them to match it (`letters_U_Z` needs 6).
- [ ] `set_path()` reads values stored by `set_general_training_variables`, `set_training_variables` and `set_path_variables`, so it must be called after them.
- [ ] Fixed in code, not in the notebook: softmax temperature 0.25, pretraining weight decay 1e-5, entropy base 2, image size 28 (the CNN only works with 28×28 images).
- [ ] There are no random seeds, so runs differ. The only seed is `np.random.seed(42)` inside `get_synthetic_np`.

**Data**
- [ ] EMNIST letter subsets keep their original labels (A–J = 10–19). They are not renumbered to 0–9.
- [ ] The `digits` and `letters` options both load EMNIST *balanced*. Only `emnist_mnist` loads different files.
- [ ] The DataLoader reshuffles on every pass. Arrays returned by one call line up with each other, but not with arrays from another call. In the plot cell, `clustering` (from training) and `latent_data, labels` (from a new pass) are in different orders. The notebook also overwrites `labels` twice.

**Model and training**
- [ ] The model is never put in `eval()` mode and nothing runs under `no_grad`. BatchNorm uses batch statistics in every pass, including k-means init and plotting.
- [ ] Soft silhouette is computed per batch, so its value depends on `batch_size`.
- [ ] Per-epoch scores use the memberships collected during training (each taken before that batch's update), not a separate clean pass.

**Logging**
- [ ] The loss values (`Rec_Loss`, `Cl_Loss`, `Soft_Sil`, printed `Entropy`, and the pretraining `Loss`) are **sums over the epoch's batches**, not means. ACC, PUR, NMI and ARI are computed once over the whole epoch.
- [ ] `Cl_Loss`, `Soft_Sil` and `Entropy` are already multiplied by their λ, so `Cl_Loss + Soft_Sil = n_batches × sil_lambda`.
- [ ] Entropy is printed but not saved to `df_eval`.

**Results folder**
- [ ] The folder name includes `dataset_name`, `n_epochs`, `latent_dim`, `n_clusters`, `batch_size`, `lr` and both λ.
- [ ] It does **not** include `n_pret_epochs`, `pret_lr`, `negative_slope` or the model type (MLP/CNN). Runs that differ only in those share a folder, and pretraining overwrites the weights.
- [ ] Changing any setting that is in the name makes `use_pretrain=False` look in a different folder, which may have no weights.
- [ ] Weights are loaded with `torch.load` without `map_location`, so weights saved on a GPU fail to load on a CPU-only machine.
- [ ] The weights folder is spelled `Weigths`.
- [ ] The plot is named after the **last sheet** in `Data.xlsx`, so Save must run before Plot.
- [ ] Excel writing (`writer.book = …`, `writer.save()`) needs pandas < 2. It is pinned to 1.4.4.

**Plot**
- [ ] The t-SNE plot colours points by **true labels**, not by predicted clusters (`y_predict` is ignored).

---

## 9. Not used by the notebook

- `get_similarity_data`, `forward_similarity`, `decoder` (`Autoencoder.py`)
- `find_images_to_plot`, `make_constraint_matrix` (`General_Functions.py`)
- `plot_image`, `plot_collage`, `plot`, `plot_3D` (`Visualization.py`)
- All RBF functions except `gaussian`, and `basis_func_dict` (`RBF.py`)
- All `Evaluator.get_*` methods. `get_cluster_accuracy` and `get_purity` would fail, because they return attributes that are never set.
