"""
============================================================
Embedding Visualization of CRL Datasets
============================================================

This script visualizes high-dimensional observation data from CRL trajectories
(e.g. environments from OGBench or similar sources) by projecting them into 2D
embeddings using **dimensionality reduction algorithms**.

--- Core Idea ---

Real-world observations (like robot states, object positions, or sensor readings)
live in *high-dimensional space*. To understand the structure of the dataset
(e.g., similarity, clustering, or diversity between episodes), we use algorithms
that map these high-dimensional vectors into 2D while preserving important
structure.

1. PCA (Principal Component Analysis)
   - A *linear* projection that finds the directions (principal components)
     explaining the most variance in the data.
   - Fast, deterministic, interpretable.
   - Often used as a first step or for roughly isotropic data.

2. t-SNE (t-distributed Stochastic Neighbor Embedding)
   - A *non-linear* embedding that preserves *local* structure:
     points that are close in the original space stay close in 2D.
   - Useful for discovering clusters or manifold-like structure.
   - Computationally more expensive, non-deterministic.

We generate two sets of embeddings:
  - **Initial observations** (first frame per episode)
  - **Episode-averaged observations** (mean of all frames in one episode)

If embeddings were computed before (stored in `embeddings.npz`), they are loaded
to save time. Otherwise, they are computed and saved automatically.

The output folder will contain:
  - PCA and t-SNE scatter plots for initial and averaged observations
  - Combined plots comparing PCA vs t-SNE
  - Density heatmaps for better intuition
  - `embeddings.npz` with numeric arrays for reuse
"""

import os
import json
import numpy as np
import argparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def load_episodes(dataset_root):
    with open(os.path.join(dataset_root, "all_episodes.json"), "r") as f:
        data = json.load(f)
    return data["episodes"]


def flatten_obs(obs):
    """Flatten a nested observation dictionary into a single 1D NumPy array."""
    if isinstance(obs, dict):
        return np.concatenate([np.ravel(v) for v in obs.values()])
    return np.ravel(obs)


def compute_embeddings(episodes, mode="initial", method="pca"):
    """Compute 2D embeddings for episode observations using PCA or t-SNE."""
    obs_list = []
    if mode == "initial":
        for ep in episodes:
            obs_list.append(flatten_obs(ep["trajectory"][0]["obs"]))
    elif mode == "full":
        for ep in episodes:
            seq = [flatten_obs(step["obs"]) for step in ep["trajectory"]]
            obs_list.append(np.mean(seq, axis=0))
    X = np.stack(obs_list)

    if method == "pca":
        emb = PCA(n_components=2).fit_transform(X)
    elif method == "tsne":
        emb = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto").fit_transform(X)
    else:
        raise ValueError(f"Unknown method: {method}")
    return emb


def plot_embeddings(emb, title, save_path, color=None):
    """Plot a simple 2D scatter of embeddings."""
    plt.figure(figsize=(6, 6))
    plt.scatter(emb[:, 0], emb[:, 1], s=18, alpha=0.7, c=color, cmap="viridis")
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_density(emb, title, save_path, bins=100):
    """Plot a 2D density map of the embedding space."""
    plt.figure(figsize=(6, 6))
    plt.hist2d(emb[:, 0], emb[:, 1], bins=bins, cmap="viridis")
    plt.colorbar(label="Density")
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_comparison(emb1, emb2, title, save_path):
    """Plot PCA vs t-SNE results side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].scatter(emb1[:, 0], emb1[:, 1], s=18, alpha=0.7)
    axes[0].set_title("PCA")
    axes[1].scatter(emb2[:, 0], emb2[:, 1], s=18, alpha=0.7)
    axes[1].set_title("t-SNE")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default="embeddings")
    parser.add_argument("--recompute", action="store_true",
                        help="Force recomputation even if precomputed embeddings exist.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    emb_path = os.path.join(args.out, "embeddings.npz")

    if os.path.exists(emb_path) and not args.recompute:
        print(f"Found precomputed embeddings in {emb_path}, loading...")
        data = np.load(emb_path)
        emb_init_pca = data["emb_init_pca"]
        emb_full_pca = data["emb_full_pca"]
        emb_init_tsne = data["emb_init_tsne"]
        emb_full_tsne = data["emb_full_tsne"]
    else:
        print("Loading dataset and computing embeddings...")
        episodes = load_episodes(args.dataset)
        emb_init_pca = compute_embeddings(episodes, mode="initial", method="pca")
        emb_full_pca = compute_embeddings(episodes, mode="full", method="pca")
        emb_init_tsne = compute_embeddings(episodes, mode="initial", method="tsne")
        emb_full_tsne = compute_embeddings(episodes, mode="full", method="tsne")

        np.savez(
            emb_path,
            emb_init_pca=emb_init_pca,
            emb_full_pca=emb_full_pca,
            emb_init_tsne=emb_init_tsne,
            emb_full_tsne=emb_full_tsne,
        )
        print(f"Saved new embeddings to {emb_path}")

    # -------------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------------

    print("Creating visualizations...")

    plot_embeddings(emb_init_pca, "Initial Observations (PCA)", os.path.join(args.out, "init_pca.png"))
    plot_embeddings(emb_full_pca, "Episode Averages (PCA)", os.path.join(args.out, "full_pca.png"))

    plot_embeddings(emb_init_tsne, "Initial Observations (t-SNE)", os.path.join(args.out, "init_tsne.png"))
    plot_embeddings(emb_full_tsne, "Episode Averages (t-SNE)", os.path.join(args.out, "full_tsne.png"))

    plot_comparison(emb_init_pca, emb_init_tsne, "Initial Observations Comparison", os.path.join(args.out, "init_compare.png"))
    plot_comparison(emb_full_pca, emb_full_tsne, "Episode Average Comparison", os.path.join(args.out, "full_compare.png"))

    plot_density(emb_init_pca, "Initial Obs Density (PCA)", os.path.join(args.out, "init_pca_density.png"))
    plot_density(emb_init_tsne, "Initial Obs Density (t-SNE)", os.path.join(args.out, "init_tsne_density.png"))

    print(f"Visualizations saved to {args.out}")


if __name__ == "__main__":
    main()
