#!/usr/bin/env python3

import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import jax
import jax.numpy as jnp
import sys
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
except ImportError:
    raise ImportError("Install UMAP via: pip install umap-learn")

THIS_DIR = os.path.dirname(__file__)
OG_IMPLS = os.path.abspath(os.path.join(THIS_DIR, "..", "ogbench", "ogbench", "impls"))
OG_IMPLS_BASE = os.path.abspath(os.path.join(THIS_DIR, "..", "ogbench", "ogbench"))
print("Adding OGBench impls to sys.path:", OG_IMPLS)
sys.path.insert(0, OG_IMPLS)
sys.path.insert(0, OG_IMPLS_BASE)

from utils.datasets import Dataset, GCDataset
from utils.flax_utils import restore_agent
from agents.crl import CRLAgent, get_config
from ogbench import load_dataset


# -------------------------------------------------------------
# Utility: get phi and psi embeddings from critic
# -------------------------------------------------------------
def get_embeddings(agent, batch):
    critic = agent.network.select("critic")

    # Forward pass with info=True gives v, phi, psi
    v, phi, psi = critic(
        batch["observations"],
        batch["value_goals"],
        actions=batch["actions"],
        info=True,
        params=agent.network.params,
    )

    # phi, psi shape: (ensemble, batch, latent_dim)
    # For visualization use ensemble mean:
    phi = np.array(phi).mean(axis=0)
    psi = np.array(psi).mean(axis=0)

    return phi, psi


# -------------------------------------------------------------
# Main script
# -------------------------------------------------------------
def main(args):

    print("Loading dataset...")

    cfg = get_config()
    # convert to plain dict
    cfg = dict(cfg)
    cfg["alpha"] = 0.03

    train_dataset = load_dataset(args.dataset_path, compact_dataset=True)
    base_train = Dataset.create(**train_dataset)
    dataset = GCDataset(base_train, cfg)
    batch = dataset.sample(args.num_samples)
    example_batch = dataset.sample(1)

    print("Loading agent checkpoint...")
    agent_tmp = CRLAgent.create(
        seed=0, ex_observations=example_batch["observations"], ex_actions=example_batch["actions"], config=cfg
    )

    agent = restore_agent(agent_tmp, args.agent_checkpoint , 20000)

    print("Computing critic embeddings...")
    phi, psi = get_embeddings(agent, batch)

    # ---------------------------------------------------------
    #  Compute similarity matrix φ·ψᵀ
    # ---------------------------------------------------------
    similarity = phi @ psi.T

    # ---------------------------------------------------------
    #  Dimensionality reduction
    # ---------------------------------------------------------
    pca = PCA(n_components=2).fit(phi)
    phi_pca = pca.transform(phi)
    psi_pca = pca.transform(psi)

    tsne = TSNE(n_components=2, perplexity=30)
    phi_tsne = tsne.fit_transform(phi)
    psi_tsne = tsne.fit_transform(psi)

    reducer = umap.UMAP()
    phi_umap = reducer.fit_transform(phi)
    psi_umap = reducer.fit_transform(psi)

    os.makedirs(args.out_dir, exist_ok=True)

    # ---------------------------------------------------------
    #  Plot helpers
    # ---------------------------------------------------------
    def plot_embed(x, y, title, path):
        plt.figure(figsize=(8,8))
        plt.scatter(x[:,0], x[:,1], s=8, alpha=0.6, label="phi")
        plt.scatter(y[:,0], y[:,1], s=8, alpha=0.6, label="psi")
        plt.title(title)
        plt.legend()
        plt.savefig(path, dpi=150)
        plt.close()


    print("Creating plots...")

    plot_embed(phi_pca, psi_pca,
               "Critic Embeddings — PCA",
               f"{args.out_dir}/pca.png")

    plot_embed(phi_tsne, psi_tsne,
               "Critic Embeddings — t-SNE",
               f"{args.out_dir}/tsne.png")

    plot_embed(phi_umap, psi_umap,
               "Critic Embeddings — UMAP",
               f"{args.out_dir}/umap.png")

    # ---------------------------------------------------------
    # Similarity heatmap
    # ---------------------------------------------------------
    plt.figure(figsize=(10,8))
    sns.heatmap(similarity, cmap="viridis")
    plt.title("φ·ψᵀ Similarity Matrix")
    plt.savefig(f"{args.out_dir}/similarity_matrix.png", dpi=150)
    plt.close()

    print("Saved all plots to", args.out_dir)


# -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--agent-checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="critic_embeddings")
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--config", type=dict, required=False,
                        help="The CRL config dict used when training")
    args = parser.parse_args()


    main(args)
