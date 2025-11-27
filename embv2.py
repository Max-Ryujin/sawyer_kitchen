#!/usr/bin/env python3

import os
import argparse
import jax
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import pandas as pd
import umap

# Path setup to include OGBench
import sys
THIS_DIR = os.path.dirname(__file__)
OG_IMPLS = os.path.abspath(os.path.join(THIS_DIR, "..", "ogbench", "ogbench", "impls"))
OG_IMPLS_BASE = os.path.abspath(os.path.join(THIS_DIR, "..", "ogbench", "ogbench"))
sys.path.insert(0, OG_IMPLS)
sys.path.insert(0, OG_IMPLS_BASE)

from utils.datasets import Dataset, GCDataset
from utils.flax_utils import restore_agent
from agents.crl import CRLAgent, get_config
from ogbench import load_dataset

def normalize(x, mean, std, eps=1e-5):
    return (x - mean) / (std + eps)

def get_indices(dataset_dict):
    """Identify start, goal, and intermediate indices."""
    terminals = dataset_dict["terminals"].flatten().astype(bool)
    if "timeouts" in dataset_dict:
        terminals = terminals | dataset_dict["timeouts"].flatten().astype(bool)
    
    N = len(terminals)
    indices = np.arange(N)
    
    # Goal/End indices
    goal_idxs = indices[terminals]
    
    # Start indices (0 and indices immediately following a terminal)
    # Note: This assumes the dataset is contiguous trajectories
    start_idxs = np.concatenate(([0], goal_idxs[:-1] + 1))
    
    # Filter out potential out-of-bounds if last terminal is last element
    start_idxs = start_idxs[start_idxs < N]
    
    return start_idxs, goal_idxs, indices

def extract_embeddings(agent, obs, goals, actions):
    """
    Get phi (state-action) and psi (state-goal) embeddings.
    Phi represents the 'current' state, Psi represents the 'target' compatibility.
    """
    # CRL Critic signature: critic(obs, value_goals, actions, ...)
    # If we want to see how the critic views the transition, we usually look at 
    # phi(s, a) vs psi(s, g)
    
    critic = agent.network.select("critic")
    
    # We use batch inference
    # Note: CRL normally takes normalized inputs. Ensure inputs are normalized before here.
    v, phi, psi = critic(
        obs,
        goals,
        actions=actions,
        info=True,
        params=agent.network.params,
    )
    
    # Handle ensemble dimension (usually (2, Batch, Dim)) -> Mean
    if phi.ndim == 3:
        phi = np.array(phi).mean(axis=0)
        psi = np.array(psi).mean(axis=0)
    else:
        phi = np.array(phi)
        psi = np.array(psi)
        
    return phi, psi


def plot_feature_correlations(raw_obs, embeddings, out_path):
    """
    Computes correlation between raw physical state dimensions and the 
    Top 3 Principal Components of the embeddings.
    
    This answers: "What physical features does the critic care about?"
    """
    # 1. Compute PCA of embeddings (reduce to top 5 components)
    pca = PCA(n_components=5)
    embed_pca = pca.fit_transform(embeddings)
    
    # 2. Compute Spearman Correlation (monotonic relationship)
    # raw_obs is (N, D_obs), embed_pca is (N, 5)
    n_features = raw_obs.shape[1]
    n_components = embed_pca.shape[1]
    
    corr_matrix = np.zeros((n_components, n_features))
    
    for i in range(n_components):
        for j in range(n_features):
            corr, _ = spearmanr(embed_pca[:, i], raw_obs[:, j])
            corr_matrix[i, j] = corr
            
    # 3. Plot Heatmap
    plt.figure(figsize=(22, 6))
    

    x_labels = [f"Obs {i}" for i in range(n_features)]
    y_labels = [f"PC {i+1} ({var:.1%} var)" for i, var in enumerate(pca.explained_variance_ratio_)]
    
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0, 
                xticklabels=x_labels, yticklabels=y_labels)
    
    plt.title("Correlation: Embedding Principal Components vs Physical State")
    plt.xlabel("Physical Observation Dimensions")
    plt.ylabel("Critic Embedding Principal Components")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main(args):
    print(f"Loading dataset from {args.dataset_path}...")
    dataset_raw = load_dataset(args.dataset_path, compact_dataset=True)
    
    # 1. Normalization Statistics
    obs_data = dataset_raw["observations"]
    obs_mean = np.mean(obs_data, axis=0)
    obs_std = np.std(obs_data, axis=0)
    obs_std[obs_std < 1e-2] = 1.0

    # 2. Identify Trajectory Boundaries
    start_idxs, goal_idxs, all_idxs = get_indices(dataset_raw)
    
    print(f"Found {len(start_idxs)} trajectories.")
    
    # 3. Create Sample Batch
    # We want a mix of full trajectories to see the 'flow'
    # Let's pick N random trajectories and extract all their steps
    np.random.seed(42)
    selected_traj_starts = np.random.choice(start_idxs, size=min(len(start_idxs), args.num_trajectories), replace=False)
    
    batch_indices = []
    batch_times = [] # Relative timestep in trajectory
    
    for s_idx in selected_traj_starts:
        # Find the end of this trajectory (the next index in goal_idxs >= s_idx)
        # Assuming goal_idxs are sorted
        relevant_goals = goal_idxs[goal_idxs >= s_idx]
        if len(relevant_goals) == 0: continue
        e_idx = relevant_goals[0]
        
        traj_indices = np.arange(s_idx, e_idx + 1)
        batch_indices.extend(traj_indices)
        batch_times.extend(np.arange(len(traj_indices)))

    batch_indices = np.array(batch_indices)
    batch_times = np.array(batch_times)
    
    # Extract Data
    raw_obs = dataset_raw["observations"][batch_indices]
    raw_actions = dataset_raw["actions"][batch_indices]
    # For goals, in CRL training, we often use the trajectory goal or random future. 
    # To visualize the critic's "value" map, let's set the goal to the ACTUAL final state of the trajectory.
    # We need to construct a goal array where every step in a traj points to that traj's last step.
    raw_goals = np.zeros_like(raw_obs)
    
    current_idx = 0
    for s_idx in selected_traj_starts:
        relevant_goals = goal_idxs[goal_idxs >= s_idx]
        if len(relevant_goals) == 0: continue
        e_idx = relevant_goals[0]
        traj_len = e_idx - s_idx + 1
        
        final_state = dataset_raw["observations"][e_idx]
        raw_goals[current_idx : current_idx + traj_len] = final_state
        current_idx += traj_len
        
    # Normalize inputs for Agent
    norm_obs = normalize(raw_obs, obs_mean, obs_std)
    norm_goals = normalize(raw_goals, obs_mean, obs_std)
    
    # 4. Load Agent
    print(f"Loading agent from {args.agent_checkpoint}...")
    cfg = get_config()
    cfg = dict(cfg) # Config needs to match training
    cfg["alpha"] = 0.03 # Ensure this matches training or is a reasonable default
    
    # Create dummy structure to init agent
    dummy_obs = norm_obs[0:1]
    dummy_act = raw_actions[0:1]
    
    agent_tmp = CRLAgent.create(seed=0, ex_observations=dummy_obs, ex_actions=dummy_act, config=cfg)
    agent = restore_agent(agent_tmp, args.agent_checkpoint, args.step)
    
    # 5. Compute Embeddings
    print("Computing embeddings...")
    # Process in chunks to avoid OOM if batch is huge
    chunk_size = 1000
    phi_list = []
    psi_list = []
    
    num_samples = len(batch_indices)
    for i in range(0, num_samples, chunk_size):
        end = min(i + chunk_size, num_samples)
        p_chunk, s_chunk = extract_embeddings(
            agent, 
            norm_obs[i:end], 
            norm_goals[i:end], 
            raw_actions[i:end]
        )
        phi_list.append(p_chunk)
        psi_list.append(s_chunk)
        
    phi = np.concatenate(phi_list, axis=0)
    psi = np.concatenate(psi_list, axis=0)
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # Visualization 1: Manifold Topology (Time & Start/Goal)
    # -------------------------------------------------------------
    print("Running UMAP...")
    # Fit UMAP on Phi (state representation)
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
    phi_umap = reducer.fit_transform(phi)
    

    # -------------------------------------------------------------
    # Visualization 2: Correlation Analysis (Interpretability)
    # -------------------------------------------------------------
    print("Computing correlations...")
    plot_feature_correlations(raw_obs, phi, os.path.join(args.out_dir, "feature_correlation_phi.png"))

    print(f"Done. Results saved to {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to .npz dataset")
    parser.add_argument("--agent-checkpoint", type=str, required=True, help="Path to checkpoint folder")
    parser.add_argument("--step", type=int, default=50000, help="Checkpoint step to load")
    parser.add_argument("--out-dir", type=str, default="analysis_output")
    parser.add_argument("--num-trajectories", type=int, default=50, help="Number of trajectories to visualize")
    args = parser.parse_args()

    main(args)