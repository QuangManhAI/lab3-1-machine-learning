"""Plotting helpers for image segmentation outputs."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np


def display_image(image, title="Image", figsize=(6, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    ax.set_title(title)
    ax.axis("off")
    return fig


def compare_images(original, segmented, titles=("Original", "Segmented"), figsize=(10, 4)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, image, title in zip(axes, (original, segmented), titles):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    return fig


def k_comparison_grid(images_by_k, original=None, color_space="RGB", figsize=(12, 7)):
    total = len(images_by_k) + (1 if original is not None else 0)
    cols = min(3, total)
    rows = math.ceil(total / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    index = 0
    if original is not None:
        axes[index].imshow(original)
        axes[index].set_title("Original")
        axes[index].axis("off")
        index += 1

    for k, image in images_by_k.items():
        axes[index].imshow(image)
        axes[index].set_title(f"K={k} ({color_space})")
        axes[index].axis("off")
        index += 1

    for ax in axes[index:]:
        ax.axis("off")
    return fig


def convergence_plot(history_by_k, figsize=(8, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    for k, history in history_by_k.items():
        ax.plot(range(1, len(history) + 1), history, marker="o", linewidth=1.8, label=f"K={k}")
    ax.set_title("K-Means Convergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Inertia")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return fig


def color_palette(centers, title="Cluster Centroid Palette", figsize=(7, 1.6)):
    centers = np.clip(np.rint(centers), 0, 255).astype(np.uint8)
    palette = centers[np.newaxis, :, :]
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(palette)
    ax.set_title(title)
    ax.set_xticks(range(len(centers)))
    ax.set_yticks([])
    ax.set_xlabel("Cluster")
    return fig


def elbow_plot(k_values, inertias, figsize=(8, 5)):
    """Plot final inertia against K values to visualize the elbow method."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(k_values, inertias, marker="o", markersize=8, color="b", linestyle="-", linewidth=2)
    ax.set_title("Elbow Method for Optimal K")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.25)
    return fig


def silhouette_plot(X, labels, sample_size=2000, random_state=42, figsize=(8, 6)):
    """Plot silhouette coefficients sorted for each cluster with the average score line."""
    import matplotlib.cm as cm
    from sklearn.metrics import silhouette_samples, silhouette_score

    # Random sampling to avoid OOM
    X = np.asarray(X, dtype=float)
    if len(X) > sample_size:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(X), size=sample_size, replace=False)
        X_sample = X[indices]
        labels_sample = labels[indices]
    else:
        X_sample = X
        labels_sample = labels

    n_clusters = len(np.unique(labels_sample))
    
    # Compute the silhouette scores for each sample
    silhouette_vals = silhouette_samples(X_sample, labels_sample)
    avg_score = silhouette_score(X_sample, labels_sample)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_vals = silhouette_vals[labels_sample == i]
        ith_cluster_silhouette_vals.sort()
        
        size_cluster_i = ith_cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_vals,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        
        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10
        
    ax.set_title("Silhouette Plot for the Various Clusters")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    
    # The vertical line for average silhouette score of all the values
    ax.axvline(x=avg_score, color="red", linestyle="--", label=f"Average ({avg_score:.3f})")
    
    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="upper right")
    
    return fig, avg_score

