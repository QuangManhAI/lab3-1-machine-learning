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
