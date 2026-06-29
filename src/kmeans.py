"""From-scratch K-Means implementation for image segmentation."""

from __future__ import annotations

import numpy as np


class KMeansFromScratch:
    """Manual K-Means clustering using vectorized NumPy distance updates."""

    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=42):
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1")
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        if self.n_clusters > len(X):
            raise ValueError("n_clusters cannot exceed number of samples")
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(len(X), size=self.n_clusters, replace=False)
        return X[indices].astype(float, copy=True)

    @staticmethod
    def _squared_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        return np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)

    def _assign_labels(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        distances = self._squared_distances(X, centers)
        return distances.argmin(axis=1)

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
        new_centers = centers.copy()
        distances_to_assigned = np.sum((X - centers[labels]) ** 2, axis=1)
        reusable_indices = np.argsort(distances_to_assigned)[::-1]
        used_reinitializers: set[int] = set()

        for cluster_id in range(self.n_clusters):
            members = X[labels == cluster_id]
            if len(members):
                new_centers[cluster_id] = members.mean(axis=0)
                continue

            for index in reusable_indices:
                if int(index) not in used_reinitializers:
                    new_centers[cluster_id] = X[index]
                    used_reinitializers.add(int(index))
                    break
        return new_centers

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D matrix with shape (n_samples, n_features)")

        centers = self._initialize_centroids(X)
        self.inertia_history_ = []
        self.center_shift_history_ = []
        self.centroids_history_ = [centers.copy()]

        for iteration in range(1, self.max_iter + 1):
            labels = self._assign_labels(X, centers)
            inertia = float(np.sum((X - centers[labels]) ** 2))
            new_centers = self._update_centroids(X, labels, centers)
            center_shift = float(np.linalg.norm(new_centers - centers))

            self.inertia_history_.append(inertia)
            self.center_shift_history_.append(center_shift)
            self.centroids_history_.append(new_centers.copy())
            centers = new_centers

            if center_shift <= self.tol:
                break

        self.cluster_centers_ = centers
        self.labels_ = self._assign_labels(X, centers)
        self.inertia_ = float(np.sum((X - centers[self.labels_]) ** 2))
        self.n_iter_ = iteration
        if not self.inertia_history_ or self.inertia_history_[-1] != self.inertia_:
            self.inertia_history_.append(self.inertia_)
        return self

    def predict(self, X):
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("This KMeansFromScratch instance is not fitted yet")
        X = np.asarray(X, dtype=float)
        return self._assign_labels(X, self.cluster_centers_)

    def fit_predict(self, X):
        return self.fit(X).labels_


def reconstruct_image(labels, centers, image_shape):
    """Map each pixel label to its centroid color and restore image shape."""
    segmented = centers[np.asarray(labels)].reshape(image_shape)
    return np.clip(np.rint(segmented), 0, 255).astype(np.uint8)


def compression_ratio(unique_colors_before: int, unique_colors_after: int) -> float:
    """Approximate color compression as after / before."""
    if unique_colors_before == 0:
        return 0.0
    return unique_colors_after / unique_colors_before


def compute_silhouette_score(X: np.ndarray, labels: np.ndarray, sample_size: int = 2000, random_state: int = 42) -> float:
    """Compute average silhouette score on a sample to avoid memory issues."""
    from sklearn.metrics import silhouette_score
    X = np.asarray(X, dtype=float)
    if len(X) > sample_size:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(X), size=sample_size, replace=False)
        X_sample = X[indices]
        labels_sample = labels[indices]
    else:
        X_sample = X
        labels_sample = labels
    return float(silhouette_score(X_sample, labels_sample))

