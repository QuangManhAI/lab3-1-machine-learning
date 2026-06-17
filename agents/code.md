# Codex Guide - Practice Exercise 3: K-Means Image Segmentation

## Project Goal

Build a complete image segmentation exercise using the K-Means algorithm.

The project should load an image, convert it into pixel feature vectors, cluster pixels into `K` color groups, replace every pixel with its cluster centroid color, and visualize the original and segmented images side by side.

Primary objective:

> Segment an image into `K` visually meaningful color regions using a from-scratch K-Means implementation.

Do not only call `sklearn.cluster.KMeans`. The core assignment/update loop should be implemented manually. A sklearn comparison can be included as an optional sanity check.

## Expected Repository Structure

Create or maintain this structure:

```text
lab3_1/
  README.md
  requirements.txt
  agents/
    code.md
  data/
    raw/
      sample_image.jpg
    processed/
      resized_image.png
      segmented_k2.png
      segmented_k3.png
      segmented_k5.png
      segmented_k8.png
      metrics/
        segmentation_metrics.csv
  notebooks/
    lab3_1.ipynb
    utils.py
    image_io.py
    kmeans.py
    visualization.py
  reports/
    figures/
      original_image.png
      color_space_preview.png
      k_comparison_grid.png
      convergence_plot.png
      original_vs_segmented.png
    image_segmentation_report.md
```

If no image is provided, use any small natural image and document its source. Prefer a landscape, object photo, or portrait with multiple visible colors.

## Implementation Principles

- Keep the raw image unchanged under `data/raw/`.
- Resize before clustering so the notebook runs quickly.
- Use deterministic behavior with `RANDOM_SEED = 42`.
- Implement K-Means from scratch:
  - initialize centroids
  - assign pixels to nearest centroid
  - update centroids
  - detect convergence
  - handle empty clusters
- Use vectorized NumPy operations where practical.
- Keep pixel values in a clear range:
  - either `0..255` as `float`
  - or normalized `0..1`
- Convert back to `uint8` before saving/displaying images.
- Save every important output image to `reports/figures/` or `data/processed/`.

## Required Dependencies

Use these packages:

```text
numpy
pandas
matplotlib
pillow
opencv-python
scikit-learn
joblib
```

PIL/Pillow is enough for loading images. OpenCV can be used for color space conversions such as RGB to HSV or LAB.

## Notebook Flow

Build `notebooks/lab3_1.ipynb` with this structure.

### Step 0 - Imports And Setup

Include:

- `PROJECT_ROOT`
- fixed `RANDOM_SEED = 42`
- local imports:
  - `utils`
  - `image_io`
  - `kmeans`
  - `visualization`
- print module paths for reproducibility
- create output folders if missing

### Step 1 - Load The Image

Load an image using PIL or OpenCV.

Requirements:

- read from `data/raw/`
- convert to RGB
- show image dimensions
- show dtype and pixel value range
- display the original image
- save a copy to `reports/figures/original_image.png`

If using OpenCV, remember OpenCV reads images as BGR by default. Convert BGR to RGB before displaying with matplotlib.

### Step 2 - Preprocess The Image

Preprocessing steps:

- resize image to a manageable size, for example max width `300` or `400`
- optionally compare RGB, HSV, and LAB color spaces
- flatten image from shape `(height, width, channels)` to `(height * width, channels)`
- cast pixel array to `float`

Expected notebook explanations:

- why resizing helps
- what each row in the flattened pixel matrix means
- why color space can affect clustering

Expected output:

```text
data/processed/resized_image.png
```

### Step 3 - Implement K-Means From Scratch

Create `notebooks/kmeans.py` with a class or functions for manual K-Means.

Required API:

```python
class KMeansFromScratch:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=42):
        ...

    def fit(self, X):
        ...
        return self

    def predict(self, X):
        ...

    def fit_predict(self, X):
        ...
```

The fitted object should expose:

- `cluster_centers_`
- `labels_`
- `inertia_`
- `n_iter_`
- `inertia_history_`
- `center_shift_history_`

Implementation details:

1. Initialize `K` random centroids by sampling pixels from the image.
2. Compute Euclidean distance from every pixel to every centroid.
3. Assign each pixel to the closest centroid.
4. Update each centroid as the mean of assigned pixels.
5. If a cluster becomes empty, reinitialize its centroid using a random pixel or the pixel farthest from existing centroids.
6. Stop when centroid shift is below `tol` or `max_iter` is reached.

Avoid loops over pixels. A loop over clusters is acceptable.

### Step 4 - Apply K-Means Clustering

Run experiments with several `K` values:

- `K = 2`
- `K = 3`
- `K = 5`
- `K = 8`

For each K:

- fit K-Means from scratch
- record inertia
- record number of iterations
- reconstruct segmented image
- save segmented image
- display segmented image

Expected outputs:

```text
data/processed/segmented_k2.png
data/processed/segmented_k3.png
data/processed/segmented_k5.png
data/processed/segmented_k8.png
data/processed/metrics/segmentation_metrics.csv
```

Suggested metrics table columns:

- `color_space`
- `k`
- `inertia`
- `n_iter`
- `compression_ratio`
- `unique_colors_before`
- `unique_colors_after`
- `output_path`

### Step 5 - Reconstruct Segmented Images

For each pixel:

- take its assigned cluster id
- replace it with the RGB color of the corresponding centroid
- reshape back to `(height, width, 3)`
- clip values to `[0, 255]`
- convert to `uint8`

Create a reusable helper:

```python
def reconstruct_image(labels, centers, image_shape):
    ...
```

The segmented image should visually contain exactly `K` dominant colors, or at most `K` unique centroid colors after rounding.

### Step 6 - Visualize Results

Create clean visual comparisons:

- original vs segmented side by side
- grid of multiple K values
- convergence plot of inertia vs iteration
- optional color palette plot showing cluster centroid colors

Required figure:

```text
reports/figures/original_vs_segmented.png
```

Recommended figures:

```text
reports/figures/k_comparison_grid.png
reports/figures/convergence_plot.png
reports/figures/color_palette_k5.png
```

The notebook should explain:

- low K gives posterized/blocky segmentation
- higher K keeps more detail
- K controls the trade-off between compression and visual fidelity

### Step 7 - Compare Color Spaces

Optional but recommended.

Run K-Means on:

- RGB
- HSV
- LAB

Use the same `K`, for example `K=5`.

Explain differences:

- RGB clusters raw color intensity directly.
- HSV can group by hue more intuitively.
- LAB can be more perceptually meaningful for color distance.

If using HSV or LAB for clustering, convert centroid output back to RGB before displaying.

### Step 8 - Optional Sklearn Check

Optionally run sklearn KMeans with the same K and compare:

- inertia
- runtime
- visual result
- centroid colors

This is only a checker. The final exercise must still include the from-scratch implementation.

### Step 9 - Save Report

Write:

```text
reports/image_segmentation_report.md
```

Report should include:

- image used
- preprocessing size
- selected color space
- K values tested
- final selected K
- visual interpretation
- limitations
- possible improvements

## Module Responsibilities

### `notebooks/utils.py`

General helpers:

- path creation
- random seed setup
- save table helper
- save figure helper
- report status helper

### `notebooks/image_io.py`

Image loading and preprocessing:

- load image as RGB
- resize image while preserving aspect ratio
- convert color spaces
- flatten image
- save image
- compute image summary

Expected functions:

```python
def load_image_rgb(path):
    ...

def resize_image(image, max_size=350):
    ...

def flatten_pixels(image):
    ...

def convert_color_space(image, color_space="RGB"):
    ...

def save_image(image, path):
    ...
```

### `notebooks/kmeans.py`

From-scratch K-Means:

- `KMeansFromScratch`
- distance computation
- centroid initialization
- assignment step
- update step
- convergence checking
- image reconstruction
- metrics generation

### `notebooks/visualization.py`

Plotting:

- display original image
- side-by-side image comparison
- K comparison grid
- convergence plot
- centroid color palette

## K-Means Implementation Notes

Distance computation can use broadcasting:

```python
distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
labels = distances.argmin(axis=1)
```

Inertia:

```python
inertia = np.sum((X - centers[labels]) ** 2)
```

Convergence:

```python
center_shift = np.linalg.norm(new_centers - centers)
if center_shift <= tol:
    break
```

Empty cluster handling:

- If cluster `j` has no pixels, do not crash.
- Reinitialize that centroid with a random pixel.
- Document that empty clusters can happen when K is high or initialization is unlucky.

## Evaluation Guidance

Image segmentation is mostly visual, but still report simple metrics:

- inertia
- number of iterations
- number of unique colors after segmentation
- compression ratio approximation:

```text
unique colors after / unique colors before
```

Important:

- Lower inertia is not always visually better because increasing K naturally lowers inertia.
- Choose final K based on visual quality and simplicity.
- For classroom exercises, `K=5` or `K=8` often gives a good balance.

## Visual Quality Requirements

All plots should be clean:

- no axis ticks for image displays
- titles include K and color space
- side-by-side comparison uses the same image size
- figures are saved with high enough DPI
- no stretched aspect ratio

Use `plt.tight_layout()` before saving.

## Acceptance Criteria

The project is complete when:

- Image loads correctly.
- Image is converted to RGB and resized.
- Pixel matrix has shape `(n_pixels, n_features)`.
- K-Means is implemented from scratch.
- Assignment and centroid update steps are visible in code.
- Multiple K values are tested.
- Segmented images are reconstructed and saved.
- Original and segmented images are displayed side by side.
- Inertia/convergence metrics are recorded.
- A final K is selected and explained.
- Notebook runs top to bottom without errors.
- Report and figures are saved.

## Recommended Development Order For Codex

1. Inspect the current folder.
2. Create the expected directory structure.
3. Add/update `requirements.txt`.
4. Add a sample image or document where to place it.
5. Implement image loading and resizing helpers.
6. Implement `KMeansFromScratch`.
7. Add reconstruction helper.
8. Build notebook step by step.
9. Run K values `2, 3, 5, 8`.
10. Save segmented images and metric table.
11. Add visual comparison figures.
12. Write the final report.
13. Update README with run instructions.

## Notes For Future Codex Runs

- If no image exists in `data/raw/`, use a generated or public-domain image and save it as `sample_image.jpg`.
- Keep image size small enough for fast iteration.
- Avoid using huge images directly; K-Means on millions of pixels will be slow.
- Prefer RGB first, then optionally compare HSV/LAB.
- Keep from-scratch implementation as the main result.
- Use sklearn only as a comparison, not as a replacement.
