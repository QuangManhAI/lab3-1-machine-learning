# Lab 3.1 - K-Means Image Segmentation

This project implements image segmentation with K-Means from scratch. It loads an RGB image, resizes it, converts pixels into color feature vectors, clusters the pixels for several K values, reconstructs segmented images from centroid colors, and saves figures plus metrics.

The core K-Means assignment/update loop is implemented manually in `src/kmeans.py`. Sklearn is used only as an optional comparison inside the notebook.

## Structure

```text
lab3_1/
  agents/code.md
  data/raw/picture.jpg
  data/processed/resized_image.png
  data/processed/segmented_k2.png
  data/processed/segmented_k3.png
  data/processed/segmented_k5.png
  data/processed/segmented_k8.png
  data/processed/metrics/segmentation_metrics.csv
  notebooks/lab3_1.ipynb
  src/image_io.py
  src/kmeans.py
  src/run_pipeline.py
  src/utils.py
  src/visualization.py
  reports/figures/
  reports/image_segmentation_report.md
```

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run The Pipeline

```bash
.venv/bin/python src/run_pipeline.py
```

This uses the first image in `data/raw/` (currently `picture.jpg`), creates a deterministic sample image only if `data/raw/` is empty, runs K-Means for `K = 2, 3, 5, 8`, and saves all processed images, figures, and metrics.

## Run The Notebook

```bash
.venv/bin/jupyter nbconvert --to notebook --execute notebooks/lab3_1.ipynb --output lab3_1.ipynb --output-dir notebooks
```

You can also open `notebooks/lab3_1.ipynb` in Jupyter and run it cell by cell.

## Main Outputs

- `data/processed/segmented_k2.png`
- `data/processed/segmented_k3.png`
- `data/processed/segmented_k5.png`
- `data/processed/segmented_k8.png`
- `data/processed/metrics/segmentation_metrics.csv`
- `reports/figures/original_vs_segmented.png`
- `reports/figures/k_comparison_grid.png`
- `reports/figures/convergence_plot.png`
- `reports/image_segmentation_report.md`

The selected result is `K=5`, which balances visual simplicity and scene detail for the provided image.
