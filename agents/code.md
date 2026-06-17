# K-Means Image Segmentation Guide

## Purpose
Build a K-Means clustering algorithm from scratch in Python to segment colors in an image. Instead of using sklearn, implement the centroid update loop manually to understand how unsupervised clustering works.

## Input
- A color image file like JPEG or PNG in the data/raw folder. It defaults to picture.jpg or an auto-generated sample image.
- Target number of clusters K, typically 2, 3, 5, or 8.

## Output
- Preprocessed and resized image at data/processed/resized_image.png.
- Segmented images for each K value at data/processed/segmented_k*.png.
- A CSV containing performance and clustering metrics at data/processed/metrics/segmentation_metrics.csv.
- Visual comparison plots and convergence charts in the reports/figures folder.
- An analysis report at reports/image_segmentation_report.md.

## How to do
1. **Setup**: Run the pip install command for requirements.txt to install dependencies like numpy, pandas, matplotlib, pillow, opencv-python, scikit-learn, and joblib.
2. **Implement Core Logic** in the src folder:
   - kmeans.py: Write the KMeansFromScratch class containing centroid initialization, Euclidean distance calculation, cluster assignments, and centroid updates. Handle empty clusters if they happen. Add reconstruct_image function to map labels back to centroid colors.
   - image_io.py: Write helper functions to load, resize, convert color spaces, and save images.
   - visualization.py: Write helpers for plotting image comparisons, grid of different Ks, convergence rate of inertia history, and centroid palettes.
   - utils.py: Write helpers for directories, seeds, and logging.
3. **Execute Pipeline**:
   - Run the script python src/run_pipeline.py to process the raw image, run clustering for all K values, and export all outputs.
   - Or open and run notebooks/lab3_1.ipynb cell-by-cell for interactive visualizations and comparison against sklearn.
