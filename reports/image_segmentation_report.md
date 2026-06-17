# Image Segmentation Report

## Image Used

The source image is `data/raw/picture.jpg`, provided by the user. The raw image is kept unchanged in `data/raw/`.

## Preprocessing

- Original image size: 1920 x 2560 pixels.
- Resized image path: `data/processed/resized_image.png`.
- Resized image size: 262 x 350 pixels.
- Pixel features: RGB values flattened from `(height, width, 3)` to `(height * width, 3)`.
- Pixel range: `0..255`, stored as float during clustering and converted back to `uint8` for saved images.

Resizing reduces the number of pixels clustered, which keeps the manual K-Means loop fast while preserving the main color regions.

## Selected Color Space

RGB is used for the main segmentation results. The notebook also saves `reports/figures/color_space_preview.png` to show RGB, HSV, and LAB representations. RGB is simple and direct for this classroom exercise, while HSV and LAB can be useful when hue grouping or perceptual color distance matters more.

## K Values Tested

| K | Inertia | Iterations | Unique Colors After | Output |
|---:|---:|---:|---:|---|
| 2 | 261719603.662 | 6 | 2 | `data/processed/segmented_k2.png` |
| 3 | 126918500.780 | 16 | 3 | `data/processed/segmented_k3.png` |
| 5 | 69047213.344 | 30 | 5 | `data/processed/segmented_k5.png` |
| 8 | 41557213.509 | 49 | 8 | `data/processed/segmented_k8.png` |

The complete metrics table is saved at `data/processed/metrics/segmentation_metrics.csv`.

## Final Selected K

The selected result is `K=5`. It gives a good balance for this image: the main color masses remain readable, while the output is still strongly compressed to five centroid colors.

## Visual Interpretation

- `K=2` creates a very coarse posterized image and merges several natural regions together.
- `K=3` separates more structure but still loses many midtones.
- `K=5` captures the main scene regions without preserving unnecessary texture.
- `K=8` keeps more detail and lowers inertia, but the extra clusters make the result less simple.

Increasing K naturally lowers inertia, so the lowest inertia is not automatically the best segmentation. The final choice should balance visual quality, simplicity, and the goal of color reduction.

## Limitations

- K-Means only uses color features, not pixel position, texture, or object boundaries.
- Euclidean RGB distance does not always match human color perception.
- Random centroid initialization can affect results, although this project fixes `RANDOM_SEED = 42`.
- The resized image is used for speed, so very fine details from the original 1920 x 2560 image are intentionally reduced before clustering.

## Possible Improvements

- Compare RGB, HSV, and LAB segmentations visually and quantitatively.
- Add spatial features such as normalized x/y coordinates.
- Try K-Means++ initialization.
- Use multiple random restarts and keep the model with the lowest inertia.
- Add silhouette-style diagnostics or perceptual color-difference metrics.
