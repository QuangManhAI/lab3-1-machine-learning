"""Run the complete K-Means image segmentation exercise."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

from image_io import (
    convert_color_space,
    convert_to_rgb,
    flatten_pixels,
    image_summary,
    load_image_rgb,
    resize_image,
    save_image,
)
from kmeans import KMeansFromScratch, compression_ratio, reconstruct_image
from utils import ensure_dir, report_status, save_figure, save_table, setup_seed
from visualization import color_palette, compare_images, convergence_plot, display_image, k_comparison_grid


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANDOM_SEED = 42
K_VALUES = [2, 3, 5, 8]


def create_sample_image(path: Path, width: int = 640, height: int = 420) -> Path:
    """Create a deterministic natural-style image with varied color regions."""
    rng = np.random.default_rng(RANDOM_SEED)
    canvas = Image.new("RGB", (width, height), "#9ec9e8")
    draw = ImageDraw.Draw(canvas)

    for y in range(height):
        t = y / max(1, height - 1)
        sky = (
            int(118 + 88 * (1 - t)),
            int(170 + 42 * (1 - t)),
            int(215 + 28 * (1 - t)),
        )
        draw.line([(0, y), (width, y)], fill=sky)

    draw.ellipse((440, 42, 540, 142), fill="#f4d35e")
    draw.ellipse((456, 58, 524, 126), fill="#f7e08b")

    mountain_a = [(0, 270), (130, 120), (270, 270)]
    mountain_b = [(155, 280), (320, 105), (520, 280)]
    mountain_c = [(390, 275), (525, 135), (640, 275)]
    draw.polygon(mountain_a, fill="#6f8f78")
    draw.polygon(mountain_b, fill="#496f7a")
    draw.polygon(mountain_c, fill="#7a966f")
    draw.polygon([(95, 160), (130, 120), (168, 162)], fill="#dce8e6")
    draw.polygon([(286, 142), (320, 105), (365, 144)], fill="#e7eded")

    draw.rectangle((0, 268, width, height), fill="#437f95")
    for y in range(280, height, 11):
        color = (62 + int(rng.integers(-4, 5)), 126 + int(rng.integers(-7, 8)), 145)
        draw.arc((-80, y - 35, width + 80, y + 35), 0, 180, fill=color, width=2)

    draw.polygon([(0, 310), (172, 255), (270, height), (0, height)], fill="#5b8c53")
    draw.polygon([(640, 302), (500, 255), (396, height), (640, height)], fill="#6f914e")
    draw.rectangle((150, 270, 166, 335), fill="#76543a")
    draw.polygon([(113, 291), (158, 205), (203, 291)], fill="#275d3a")
    draw.polygon([(122, 254), (158, 184), (195, 254)], fill="#2f7043")
    draw.rectangle((496, 262, 511, 326), fill="#6d4d32")
    draw.polygon([(462, 280), (503, 208), (545, 280)], fill="#376d3b")
    draw.polygon([(470, 246), (503, 190), (536, 246)], fill="#3d7d45")

    noise = rng.normal(0, 4, size=(height, width, 3))
    array = np.clip(np.asarray(canvas, dtype=float) + noise, 0, 255).astype(np.uint8)
    smoothed = Image.fromarray(array).filter(ImageFilter.SMOOTH_MORE)
    path.parent.mkdir(parents=True, exist_ok=True)
    smoothed.save(path, quality=95)
    return path


def make_color_space_preview(rgb_image: np.ndarray, output_path: Path) -> None:
    previews = {"RGB": rgb_image}
    for space in ("HSV", "LAB"):
        converted = convert_color_space(rgb_image, space)
        previews[space] = convert_to_rgb(converted, space)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (space, image) in zip(axes, previews.items()):
        ax.imshow(image)
        ax.set_title(space)
        ax.axis("off")
    save_figure(fig, output_path)
    plt.close(fig)


def run_experiments(feature_image: np.ndarray, display_image_rgb: np.ndarray, output_dir: Path):
    pixels = flatten_pixels(feature_image)
    unique_before = image_summary(display_image_rgb)["unique_colors"]
    images_by_k = {}
    models_by_k = {}
    history_by_k = {}
    metrics = []

    for k in K_VALUES:
        report_status(f"Fitting K={k}")
        model = KMeansFromScratch(n_clusters=k, max_iter=100, tol=1e-3, random_state=RANDOM_SEED)
        model.fit(pixels)
        segmented_features = reconstruct_image(model.labels_, model.cluster_centers_, feature_image.shape)
        segmented_rgb = convert_to_rgb(segmented_features, "RGB")
        output_path = output_dir / f"segmented_k{k}.png"
        save_image(segmented_rgb, output_path)

        unique_after = image_summary(segmented_rgb)["unique_colors"]
        metrics.append(
            {
                "color_space": "RGB",
                "k": k,
                "inertia": round(model.inertia_, 3),
                "n_iter": model.n_iter_,
                "compression_ratio": round(compression_ratio(unique_before, unique_after), 6),
                "unique_colors_before": unique_before,
                "unique_colors_after": unique_after,
                "output_path": str(output_path.relative_to(PROJECT_ROOT)),
            }
        )
        images_by_k[k] = segmented_rgb
        models_by_k[k] = model
        history_by_k[k] = model.inertia_history_

    return images_by_k, models_by_k, history_by_k, pd.DataFrame(metrics)


def main() -> None:
    setup_seed(RANDOM_SEED)
    raw_dir = ensure_dir(PROJECT_ROOT / "data" / "raw")
    processed_dir = ensure_dir(PROJECT_ROOT / "data" / "processed")
    metrics_dir = ensure_dir(processed_dir / "metrics")
    figures_dir = ensure_dir(PROJECT_ROOT / "reports" / "figures")

    preferred_image = raw_dir / "picture.jpg"
    image_candidates = sorted(raw_dir.glob("*.[jp][pn]g")) + sorted(raw_dir.glob("*.jpeg"))
    if preferred_image.exists():
        image_path = preferred_image
    else:
        image_path = image_candidates[0] if image_candidates else create_sample_image(raw_dir / "sample_image.jpg")
    report_status(f"Using image: {image_path.relative_to(PROJECT_ROOT)}")

    original = load_image_rgb(image_path)
    resized = resize_image(original, max_size=350)
    save_image(resized, processed_dir / "resized_image.png")
    save_image(original, figures_dir / "original_image.png")
    make_color_space_preview(resized, figures_dir / "color_space_preview.png")

    images_by_k, models_by_k, history_by_k, metrics_df = run_experiments(resized, resized, processed_dir)
    save_table(metrics_df, metrics_dir / "segmentation_metrics.csv")

    fig = compare_images(resized, images_by_k[5], titles=("Original resized", "Segmented K=5"))
    save_figure(fig, figures_dir / "original_vs_segmented.png")
    plt.close(fig)

    fig = k_comparison_grid(images_by_k, original=resized, color_space="RGB")
    save_figure(fig, figures_dir / "k_comparison_grid.png")
    plt.close(fig)

    fig = convergence_plot(history_by_k)
    save_figure(fig, figures_dir / "convergence_plot.png")
    plt.close(fig)

    fig = color_palette(models_by_k[5].cluster_centers_, title="K=5 Centroid Palette")
    save_figure(fig, figures_dir / "color_palette_k5.png")
    plt.close(fig)

    report_status("Metrics")
    print(metrics_df)


if __name__ == "__main__":
    main()
