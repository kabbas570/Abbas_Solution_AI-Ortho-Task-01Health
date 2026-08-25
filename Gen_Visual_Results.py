import argparse
import json
from pathlib import Path

import numpy as np

from contract import IDENTITY_Q, qconj, qnormalize, qrot


def apply_transform(points, centroid, translation, rotation):
    centered = points - centroid
    rotated = qrot(rotation, centered)
    return rotated + centroid + translation


def load_transforms(path):
    with open(path, encoding="utf-8") as file:
        data = json.load(file)

    transforms = data.get("transforms", data)

    return {
        int(tooth_id): (
            np.asarray(value["t_mm"], dtype=np.float64),
            qnormalize(np.asarray(value["q"], dtype=np.float64)),
        )
        for tooth_id, value in transforms.items()
    }


def make_axes(centroid, frame_q, transform_q, axis_length=1.0):
    """Return Napari vectors as ``[start, direction]`` pairs."""
    local_axes = np.eye(3) * axis_length
    original_axes = qrot(qconj(frame_q), local_axes)
    rotated_axes = qrot(transform_q, original_axes)
    original_vectors = np.stack(
        [np.stack([centroid, axis]) for axis in original_axes]
    )
    new_centroid = centroid
    rotated_vectors = np.stack(
        [np.stack([new_centroid, axis]) for axis in rotated_axes]
    )
    return original_vectors, rotated_vectors


def add_view_titles(viewer, original_centroids, gold_centroids, predicted_centroids):
    """Add large labels above the three separated views."""
    groups = [
        ("Original", np.asarray(original_centroids), "white"),
        ("Gold transform", np.asarray(gold_centroids), "yellow"),
        ("Model prediction", np.asarray(predicted_centroids), "magenta"),
    ]
    positions = []
    labels = []
    colors = []
    z_top = max(float(points[:, 2].max()) for _, points, _ in groups) + 10.0
    for label, points, color in groups:
        center = points.mean(axis=0)
        positions.append([center[0], center[1], z_top])
        labels.append(label)
        colors.append(color)
    viewer.add_points(
        np.asarray(positions),
        size=0.01,
        opacity=0.0,
        face_color=colors,
        text={"string": labels, "size": 18, "color": colors, "anchor": "center"},
        name="view_titles",
    )


def save_rotating_gif(original_points, gold_points, predicted_points, output_path, frames=72):
    """Save an offline rotating GIF of the three separated point-cloud views."""
    try:
        import imageio.v2 as imageio
        import matplotlib.pyplot as plt
    except ImportError as error:
        print(f"Skipping GIF export because a dependency is missing: {error}")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original = np.concatenate(original_points)
    gold = np.concatenate(gold_points)
    predicted = np.concatenate(predicted_points)
    all_points = np.concatenate([original, gold, predicted])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)

    images = []
    for frame in range(frames):
        fig = plt.figure(figsize=(11, 5.5), dpi=110)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(original[:, 0], original[:, 1], original[:, 2], s=0.08, c="royalblue", alpha=0.28)
        ax.scatter(gold[:, 0], gold[:, 1], gold[:, 2], s=0.08, c="limegreen", alpha=0.42)
        ax.scatter(predicted[:, 0], predicted[:, 1], predicted[:, 2], s=0.08, c="magenta", alpha=0.42)

        for label, points, color in [
            ("Original", original, "royalblue"),
            ("Gold transform", gold, "green"),
            ("Model prediction", predicted, "magenta"),
        ]:
            group_center = points.mean(axis=0)
            ax.text(group_center[0], group_center[1], maxs[2] + 8.0, label, color=color, fontsize=12, ha="center")

        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_axis_off()
        ax.view_init(elev=18, azim=frame * 360.0 / frames)
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        images.append(image)
        plt.close(fig)

    imageio.mimsave(output_path, images, duration=0.06)
    print(f"Saved rotating GIF: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        type=Path,
        default=Path(
            r"C:\Users\ARayabat\Downloads\AI-Ortho-Task-01Health-main"
            r"\AI-Ortho-Task-01Health-main\train\prod_0245"
        ),
    )
    parser.add_argument(
        "--points-per-tooth",
        type=int,
        default=1048,
        help="Maximum points displayed per tooth; use 128 or 256 for a faster view.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=0.6,
        help="Displayed point size in Napari units.",
    )
    parser.add_argument(
        "--hide-labels",
        action="store_true",
        help="Do not render FDI text labels.",
    )
    parser.add_argument(
        "--display-gap",
        type=float,
        default=60.0,
        help="Artificial X-axis gap between before, gold, and prediction views in mm.",
    )
    parser.add_argument(
        "--plan",
        "--predicted-plan",
        dest="plan",
        type=Path,
        default=Path(
            r"C:\Users\ARayabat\Downloads\AI-Ortho-Task-01Health-main"
            r"\AI-Ortho-Task-01Health-main\submissions"
            r"\train_predictions_with_llm\prod_0245__i0.json"
        ),
        help="Model prediction task-plan JSON shown as the third scan.",
    )
    parser.add_argument(
        "--gif",
        type=Path,
        default=None,
        help="Output GIF path. Defaults to viz_result/<case>_comparison.gif.",
    )
    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Disable automatic rotating GIF export.",
    )
    parser.add_argument(
        "--show-napari",
        action="store_true",
        help="Open an interactive Napari viewer after saving the GIF.",
    )
    args = parser.parse_args()

    if args.points_per_tooth < 1 or args.point_size <= 0 or args.display_gap <= 0:
        raise ValueError("--points-per-tooth, --point-size, and --display-gap must be positive")

    points_path = args.case / "points.npz"
    teeth_path = args.case / "teeth.json"
    transforms_path = args.case / "gold_transforms.json"
    if not args.plan.exists():
        raise FileNotFoundError(f"Predicted plan does not exist: {args.plan}")

    points_data = np.load(points_path)

    with open(teeth_path, encoding="utf-8") as file:
        teeth_metadata = json.load(file)

    transforms = load_transforms(transforms_path)
    predicted_transforms = load_transforms(args.plan)
    print(f"before: {points_path}")
    print(f"gold:   {transforms_path}")
    print(f"pred:   {args.plan}")

    original_centroids = []
    gold_centroids = []
    predicted_centroids = []
    original_labels = []
    gold_labels = []
    predicted_labels = []
    original_points = []
    gold_points_all = []
    predicted_points_all = []
    original_axes_all = []
    gold_axes_all = []
    predicted_axes_all = []
    gold_offset = np.array([args.display_gap, 0.0, 0.0])
    predicted_offset = np.array([2.0 * args.display_gap, 0.0, 0.0])

    for tooth_id in sorted(teeth_metadata, key=int):
        fdi = int(tooth_id)
        points = np.asarray(points_data[f"t{fdi}"], dtype=np.float64)
        if len(points) > args.points_per_tooth:
            sample_indices = np.linspace(
                0, len(points) - 1, args.points_per_tooth, dtype=int
            )
            points = points[sample_indices]

        centroid = np.asarray(
            teeth_metadata[tooth_id]["centroid"],
            dtype=np.float64,
        )
        frame_q = qnormalize(
            np.asarray(
                teeth_metadata[tooth_id]["frame_q"],
                dtype=np.float64,
            )
        )

        translation, rotation = transforms.get(
            fdi,
            (np.zeros(3), IDENTITY_Q),
        )

        gold_points = apply_transform(
            points,
            centroid,
            translation,
            rotation,
        )

        # This offset is visual only. It separates the two mouths in Napari;
        # it is not part of the clinical transform or the saved plan.
        gold_centroid = centroid + translation + gold_offset
        gold_points = gold_points + gold_offset

        original_points.append(points)
        gold_points_all.append(gold_points)

        original_centroids.append(centroid)
        gold_centroids.append(gold_centroid)
        original_labels.append(f"B{fdi}")
        gold_labels.append(f"G{fdi}")

        original_axes, rotated_axes = make_axes(
            centroid,
            frame_q,
            rotation,
        )

        original_axes_all.append(original_axes)
        gold_axes_all.append(
            rotated_axes
            + np.array([translation + gold_offset, np.zeros(3)])
        )

        pred_translation, pred_rotation = predicted_transforms.get(
            fdi,
            (np.zeros(3), IDENTITY_Q),
        )
        predicted_points = apply_transform(
            points,
            centroid,
            pred_translation,
            pred_rotation,
        )
        predicted_points = predicted_points + predicted_offset
        predicted_centroid = centroid + pred_translation + predicted_offset
        predicted_points_all.append(predicted_points)
        predicted_centroids.append(predicted_centroid)
        predicted_labels.append(f"P{fdi}")
        _, predicted_axes = make_axes(
            centroid,
            frame_q,
            pred_rotation,
        )
        predicted_axes_all.append(
            predicted_axes
            + np.array([pred_translation + predicted_offset, np.zeros(3)])
        )

    if not args.no_gif:
        gif_path = args.gif or (Path("viz_result") / f"{args.case.name}_comparison.gif")
        save_rotating_gif(original_points, gold_points_all, predicted_points_all, gif_path)

    if args.show_napari:
        import napari

        viewer = napari.Viewer(ndisplay=3)

        # Use a few aggregate layers instead of one layer per tooth. This is much
        # faster for Napari and still keeps before/after visually distinguishable.
        viewer.add_points(
            np.concatenate(original_points),
            size=args.point_size,
            face_color="blue",
            opacity=0.35,
            name="before_all_teeth",
        )
        viewer.add_points(
            np.concatenate(gold_points_all),
            size=args.point_size,
            face_color="green",
            opacity=0.65,
            name="gold_all_teeth",
        )
        viewer.add_vectors(
            np.concatenate(original_axes_all),
            edge_color="blue",
            edge_width=1,
            name="original_axes",
        )
        viewer.add_vectors(
            np.concatenate(gold_axes_all),
            edge_color="yellow",
            edge_width=1,
            name="gold_axes_after",
        )

        viewer.add_points(
            np.asarray(original_centroids),
            size=1.5,
            face_color="red",
            **({"text": {"string": original_labels, "size": 10}} if not args.hide_labels else {}),
            name="original_centroids",
        )

        viewer.add_points(
            np.asarray(gold_centroids),
            size=1.5,
            face_color="yellow",
            **({"text": {"string": gold_labels, "size": 10}} if not args.hide_labels else {}),
            name="gold_centroids",
        )

        viewer.add_points(
            np.concatenate(predicted_points_all),
            size=args.point_size,
            face_color="magenta",
            opacity=0.65,
            name="predicted_all_teeth",
        )
        viewer.add_vectors(
            np.concatenate(predicted_axes_all),
            edge_color="magenta",
            edge_width=1,
            name="predicted_axes_after",
        )
        viewer.add_points(
            np.asarray(predicted_centroids),
            size=1.5,
            face_color="magenta",
            **({"text": {"string": predicted_labels, "size": 10}} if not args.hide_labels else {}),
            name="predicted_centroids",
        )

        add_view_titles(viewer, original_centroids, gold_centroids, predicted_centroids)
        napari.run()


if __name__ == "__main__":
    main()