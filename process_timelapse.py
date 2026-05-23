import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_ROTATION_DEG = -5.0
DEFAULT_CROP = (0.04, 0.06, 0.96, 0.94)
DEFAULT_TIMESTAMP_CROP_PX = 72
DEFAULT_DAY_THRESHOLD = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rotate each frame, crop the useful center area, "
            "remove the burned-in timestamp bar, and optionally keep only daytime frames."
        )
    )
    parser.add_argument(
        "--input",
        default="DSCF0001.AVI",
        help="Source AVI file path",
    )
    parser.add_argument(
        "--output",
        default="DSCF0001_cleaned.avi",
        help="Output video path",
    )
    parser.add_argument(
        "--rotation-deg",
        type=float,
        default=DEFAULT_ROTATION_DEG,
        help="Rotation angle in degrees. Negative values rotate clockwise.",
    )
    parser.add_argument(
        "--crop",
        type=float,
        nargs=4,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        default=DEFAULT_CROP,
        help="Crop rectangle as percentages of width/height after rotation.",
    )
    parser.add_argument(
        "--timestamp-crop-px",
        type=int,
        default=DEFAULT_TIMESTAMP_CROP_PX,
        help="Pixels to remove from the bottom after the center crop.",
    )
    parser.add_argument(
        "--day-threshold",
        type=float,
        default=DEFAULT_DAY_THRESHOLD,
        help="Optional brightness threshold for keeping only daytime frames.",
    )
    return parser.parse_args()


def rotate_frame(frame: np.ndarray, rotation_deg: float) -> np.ndarray:
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
    return cv2.warpAffine(
        frame,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def crop_frame(frame: np.ndarray, crop: tuple[float, float, float, float], timestamp_crop_px: int) -> np.ndarray:
    h, w = frame.shape[:2]
    left_p, top_p, right_p, bottom_p = crop

    x1 = int(w * left_p)
    y1 = int(h * top_p)
    x2 = int(w * right_p)
    y2 = int(h * bottom_p)

    cropped = frame[y1:y2, x1:x2]
    if timestamp_crop_px > 0:
        cropped = cropped[:-timestamp_crop_px, :]

    return cropped


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0

    writer = None
    frame_count = 0
    kept_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rotated = rotate_frame(frame, args.rotation_deg)
            cleaned = crop_frame(rotated, tuple(args.crop), args.timestamp_crop_px)

            if cleaned.size == 0:
                continue

            if args.day_threshold is not None:
                gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
                avg_brightness = float(np.mean(gray))
                if avg_brightness <= args.day_threshold:
                    frame_count += 1
                    continue

            if writer is None:
                height, width = cleaned.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise SystemExit(f"Could not open output video for writing: {output_path}")

            writer.write(cleaned)
            kept_count += 1
            frame_count += 1

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    print(f"Processed {frame_count} frames")
    print(f"Saved {kept_count} frames to {output_path}")


if __name__ == "__main__":
    main()
