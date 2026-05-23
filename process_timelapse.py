import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


DEFAULT_ROTATION_DEG = 4.0
DEFAULT_CROP = (0.04, 0.06, 0.96, 0.94)
DEFAULT_TIMESTAMP_CROP_PX = 72
DEFAULT_DAY_THRESHOLD = None
DEFAULT_START_SECONDS = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rotate each frame, crop the useful center area, "
            "remove the burned-in timestamp bar, and optionally keep only daytime frames."
        )
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Source video file path. If omitted, you will be prompted to choose one.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output video path. If omitted, a name will be derived from the input file.",
    )
    parser.add_argument(
        "--rotation-deg",
        type=float,
        default=DEFAULT_ROTATION_DEG,
        help="Rotation angle in degrees. Positive values rotate counterclockwise.",
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
    parser.add_argument(
        "--start-seconds",
        type=float,
        default=DEFAULT_START_SECONDS,
        help="Skip the first N seconds of the source video before processing.",
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


def prompt_for_input_path() -> Path:
    try:
        from tkinter import Tk, filedialog
    except Exception:
        response = input("Enter the video file path to render: ").strip('"')
        if not response:
            raise SystemExit("No input file selected.")
        return Path(response)

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askopenfilename(
        title="Select a video file to render",
        filetypes=[("Video files", "*.avi *.mp4 *.mov *.mkv"), ("All files", "*.*")],
    )
    root.destroy()
    if not selected:
        raise SystemExit("No input file selected.")
    return Path(selected)


def prompt_for_rotation(default_rotation: float) -> float:
    response = input(f"Rotation angle in degrees [{default_rotation}]: ").strip()
    if not response:
        return default_rotation
    try:
        return float(response)
    except ValueError:
        raise SystemExit("Rotation angle must be a number.")


def main() -> None:
    args = parse_args()

    input_path = Path(args.input) if args.input else prompt_for_input_path()
    rotation_deg = args.rotation_deg
    if args.rotation_deg == DEFAULT_ROTATION_DEG and "--rotation-deg" not in sys.argv:
        rotation_deg = prompt_for_rotation(DEFAULT_ROTATION_DEG)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_cleaned{input_path.suffix}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    start_frame = max(0, int(args.start_seconds * fps))
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        if total_frames > 0:
            total_frames = max(0, total_frames - start_frame)

    writer = None
    frame_count = 0
    kept_count = 0
    start_time = time.time()
    last_reported = -1

    def report(force: bool = False) -> None:
        nonlocal last_reported
        if total_frames > 0:
            percent = int((frame_count / total_frames) * 100)
            if not force and percent == last_reported:
                return
            last_reported = percent
            elapsed = time.time() - start_time
            rate = frame_count / elapsed if elapsed > 0 else 0.0
            message = (
                f"\rProcessing {frame_count}/{total_frames} frames "
                f"({percent}%) | kept {kept_count} | {rate:.1f} fps"
            )
        else:
            elapsed = time.time() - start_time
            rate = frame_count / elapsed if elapsed > 0 else 0.0
            message = (
                f"\rProcessing {frame_count} frames "
                f"| kept {kept_count} | {rate:.1f} fps"
            )
        print(message, end="", file=sys.stdout, flush=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rotated = rotate_frame(frame, rotation_deg)
            cleaned = crop_frame(rotated, tuple(args.crop), args.timestamp_crop_px)

            if cleaned.size == 0:
                frame_count += 1
                report()
                continue

            if args.day_threshold is not None:
                gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
                avg_brightness = float(np.mean(gray))
                if avg_brightness <= args.day_threshold:
                    frame_count += 1
                    report()
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
            report()

    finally:
        cap.release()
        if writer is not None:
            writer.release()

    report(force=True)
    print()
    print(f"Processed {frame_count} frames")
    print(f"Saved {kept_count} frames to {output_path}")


if __name__ == "__main__":
    main()
