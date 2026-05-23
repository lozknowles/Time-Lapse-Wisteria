import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

TORCH_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "torch"
os.environ.setdefault("TORCH_HOME", str(TORCH_CACHE_DIR))

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - handled at runtime
    YOLO = None


DEFAULT_ROTATION_DEG = 4.0
DEFAULT_CROP = (0.04, 0.06, 0.96, 0.94)
DEFAULT_TIMESTAMP_CROP_PX = 72
DEFAULT_BOTTOM_CROP_PX = 50
DEFAULT_DAY_THRESHOLD = 70.0
DEFAULT_START_SECONDS = 1.0
DEFAULT_OBJECT_ACTION = "discard"
DEFAULT_OBJECT_CONFIDENCE = 0.35
DEFAULT_INPAINT_RADIUS = 3
DEFAULT_MASK_PADDING = 18
DEFAULT_YOLO_MODEL = "yolov8n.pt"

COCO_PERSON = [0]
COCO_VEHICLES = [1, 2, 3, 5, 7]
COCO_ANIMALS = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]


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
        "--bottom-crop-px",
        type=int,
        default=DEFAULT_BOTTOM_CROP_PX,
        help="Extra pixels to remove from the bottom after timestamp cropping.",
    )
    parser.add_argument(
        "--day-threshold",
        type=float,
        default=DEFAULT_DAY_THRESHOLD,
        help="Brightness threshold for keeping only daytime frames.",
    )
    parser.add_argument(
        "--start-seconds",
        type=float,
        default=DEFAULT_START_SECONDS,
        help="Skip the first N seconds of the source video before processing.",
    )
    parser.add_argument(
        "--detect-people",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable person detection.",
    )
    parser.add_argument(
        "--detect-cars",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable vehicle detection.",
    )
    parser.add_argument(
        "--detect-animals",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable common animal detection.",
    )
    parser.add_argument(
        "--object-action",
        choices=("discard", "inpaint"),
        default=DEFAULT_OBJECT_ACTION,
        help="What to do when a person/car/animal is detected.",
    )
    parser.add_argument(
        "--object-confidence",
        type=float,
        default=DEFAULT_OBJECT_CONFIDENCE,
        help="Minimum YOLO confidence for detected objects.",
    )
    parser.add_argument(
        "--mask-padding",
        type=int,
        default=DEFAULT_MASK_PADDING,
        help="Extra pixels to expand detected boxes before masking or discarding.",
    )
    parser.add_argument(
        "--inpaint-radius",
        type=int,
        default=DEFAULT_INPAINT_RADIUS,
        help="Inpaint radius used when object-action is inpaint.",
    )
    parser.add_argument(
        "--yolo-model",
        default=DEFAULT_YOLO_MODEL,
        help="YOLO model weights to load, for example yolov8n.pt.",
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


def crop_frame(
    frame: np.ndarray,
    crop: tuple[float, float, float, float],
    timestamp_crop_px: int,
    bottom_crop_px: int,
) -> np.ndarray:
    h, w = frame.shape[:2]
    left_p, top_p, right_p, bottom_p = crop

    x1 = int(w * left_p)
    y1 = int(h * top_p)
    x2 = int(w * right_p)
    y2 = int(h * bottom_p)

    cropped = frame[y1:y2, x1:x2]
    if timestamp_crop_px > 0:
        cropped = cropped[:-timestamp_crop_px, :]
    if bottom_crop_px > 0:
        cropped = cropped[:-bottom_crop_px, :]

    return cropped


def build_target_classes(args: argparse.Namespace) -> list[int]:
    classes: set[int] = set()
    if args.detect_people:
        classes.update(COCO_PERSON)
    if args.detect_cars:
        classes.update(COCO_VEHICLES)
    if args.detect_animals:
        classes.update(COCO_ANIMALS)
    return sorted(classes)


def expand_boxes(boxes: list[tuple[int, int, int, int]], width: int, height: int, padding: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
    return mask


def detect_boxes(
    model: "YOLO",
    frame: np.ndarray,
    class_ids: list[int],
    confidence: float,
) -> list[tuple[int, int, int, int]]:
    predict_kwargs = {
        "imgsz": 640,
        "conf": confidence,
        "device": "cpu",
        "verbose": False,
    }
    if class_ids:
        predict_kwargs["classes"] = class_ids

    result = model.predict(frame, **predict_kwargs)[0]
    if result.boxes is None:
        return []

    boxes: list[tuple[int, int, int, int]] = []
    for xyxy in result.boxes.xyxy.cpu().numpy().astype(int):
        x1, y1, x2, y2 = map(int, xyxy.tolist())
        boxes.append((x1, y1, x2, y2))
    return boxes


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

    target_classes = build_target_classes(args)
    yolo_model = None
    if target_classes:
        if YOLO is None:
            raise SystemExit("ultralytics is not installed. Install it to use object detection.")
        yolo_model = YOLO(args.yolo_model)

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
    filtered_count = 0
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
                f"({percent}%) | kept {kept_count} | filtered {filtered_count} | {rate:.1f} fps"
            )
        else:
            elapsed = time.time() - start_time
            rate = frame_count / elapsed if elapsed > 0 else 0.0
            message = (
                f"\rProcessing {frame_count} frames "
                f"| kept {kept_count} | filtered {filtered_count} | {rate:.1f} fps"
            )
        print(message, end="", file=sys.stdout, flush=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rotated = rotate_frame(frame, rotation_deg)
            cleaned = crop_frame(
                rotated,
                tuple(args.crop),
                args.timestamp_crop_px,
                args.bottom_crop_px,
            )

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

            detected_boxes: list[tuple[int, int, int, int]] = []
            if yolo_model is not None and target_classes:
                detected_boxes = detect_boxes(
                    yolo_model,
                    cleaned,
                    target_classes,
                    args.object_confidence,
                )
                if detected_boxes and args.object_action == "discard":
                    filtered_count += 1
                    frame_count += 1
                    report()
                    continue
                if detected_boxes and args.object_action == "inpaint":
                    mask = expand_boxes(
                        detected_boxes,
                        cleaned.shape[1],
                        cleaned.shape[0],
                        args.mask_padding,
                    )
                    cleaned = cv2.inpaint(cleaned, mask, args.inpaint_radius, cv2.INPAINT_TELEA)
                    filtered_count += 1

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
