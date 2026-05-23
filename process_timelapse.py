import argparse
from datetime import datetime
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

TORCH_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "torch"
os.environ.setdefault("TORCH_HOME", str(TORCH_CACHE_DIR))

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - handled at runtime
    YOLO = None

try:
    from easyocr import Reader as EasyOCRReader
except ImportError:  # pragma: no cover - handled at runtime
    EasyOCRReader = None


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
DEFAULT_OVERLAY_TIMESTAMP = True
DEFAULT_TIMESTAMP_FONT_SIZE = 28
DEFAULT_TIMESTAMP_MARGIN = 16
DEFAULT_TIMESTAMP_BOX_ALPHA = 120
DEFAULT_TIMESTAMP_OCR_TOP_FRAC = 0.88
DEFAULT_TIMESTAMP_OCR_LEFT_FRAC = 0.75
DEFAULT_TIMESTAMP_PANEL_PADDING_X = 18
DEFAULT_TIMESTAMP_PANEL_PADDING_Y = 14
DEFAULT_TIMESTAMP_PANEL_WIDTH_FRAC = 0.28
DEFAULT_TIMESTAMP_PANEL_HEIGHT_FRAC = 0.17

COCO_PERSON = [0]
COCO_VEHICLES = [1, 2, 3, 5, 7]
COCO_ANIMALS = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

DATE_RE = re.compile(r"\d{4}/\d{2}/\d{2}")
TIME_RE = re.compile(r"\d{2}:\d{2}:\d{2}")
TEMP_RE = re.compile(r"\d+(?:\s+\d+)*\s*[CF]")
TEMP_VALUE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:°\s*)?([CF])", re.IGNORECASE)

SEGMENT_MAP = {
    "0": "ab cdef".replace(" ", ""),
    "1": "bc",
    "2": "abged",
    "3": "abgcd",
    "4": "fgbc",
    "5": "afgcd",
    "6": "afgecd",
    "7": "abc",
    "8": "abcdefg",
    "9": "abfgcd",
}


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
    parser.add_argument(
        "--overlay-timestamp",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_OVERLAY_TIMESTAMP,
        help="Overlay the source date/time at the bottom-left of each kept frame.",
    )
    parser.add_argument(
        "--timestamp-font-size",
        type=int,
        default=DEFAULT_TIMESTAMP_FONT_SIZE,
        help="Font size for the timestamp overlay.",
    )
    parser.add_argument(
        "--timestamp-margin",
        type=int,
        default=DEFAULT_TIMESTAMP_MARGIN,
        help="Margin in pixels for the timestamp overlay from the bottom-left corner.",
    )
    parser.add_argument(
        "--timestamp-box-alpha",
        type=int,
        default=DEFAULT_TIMESTAMP_BOX_ALPHA,
        help="Opacity of the timestamp box background, 0-255.",
    )
    parser.add_argument(
        "--timestamp-ocr-top-frac",
        type=float,
        default=DEFAULT_TIMESTAMP_OCR_TOP_FRAC,
        help="Top fraction of the frame used for OCR of the timestamp strip.",
    )
    parser.add_argument(
        "--timestamp-ocr-left-frac",
        type=float,
        default=DEFAULT_TIMESTAMP_OCR_LEFT_FRAC,
        help="Left fraction of the frame used for OCR so the right-side logo is excluded.",
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


def load_timestamp_reader() -> "EasyOCRReader | None":
    if EasyOCRReader is None:
        return None
    return EasyOCRReader(["en"], gpu=False, verbose=False)


def extract_timestamp_parts(
    reader: "EasyOCRReader",
    frame: np.ndarray,
    top_frac: float,
    left_frac: float,
) -> tuple[str | None, str | None, str | None]:
    h, w = frame.shape[:2]
    y1 = min(max(0, int(h * top_frac)), h - 1)
    x2 = min(max(1, int(w * left_frac)), w)
    roi = frame[y1:h, 0:x2]
    fragments = reader.readtext(
        roi,
        detail=0,
        paragraph=False,
        allowlist="0123456789/:.CFcf° ",
    )

    date_text = None
    time_text = None
    temp_text = None
    for fragment in fragments:
        cleaned = re.sub(r"\s+", "", fragment)
        if date_text is None:
            match = DATE_RE.search(cleaned)
            if match:
                date_text = match.group(0)
        if time_text is None:
            match = TIME_RE.search(cleaned)
            if match:
                time_text = match.group(0)
        if temp_text is None:
            normalized = re.sub(r"\s+", " ", fragment).strip()
            match = TEMP_VALUE_RE.search(normalized)
            if match:
                temp_text = f"{match.group(1)}°{match.group(2).upper()}"
    if temp_text is None and fragments:
        # Fall back to the most likely temperature fragment when OCR sees the line but not the unit.
        for fragment in fragments:
            normalized = re.sub(r"\s+", " ", fragment).strip()
            match = TEMP_VALUE_RE.search(normalized)
            if match:
                temp_text = f"{match.group(1)}°{match.group(2).upper()}"
                break
    return date_text, time_text, temp_text


def find_monospace_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\consolab.ttf",
        r"C:\Windows\Fonts\cour.ttf",
        r"C:\Windows\Fonts\courbd.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, font_size)
    return ImageFont.load_default()


def find_ui_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf",
        r"C:\Windows\Fonts\seguiemj.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, font_size)
    return ImageFont.load_default()


def parse_timestamp_fields(
    date_text: str | None,
    time_text: str | None,
    temp_text: str | None,
) -> dict[str, str] | None:
    if date_text is None or time_text is None:
        return None

    try:
        dt = datetime.strptime(f"{date_text} {time_text}", "%Y/%m/%d %H:%M:%S")
    except ValueError:
        return None

    hour = dt.hour
    if 5 <= hour < 12:
        period = "MORNING"
    elif 12 <= hour < 17:
        period = "AFTERNOON"
    elif 17 <= hour < 21:
        period = "EVENING"
    else:
        period = "NIGHT"

    time_24h = dt.strftime("%H:%M")

    temp_label = temp_text or ""
    if temp_label:
        match = TEMP_VALUE_RE.search(temp_label)
        if match:
            temp_label = f"{match.group(1)}{match.group(2).upper()}"
        else:
            temp_label = temp_label.replace(" ", "").replace("°", "")

    return {
        "weekday": dt.strftime("%A").upper(),
        "period": period,
        "time_24h": time_24h,
        "day": dt.strftime("%d").lstrip("0") or "0",
        "month": dt.strftime("%B").upper(),
        "year": dt.strftime("%Y"),
        "temp": temp_label,
    }


def draw_seven_segment_char(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    char: str,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    thickness = max(4, int(min(w, h) * 0.14))
    gap = max(2, thickness // 4)
    top_h = max(1, (h - thickness * 3 - gap * 2) // 2)
    mid_y = y0 + top_h + thickness + gap
    bottom_y = y0 + top_h * 2 + thickness * 2 + gap * 2

    def rect(a: int, b: int, c: int, d: int) -> None:
        draw.rounded_rectangle([a, b, c, d], radius=max(2, thickness // 2), fill=fill)

    active = set(SEGMENT_MAP.get(char, ""))
    if char == ":":
        cx = x0 + w // 2
        dot = max(4, thickness // 2)
        cy1 = y0 + h // 3
        cy2 = y0 + (h * 2) // 3
        rect(cx - dot // 2, cy1 - dot // 2, cx + dot // 2, cy1 + dot // 2)
        rect(cx - dot // 2, cy2 - dot // 2, cx + dot // 2, cy2 + dot // 2)
        return

    if char not in SEGMENT_MAP:
        return

    # Horizontal segments
    if "a" in active:
        rect(x0 + thickness, y0, x1 - thickness, y0 + thickness)
    if "g" in active:
        rect(x0 + thickness, mid_y, x1 - thickness, mid_y + thickness)
    if "d" in active:
        rect(x0 + thickness, bottom_y, x1 - thickness, bottom_y + thickness)

    # Vertical segments
    left_top_y1 = y0 + thickness
    left_top_y2 = y0 + thickness + top_h
    right_top_y1 = y0 + thickness
    right_top_y2 = y0 + thickness + top_h
    left_bottom_y1 = mid_y + thickness
    left_bottom_y2 = mid_y + thickness + top_h
    right_bottom_y1 = mid_y + thickness
    right_bottom_y2 = mid_y + thickness + top_h

    if "f" in active:
        rect(x0, left_top_y1, x0 + thickness, left_top_y2)
    if "b" in active:
        rect(x1 - thickness, right_top_y1, x1, right_top_y2)
    if "e" in active:
        rect(x0, left_bottom_y1, x0 + thickness, left_bottom_y2)
    if "c" in active:
        rect(x1 - thickness, right_bottom_y1, x1, right_bottom_y2)


def draw_seven_segment_text(
    panel: Image.Image,
    text: str,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int, int],
) -> None:
    draw = ImageDraw.Draw(panel)
    x0, y0, x1, y1 = box
    chars = list(text)
    if not chars:
        return

    count = len(chars)
    gap = max(6, (x1 - x0) // max(14, count * 5))
    total_gap = gap * (count - 1)
    char_w = max(12, ((x1 - x0) - total_gap) // count)
    char_h = max(12, y1 - y0)

    # Light glow
    glow_layer = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)
    for idx, char in enumerate(chars):
        cx0 = x0 + idx * (char_w + gap)
        cx1 = cx0 + char_w
        draw_seven_segment_char(panel, glow_draw, char, (cx0, y0, cx1, y0 + char_h), (24, 160, 255, 110))
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=2))
    panel.alpha_composite(glow_layer)

    # Crisp foreground
    draw = ImageDraw.Draw(panel)
    for idx, char in enumerate(chars):
        cx0 = x0 + idx * (char_w + gap)
        cx1 = cx0 + char_w
        draw_seven_segment_char(panel, draw, char, (cx0, y0, cx1, y0 + char_h), fill)


def draw_weather_icon(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    daypart: str,
) -> None:
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    cx = x0 + w // 2
    cy = y0 + h // 2
    sun_color = (255, 200, 55, 255)
    cloud_color = (35, 115, 220, 255)
    white = (245, 245, 245, 255)

    if daypart in {"MORNING", "AFTERNOON"}:
        r = max(10, min(w, h) // 4)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=sun_color)
        for angle in range(0, 360, 30):
            rad = np.deg2rad(angle)
            inner = r + 4
            outer = r + 18
            x_start = cx + int(np.cos(rad) * inner)
            y_start = cy + int(np.sin(rad) * inner)
            x_end = cx + int(np.cos(rad) * outer)
            y_end = cy + int(np.sin(rad) * outer)
            draw.line([x_start, y_start, x_end, y_end], fill=sun_color, width=4)
        cloud = [
            (cx - r - 8, cy + 4, cx + r + 8, cy + 28),
            (cx - r + 5, cy - 8, cx + r - 2, cy + 16),
        ]
        draw.rounded_rectangle(cloud[0], radius=12, fill=cloud_color)
        draw.rounded_rectangle(cloud[1], radius=10, fill=cloud_color)
    else:
        r = max(10, min(w, h) // 4)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=white)
        draw.ellipse([cx - r + 8, cy - r + 2, cx + r + 8, cy + r + 2], fill=(0, 0, 0, 0))
        draw.rounded_rectangle([cx - r, cy + 2, cx + r + 12, cy + 24], radius=12, fill=cloud_color)


def overlay_timestamp(
    frame: np.ndarray,
    date_text: str,
    time_text: str,
    temp_text: str | None,
    font_size: int,
    margin: int,
    box_alpha: int,
) -> np.ndarray:
    meta = parse_timestamp_fields(date_text, time_text, temp_text)
    if meta is None:
        return frame

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    panel_w = min(int(image.size[0] * DEFAULT_TIMESTAMP_PANEL_WIDTH_FRAC), image.size[0] - margin * 2)
    panel_h = min(int(image.size[1] * DEFAULT_TIMESTAMP_PANEL_HEIGHT_FRAC), image.size[1] - margin * 2)
    panel_w = max(360, panel_w)
    panel_h = max(180, panel_h)
    x0 = margin
    y0 = image.size[1] - margin - panel_h
    x1 = x0 + panel_w
    y1 = y0 + panel_h

    panel = Image.new("RGBA", (panel_w, panel_h), (0, 0, 0, 0))
    panel_draw = ImageDraw.Draw(panel)

    # Backdrop
    panel_draw.rounded_rectangle(
        [0, 0, panel_w - 1, panel_h - 1],
        radius=max(16, font_size // 2),
        fill=(8, 14, 24, max(0, min(255, box_alpha))),
        outline=(35, 110, 200, 180),
        width=3,
    )

    inner = [8, 8, panel_w - 8, panel_h - 8]
    panel_draw.rounded_rectangle(
        inner,
        radius=max(12, font_size // 3),
        outline=(20, 80, 145, 120),
        width=1,
    )

    # Labels
    weekday_font = find_ui_font(max(18, font_size + 4))
    period_font = find_ui_font(max(14, font_size - 2))
    bottom_font = find_ui_font(max(16, font_size - 4))
    small_font = find_ui_font(max(14, font_size - 6))

    panel_draw.text((panel_w // 2, 12), meta["weekday"], font=weekday_font, fill=(245, 245, 245, 255), anchor="ma")

    # Main time and icon layout
    time_box = (18, 58, int(panel_w * 0.68), panel_h - 58)
    draw_seven_segment_text(panel, meta["time_24h"], time_box, (245, 245, 245, 255))

    icon_box = (int(panel_w * 0.72), 52, panel_w - 12, 100)
    draw_weather_icon(panel_draw, icon_box, meta["period"])
    if meta["temp"]:
        panel_draw.text((panel_w - 14, 104), meta["temp"], font=bottom_font, fill=(245, 245, 245, 255), anchor="ra")

    # Bottom strip
    strip_y = panel_h - 52
    panel_draw.line([18, strip_y, panel_w - 18, strip_y], fill=(45, 140, 255, 180), width=2)
    panel_draw.line([panel_w * 0.28, strip_y + 10, panel_w * 0.28, panel_h - 16], fill=(45, 140, 255, 150), width=2)
    panel_draw.line([panel_w * 0.72, strip_y + 10, panel_w * 0.72, panel_h - 16], fill=(45, 140, 255, 150), width=2)

    day_font = find_ui_font(max(20, font_size + 2))
    month_font = find_ui_font(max(20, font_size + 4))
    year_font = find_ui_font(max(20, font_size + 4))
    label_font = find_ui_font(max(12, font_size - 8))
    period_font_bottom = find_ui_font(max(13, font_size - 7))

    left_center = int(panel_w * 0.14)
    mid_center = int(panel_w * 0.50)
    right_center = int(panel_w * 0.86)
    day_x = int(panel_w * 0.11)
    day_y = panel_h - 34
    panel_draw.text((day_x, day_y), meta["day"], font=day_font, fill=(245, 245, 245, 255), anchor="mm")
    panel_draw.text((day_x + 46, day_y), meta["period"], font=period_font_bottom, fill=(60, 150, 255, 255), anchor="lm")
    panel_draw.text((left_center, panel_h - 14), "DAY", font=label_font, fill=(60, 150, 255, 255), anchor="mm")
    panel_draw.text((mid_center, panel_h - 34), meta["month"], font=month_font, fill=(245, 245, 245, 255), anchor="mm")
    panel_draw.text((mid_center, panel_h - 14), "MONTH", font=label_font, fill=(60, 150, 255, 255), anchor="mm")
    panel_draw.text((right_center, panel_h - 34), meta["year"], font=year_font, fill=(245, 245, 245, 255), anchor="mm")
    panel_draw.text((right_center, panel_h - 14), "YEAR", font=label_font, fill=(60, 150, 255, 255), anchor="mm")

    # Composite panel onto the frame.
    panel = panel.filter(ImageFilter.GaussianBlur(radius=0.4))
    overlay.alpha_composite(panel, dest=(x0, y0))

    combined = Image.alpha_composite(image, overlay).convert("RGB")
    return cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)


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

    timestamp_reader = None
    if args.overlay_timestamp:
        timestamp_reader = load_timestamp_reader()
        if timestamp_reader is None:
            raise SystemExit("easyocr is not installed. Install it to overlay timestamps.")

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
    last_date_text = None
    last_time_text = None
    last_temp_text = None
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

            if args.overlay_timestamp and timestamp_reader is not None:
                date_text, time_text, temp_text = extract_timestamp_parts(
                    timestamp_reader,
                    frame,
                    args.timestamp_ocr_top_frac,
                    args.timestamp_ocr_left_frac,
                )
                if date_text is not None:
                    last_date_text = date_text
                if time_text is not None:
                    last_time_text = time_text
                if temp_text is not None:
                    last_temp_text = temp_text
                if last_date_text is not None and last_time_text is not None:
                    cleaned = overlay_timestamp(
                        cleaned,
                        last_date_text,
                        last_time_text,
                        last_temp_text,
                        args.timestamp_font_size,
                        args.timestamp_margin,
                        args.timestamp_box_alpha,
                    )

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
