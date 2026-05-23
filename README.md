# Time-Lapse Wisteria

This project cleans a timelapse AVI by:

- letting you choose the source file
- prompting for a rotation angle if one is not supplied
- skipping the first second of footage by default
- rotating and cropping each frame
- removing the burned-in timestamp strip at the bottom
- trimming an extra 50 pixels from the bottom of the final frame
- detecting people, cars, and animals with YOLOv8
- either discarding frames with detections or inpainting the detected boxes
- OCRing the burned-in date, time, and temperature from the source frame while excluding the right-side logo
- rendering the extracted data in a digital-panel style overlay inspired by a seven-segment clock display
- showing live progress while the render runs

## Run

```bash
python process_timelapse.py
```

If you want to provide everything up front:

```bash
python process_timelapse.py --input DSCF0001.AVI --output DSCF0001_cleaned.avi --rotation-deg 4 --start-seconds 1
```

## Options

- `--input`: source video file
- `--output`: output video path
- `--rotation-deg`: rotation angle in degrees
- `--crop`: crop rectangle as percentages of width and height
- `--timestamp-crop-px`: bottom crop in pixels to remove the timestamp bar
- `--bottom-crop-px`: extra bottom crop in pixels after timestamp removal
- `--day-threshold`: brightness filter for daytime-only output
- `--start-seconds`: number of seconds to skip before processing
- `--detect-people`, `--detect-cars`, `--detect-animals`: enable object classes to filter
- `--object-action`: `discard` or `inpaint`
- `--object-confidence`: YOLO confidence threshold
- `--mask-padding`: padding around detected boxes before masking or discarding
- `--inpaint-radius`: radius used for inpainting detected regions
- `--yolo-model`: YOLO weights file, default `yolov8n.pt`
- `--overlay-timestamp` / `--no-overlay-timestamp`: enable or disable OCR timestamp overlay
- `--timestamp-ocr-top-frac`: top fraction of the source frame used for OCR
- `--timestamp-ocr-left-frac`: left fraction of the source frame used for OCR so the logo area is excluded
- `--timestamp-font-size`: font size used for the timestamp overlay
- `--timestamp-margin`: margin for the timestamp overlay from the bottom-left corner
- `--timestamp-box-alpha`: background opacity for the timestamp overlay box

## Notes

- Positive rotation values rotate counterclockwise in OpenCV.
- If `--input` is omitted, a file picker will open when available.
- If `--rotation-deg` is omitted, you will be prompted in the terminal.
- Daytime filtering is enabled by default with a threshold of `70`.
- The script stores YOLO cache data inside `.cache/torch` in the repo folder.
