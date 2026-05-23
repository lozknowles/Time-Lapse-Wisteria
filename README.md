# Time-Lapse Wisteria

This project cleans a timelapse AVI by:

- letting you choose the source file
- prompting for a rotation angle if one is not supplied
- skipping the first second of footage by default
- rotating and cropping each frame
- removing the burned-in timestamp strip at the bottom
- trimming an extra 50 pixels from the bottom of the final frame
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

## Notes

- Positive rotation values rotate counterclockwise in OpenCV.
- If `--input` is omitted, a file picker will open when available.
- If `--rotation-deg` is omitted, you will be prompted in the terminal.
- Daytime filtering is enabled by default with a threshold of `70`.
