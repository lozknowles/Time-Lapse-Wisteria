# AGENTS.md

## Project Rules

- Keep the repo focused on the timelapse processing workflow.
- Prefer small, readable Python changes over large refactors.
- Do not commit generated videos, sample renders, or cache files.
- Use the CLI script as the source of truth for render behavior.

## Current Workflow

- Select a source video when prompted, or pass `--input`.
- Provide a rotation angle at the prompt, or pass `--rotation-deg`.
- The script reports progress while it processes frames.
- The cleaned output is written beside the input file unless `--output` is provided.

## Notes For Future Changes

- Keep the timestamp crop behavior configurable.
- Keep the skip-start behavior configurable.
- If the render pipeline changes, update the README and changelog together.
