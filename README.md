# Video Spy (Research Project)

## Important Note

This repository is primarily a **research/experimental project**.

The goal is to test practical real-time detection workflows (person/animal), model fallback strategies, packaging behavior, and notification pipelines. It is not positioned as a production-hardened security system.

## What It Does

- One application with built-in GUI (Start/Stop + settings)
- Webcam detection of person/animals
- Photo mode: one photo per continuous appearance
- Video mode: records a clip when object appears
- Telegram sending for photo/video (optional)
- Runtime fallback chain for models (depending on config)

## Single App Workflow

- Main app file: `web_cam_person_detection.py`
- Single packaged app: `dist/web_cam_person_detection.exe`

When launched normally, the app opens the GUI.
When launched internally with `--run-detector`, it starts detector mode.

## Configuration

1. Copy `config.json.simple` to `config.json`.
2. Fill Telegram fields if needed.

Minimal example:

```json
{
  "telegram_token": "YOUR_TELEGRAM_BOT_TOKEN",
  "telegram_chat_id": "YOUR_TELEGRAM_CHAT_ID"
}
```

## Run from Source

```sh
python web_cam_person_detection.py
```

## Build EXE

```sh
pyinstaller --noconfirm --clean .\web_cam_person_detection.spec
```

Output:

- `dist/web_cam_person_detection.exe`

## Git Safety

- `config.json` is local-only and ignored (contains secrets)
- use `config.json.simple` in repository as a template
