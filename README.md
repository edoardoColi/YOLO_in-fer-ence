# YOLO In-Fer-Ence

Flask control panel for running YOLO inference from a camera device or stream inside a Docker container.

## Quick Start

1. Put one or more YOLO model files in `Inference/models/`.
2. Start the service:

```bash
make build
make up
```

3. Open the control panel at `http://localhost:5000`.
4. Upload one or more model files from the panel, or put them manually in `Inference/models/`.
5. Select a model, set the input source, change the status to `inference`, then press `Start`.

## Configuration

The service is configured through environment variables in `docker-compose.yml`.

| Variable | Default | Description |
| --- | --- | --- |
| `RUNNING_PORT` | `5000` | Flask port. With `network_mode: host`, this is exposed on the host directly. |
| `MODEL_DIR` | `/models` | Directory scanned for model files. |
| `STARTUP_STATUS` | `idle` | Initial state. Use `inference` only when the model and source are ready. |
| `STARTUP_MODEL` | first discovered model | Optional model filename relative to `MODEL_DIR`. |
| `INPUT_SRC` | `/dev/video0` | Camera device, numeric device id, local file, or stream URL. |
| `SOURCE_WIDTH` | `1080` | Requested capture width. |
| `SOURCE_HEIGHT` | `720` | Requested capture height. |
| `SOURCE_FPS` | `30` | Requested capture FPS. |
| `LOOP_SLEEP_SECONDS` | `0` | Optional delay after each inference iteration. |
| `MAX_UPLOAD_MB` | `1024` | Maximum total model upload size per request. |

## HTTP Endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` | `GET` | Web control panel. |
| `/health` | `GET` | Health and current state. Returns `503` when the last run failed. |
| `/api/state` | `GET` | JSON state snapshot. |
| `/api/status` | `POST` | Change status with `status`. |
| `/api/model` | `POST` | Change model with `model`. |
| `/api/input` | `POST` | Change input with `input_source` or `input_video`. |
| `/api/models/upload` | `POST` | Upload one or more model files as multipart field `models`. |
| `/start_loop` | `POST` | Start the background inference loop. |
| `/stop_loop` | `POST` | Stop the background inference loop and set status to `idle`. |

The older form endpoints (`/change_status`, `/change_model`, `/change_input_video`) are still available for the HTML panel. The upload form endpoint is `/upload_models`.

## Development

```bash
make check
make logs
make down
```

`make check` compiles the Flask app and catches syntax errors without requiring a camera or model.

## Notes

Model files are mounted from `Inference/models` into `/models` at runtime, so replacing or uploading a model does not require rebuilding the image. The local `.gitignore` keeps model binaries out of git while preserving `Inference/models/.gitkeep`.
