import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import cv2
from flask import Flask, jsonify, redirect, render_template, request, url_for
from ultralytics import YOLO
from werkzeug.utils import secure_filename


VALID_STATUSES = {
    "idle": 0,
    "offload_raw": 1,
    "load_raw": 2,
    "offload_jpg": 3,
    "load_jpg": 4,
    "offload_backbone": 5,
    "load_backbone": 6,
    "inference": 7,
}

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
DEFAULT_INPUT_SRC = os.getenv("INPUT_SRC", "/dev/video0")
RUNNING_PORT = int(os.getenv("RUNNING_PORT", "5000"))
STATS_PRINT = os.getenv("STATS_PRINT", "true").lower() in {"true", "1", "t", "yes", "y"}
SOURCE_WIDTH = int(os.getenv("SOURCE_WIDTH", "1080"))
SOURCE_HEIGHT = int(os.getenv("SOURCE_HEIGHT", "720"))
SOURCE_FPS = int(os.getenv("SOURCE_FPS", "30"))
LOOP_SLEEP_SECONDS = float(os.getenv("LOOP_SLEEP_SECONDS", "0"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "1024"))
ALLOWED_MODEL_EXTENSIONS = {".pt", ".onnx", ".engine", ".torchscript", ".tflite", ".pb", ".yaml", ".yml"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
if STATS_PRINT:
    app.logger.setLevel(logging.INFO)

state_lock = threading.RLock()
control_background_thread: threading.Thread | None = None
stop_event = threading.Event()
model_cache: dict[str, YOLO] = {}


def discover_models(model_dir: Path = MODEL_DIR) -> dict[str, str]:
    if not model_dir.exists():
        return {}

    models: dict[str, str] = {}
    for path in sorted(model_dir.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            models[str(path.relative_to(model_dir))] = str(path.resolve())
    return models


MODEL_LIST = discover_models()


def refresh_models() -> None:
    global MODEL_LIST

    with state_lock:
        MODEL_LIST = discover_models()
        if app.config.get("MODEL") not in MODEL_LIST:
            app.config["MODEL"] = next(iter(MODEL_LIST), None)
        model_cache_keys = set(model_cache)
        for stale_model in model_cache_keys - set(MODEL_LIST):
            model_cache.pop(stale_model, None)


def is_allowed_model(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_MODEL_EXTENSIONS


def unique_model_path(filename: str) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    candidate = MODEL_DIR / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1
    while True:
        next_candidate = MODEL_DIR / f"{stem}_{counter}{suffix}"
        if not next_candidate.exists():
            return next_candidate
        counter += 1


def save_uploaded_models(files: list[Any]) -> tuple[list[str], list[str]]:
    saved: list[str] = []
    rejected: list[str] = []

    for uploaded_file in files:
        original_name = uploaded_file.filename or ""
        safe_name = secure_filename(original_name)
        if not safe_name or not is_allowed_model(safe_name):
            rejected.append(original_name or "unnamed file")
            continue

        target_path = unique_model_path(safe_name)
        uploaded_file.save(target_path)
        saved.append(str(target_path.relative_to(MODEL_DIR)))

    if saved:
        refresh_models()
        with state_lock:
            app.config["MODEL"] = saved[0]
            app.config["LATEST_RESULT"] = None

    return saved, rejected


def startup_status() -> int:
    status_name = os.getenv("STARTUP_STATUS", "idle").lower()
    return VALID_STATUSES.get(status_name, VALID_STATUSES["idle"])


def startup_model() -> str | None:
    requested = os.getenv("STARTUP_MODEL", "")
    if requested in MODEL_LIST:
        return requested
    return next(iter(MODEL_LIST), None)


app.config.update(
    STATUS=startup_status(),
    MODEL=startup_model(),
    INPUT_SRC=DEFAULT_INPUT_SRC,
    LAST_ERROR=None,
    LATEST_RESULT=None,
    LOOP_STARTED_AT=None,
)


def status_name(status_code: int | None = None) -> str:
    status_code = app.config["STATUS"] if status_code is None else status_code
    return next((name for name, code in VALID_STATUSES.items() if code == status_code), "unknown")


def normalize_video_source(source: str) -> str | int:
    stripped = source.strip()
    return int(stripped) if stripped.isdigit() else stripped


def current_state() -> dict[str, Any]:
    with state_lock:
        thread_running = control_background_thread is not None and control_background_thread.is_alive()
        return {
            "status": status_name(),
            "status_code": app.config["STATUS"],
            "model": app.config["MODEL"],
            "models": sorted(MODEL_LIST.keys()),
            "allowed_model_extensions": sorted(ALLOWED_MODEL_EXTENSIONS),
            "max_upload_mb": MAX_UPLOAD_MB,
            "input_source": app.config["INPUT_SRC"],
            "loop_running": thread_running,
            "loop_started_at": app.config["LOOP_STARTED_AT"],
            "last_error": app.config["LAST_ERROR"],
            "latest_result": app.config["LATEST_RESULT"],
        }


def set_error(message: str | None) -> None:
    with state_lock:
        app.config["LAST_ERROR"] = message


def mark_loop_stopped() -> None:
    with state_lock:
        app.config["LOOP_STARTED_AT"] = None


def set_status(status: str) -> bool:
    if status not in VALID_STATUSES:
        return False

    with state_lock:
        app.config["STATUS"] = VALID_STATUSES[status]
        if status == "idle":
            stop_event.set()
    return True


def set_model(model_name: str) -> bool:
    if model_name not in MODEL_LIST:
        return False

    with state_lock:
        app.config["MODEL"] = model_name
        app.config["LATEST_RESULT"] = None
    return True


def set_input_source(input_source: str) -> bool:
    clean_source = input_source.strip()
    if not clean_source:
        return False

    with state_lock:
        app.config["INPUT_SRC"] = clean_source
        app.config["LATEST_RESULT"] = None
    return True


def init_source_settings(cap: cv2.VideoCapture) -> None:
    app.logger.info("Initializing video source")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SOURCE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SOURCE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, SOURCE_FPS)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    app.logger.info("Requested source: %sx%s @ %s FPS", SOURCE_WIDTH, SOURCE_HEIGHT, SOURCE_FPS)
    app.logger.info("Opened source: %sx%s @ %s FPS", width, height, fps)


def get_model(model_name: str) -> YOLO:
    if model_name not in model_cache:
        model_cache[model_name] = YOLO(MODEL_LIST[model_name])
    return model_cache[model_name]


def summarize_result(result: Any, elapsed_seconds: float) -> dict[str, Any]:
    boxes = result.boxes.data.tolist() if result.boxes is not None else []
    names = result.names or {}

    counts: dict[str, int] = {}
    for box in boxes:
        class_id = int(box[5])
        class_name = names.get(class_id, str(class_id))
        counts[class_name] = counts.get(class_name, 0) + 1

    inference_ms = float(result.speed.get("inference", 0.0))
    return {
        "inference_ms": inference_ms,
        "yolo_fps": round(1000 / inference_ms, 2) if inference_ms > 0 else None,
        "loop_seconds": round(elapsed_seconds, 4),
        "detections": len(boxes),
        "counts": counts,
    }


def control_loop() -> None:
    app.logger.info("Control loop started")
    set_error(None)

    with state_lock:
        app.config["LOOP_STARTED_AT"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        model_name = app.config["MODEL"]
        input_source = app.config["INPUT_SRC"]

    if status_name() != "inference":
        set_error("Only the 'inference' status is implemented for the local control loop.")
        app.logger.warning(app.config["LAST_ERROR"])
        mark_loop_stopped()
        return

    if not model_name:
        set_error("No YOLO model found. Mount or copy a model into /models.")
        app.logger.error(app.config["LAST_ERROR"])
        mark_loop_stopped()
        return

    try:
        model = get_model(model_name)
    except Exception as exc:
        set_error(f"Could not load model '{model_name}': {exc}")
        app.logger.exception(app.config["LAST_ERROR"])
        mark_loop_stopped()
        return

    cap = cv2.VideoCapture(normalize_video_source(input_source))
    try:
        if not cap.isOpened():
            set_error(f"Could not open video source '{input_source}'.")
            app.logger.error(app.config["LAST_ERROR"])
            return

        init_source_settings(cap)
        while not stop_event.is_set():
            with state_lock:
                if app.config["STATUS"] == VALID_STATUSES["idle"]:
                    break

            started = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                set_error(f"Could not read a frame from '{input_source}'.")
                app.logger.error(app.config["LAST_ERROR"])
                break

            results = model.predict(source=frame, verbose=False)
            summary = summarize_result(results[0], time.perf_counter() - started)
            with state_lock:
                app.config["LATEST_RESULT"] = summary

            app.logger.info("Inference summary: %s", summary)
            if LOOP_SLEEP_SECONDS > 0:
                time.sleep(LOOP_SLEEP_SECONDS)
    finally:
        cap.release()
        mark_loop_stopped()
        app.logger.info("Control loop stopped")


def wants_json_response() -> bool:
    return request.is_json or request.accept_mimetypes.best == "application/json"


@app.get("/health")
def health() -> tuple[dict[str, Any], int]:
    state = current_state()
    status_code = 200 if not state["last_error"] else 503
    return jsonify({"ok": status_code == 200, **state}), status_code


@app.get("/api/state")
def api_state() -> Any:
    return jsonify(current_state())


@app.post("/api/status")
def api_change_status() -> Any:
    payload = request.get_json(silent=True) or request.form
    new_status = payload.get("status", "")
    if not set_status(new_status):
        return jsonify({"error": f"Invalid status '{new_status}'."}), 400
    return jsonify(current_state())


@app.post("/api/model")
def api_change_model() -> Any:
    payload = request.get_json(silent=True) or request.form
    new_model = payload.get("model", "")
    if not set_model(new_model):
        return jsonify({"error": f"Invalid model '{new_model}'."}), 400
    return jsonify(current_state())


@app.post("/api/input")
def api_change_input() -> Any:
    payload = request.get_json(silent=True) or request.form
    input_source = payload.get("input_video", payload.get("input_source", ""))
    if not set_input_source(input_source):
        return jsonify({"error": "Input source cannot be empty."}), 400
    return jsonify(current_state())


@app.post("/api/models/upload")
def api_upload_models() -> Any:
    files = request.files.getlist("models")
    if not files:
        return jsonify({"error": "No model files were uploaded."}), 400

    saved, rejected = save_uploaded_models(files)
    if not saved:
        return jsonify({"error": "No valid model files were uploaded.", "rejected": rejected}), 400

    status_code = 207 if rejected else 201
    return jsonify({"saved": saved, "rejected": rejected, **current_state()}), status_code


@app.post("/change_status")
def change_status() -> Any:
    response = api_change_status()
    return response if wants_json_response() else redirect(url_for("index"))


@app.post("/change_model")
def change_model() -> Any:
    response = api_change_model()
    return response if wants_json_response() else redirect(url_for("index"))


@app.post("/change_input_video")
def change_input_video() -> Any:
    response = api_change_input()
    return response if wants_json_response() else redirect(url_for("index"))


@app.post("/upload_models")
def upload_models() -> Any:
    response = api_upload_models()
    return response if wants_json_response() else redirect(url_for("index"))


@app.post("/start_loop")
def start_loop() -> Any:
    global control_background_thread

    stop_event.clear()
    with state_lock:
        already_running = control_background_thread is not None and control_background_thread.is_alive()

    if not already_running:
        control_background_thread = threading.Thread(target=control_loop, daemon=True)
        control_background_thread.start()
    else:
        app.logger.info("Control loop already running")

    return jsonify(current_state()) if wants_json_response() else redirect(url_for("index"))


@app.post("/stop_loop")
def stop_loop() -> Any:
    set_status("idle")
    return jsonify(current_state()) if wants_json_response() else redirect(url_for("index"))


@app.get("/")
def index() -> str:
    state = current_state()
    return render_template(
        "index.html",
        state=state,
        valid_statuses=sorted(VALID_STATUSES.items()),
        model_list=sorted(MODEL_LIST.items()),
    )


if __name__ == "__main__":
    app.logger.info("Configuration: %s", current_state())
    app.run(host="0.0.0.0", port=RUNNING_PORT)
