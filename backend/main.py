import os
import cv2
import math
import tempfile
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Pothole Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your Vercel URL in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model loading (once at startup) ───────────────────────────────────────────
model: YOLO | None = None

CLASS_NAMES = ["Pothole"]
CONF_THRESHOLD = 0.45
FRAME_SIZE = (640, 384)


@app.on_event("startup")
async def load_model():
    global model

    # ── Torch 2.6+ safety patch ────────────────────────────────────────────
    # PyTorch 2.6 changed torch.load() to default weights_only=True, which
    # blocks YOLO's custom classes. We allowlist them explicitly so the load
    # succeeds without falling back to unsafe pickle execution.
    import torch
    import ultralytics.nn.tasks as _tasks
    import ultralytics.nn.modules as _modules

    _safe = [
        _tasks.DetectionModel,
        _tasks.SegmentationModel,
        _tasks.ClassificationModel,
        _tasks.PoseModel,
    ]
    # add_safe_globals is available from torch 2.6+; guard for older builds
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals(_safe)

    # ── Download from HuggingFace Hub ─────────────────────────────────────
    # Cache lives in ~/.cache/huggingface for the container's lifetime.
    print("Downloading / loading model from HuggingFace Hub…")
    downloaded = hf_hub_download(
        repo_id=os.environ["HF_REPO_ID"],
        filename="Nano_model.pt",
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"Model path: {downloaded}")
    model = YOLO(downloaded)
    print("Model loaded ✓")


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


# ── Single image detection ─────────────────────────────────────────────────────
@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    data = await file.read()
    arr  = np.frombuffer(data, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")

    img = cv2.resize(img, FRAME_SIZE)
    detections = _run_inference(img)

    return JSONResponse({"width": FRAME_SIZE[0], "height": FRAME_SIZE[1],
                         "detections": detections})


# ── Video detection ────────────────────────────────────────────────────────────
@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    if not file.content_type.startswith("video/"):
        raise HTTPException(400, "File must be a video")

    # Save upload to a temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
        tmp_in.write(await file.read())
        in_path = tmp_in.name

    out_path = in_path.replace(".mp4", "_annotated.mp4")

    try:
        _process_video(in_path, out_path)
    except Exception as e:
        raise HTTPException(500, f"Video processing failed: {e}")
    finally:
        os.unlink(in_path)

    return FileResponse(
        out_path,
        media_type="video/mp4",
        filename="pothole_annotated.mp4",
        background=_cleanup(out_path),   # delete after sending
    )


# ── Internal helpers ───────────────────────────────────────────────────────────
def _run_inference(img: np.ndarray) -> list[dict]:
    results = model(img, stream=False, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            conf = math.ceil(box.conf[0].item() * 100) / 100
            if conf < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0].item())
            detections.append({
                "x": int(x1), "y": int(y1),
                "w": int(x2 - x1), "h": int(y2 - y1),
                "conf": conf,
                "label": CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls),
            })
    return detections


def _process_video(in_path: str, out_path: str):
    cap = cv2.VideoCapture(in_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, FRAME_SIZE)

    frame_idx = 0
    last_detections: list[dict] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, FRAME_SIZE)

        # Run inference every 3rd frame to keep it fast on free tier
        if frame_idx % 3 == 0:
            last_detections = _run_inference(frame)

        _draw_boxes(frame, last_detections)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


def _draw_boxes(img: np.ndarray, detections: list[dict]):
    for d in detections:
        x, y, w, h = d["x"], d["y"], d["w"], d["h"]
        label = f"{d['label']} {d['conf']:.2f}"
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, label, (max(0, x), max(35, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)


class _cleanup:
    """Background task to delete temp file after response is sent."""
    def __init__(self, path): self.path = path
    async def __call__(self): 
        try: os.unlink(self.path)
        except: pass