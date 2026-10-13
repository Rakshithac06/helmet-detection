import io
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="Hard Hat Detection API")
model = YOLO("best.pt")


@app.get("/health")
def health():
    return {"status": "ok", "classes": model.names}


@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float = Form(0.35)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the image")

    r = model.predict(img, conf=conf, imgsz=640, iou=0.5, verbose=False)[0]
    detections = []
    for box, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        detections.append({
            "label": model.names[int(c)],
            "confidence": round(s, 3),
            "box": [round(v, 1) for v in box],  # x1, y1, x2, y2 in original pixels
        })
    return {"width": img.width, "height": img.height, "detections": detections}


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")