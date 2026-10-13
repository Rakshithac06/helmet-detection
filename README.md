# Helmet Detection with YOLOv8 + FastAPI

Detects `helmet` vs. `no_helmet` in images. A YOLOv8n model is served with FastAPI, with a custom HTML/JavaScript frontend that draws boxes on a canvas.

## Stack
Python, YOLOv8 (Ultralytics), FastAPI, HTML/JavaScript

## Data
Two Roboflow Universe datasets (construction hard hats, and motorcycle/cycling helmets), merged and remapped to two classes: `no_helmet` (0) and `helmet` (1). Near-duplicates were removed with perceptual hashing (including mirrored copies), then the data was split 80/10/10 (1,200 / 150 / 150 images). Check each source dataset's license on Roboflow before reusing.

## Training
YOLOv8n, 60 epochs, 640px, seed 42, on a Colab T4 GPU (about 20 minutes).

## Results (held-out test set, 150 images)
| Class | Precision | Recall | mAP@50 |
|---|---|---|---|
| all | 0.860 | 0.834 | 0.902 |
| no_helmet | 0.819 | 0.824 | 0.886 |
| helmet | 0.901 | 0.844 | 0.917 |

## History: the first model failed on new photos
My first model, trained on a single dataset, scored about 90% mAP@50 on its own test set. It still labeled a white hard hat as `no_helmet` (0.89) and a pink motorcycle helmet as `no_helmet` (0.91). Perceptual hashing also showed that 34% of that test set overlapped with training data. I added a second dataset, rebuilt a deduplicated split and retrained. Both photos are now detected as `helmet`.

## Limitations
- Manual checks used only a handful of photos, not a systematic evaluation.
- Caps may be confused with helmets (one cap photo was labeled `helmet`; needs more testing).
- Isolated product shots (a hat on a mannequin) were missed.
- Bounding boxes are loose, especially for `no_helmet` (mAP@50-95 is about 0.55).
- Small or distant heads are often missed.
- Source datasets are small, so behavior outside them is not guaranteed.

## Run locally
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    uvicorn main:app --reload

Open http://127.0.0.1:8000 (API docs at /docs).

## Future work
Add cap and hat images labeled `no_helmet`, more varied bare-head data, higher input resolution, and a larger hand-collected evaluation set.
