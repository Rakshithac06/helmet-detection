import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Hard Hat Detection", layout="wide")
st.title("Hard Hat Detection with YOLOv8")
st.caption("Detects construction hard hats vs. no hard hat (caps, hats, bare heads). "
           "Trained on a small Roboflow Universe dataset of construction-site images. "
           "Known limitation: does not generalize to motorcycle or cycling helmets.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()
conf = st.slider("Confidence threshold", 0.1, 0.9, 0.35, 0.05)
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    r = model.predict(img, conf=conf, imgsz=640)[0]
    c1, c2 = st.columns(2)
    c1.image(img, caption="Input", use_container_width=True)
    c2.image(r.plot()[:, :, ::-1], caption="Detections", use_container_width=True)
    counts = {}
    for k in r.boxes.cls.tolist():
        n = model.names[int(k)]
        counts[n] = counts.get(n, 0) + 1
    st.json(counts or {"no detections": 0})
