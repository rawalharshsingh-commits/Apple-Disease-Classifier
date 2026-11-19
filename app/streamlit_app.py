"""A small local demo for the trained Apple Leaf paper models."""
import sys
from pathlib import Path

import albumentations as A
import numpy as np
import streamlit as st
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apple_disease import CLASSES
from apple_disease.models import MODEL_NAMES, build_model

st.set_page_config(page_title="Apple Leaf Disease Classifier", page_icon="🍎")


@st.cache_resource
def load_model(model_key: str, checkpoint_path: str, modified_at: float):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, _ = build_model(model_key, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


st.title("Apple Leaf Disease Classifier")
st.write("A beginner-friendly demo of the three-model approach from the research paper.")
st.info("Research demonstration only. A prediction is not a substitute for expert diagnosis.")

available = [key for key in MODEL_NAMES
             if (ROOT / "outputs" / "checkpoints" / f"{key}.pt").exists()]
if not available:
    st.warning("No trained checkpoints found. Train at least one model first.")
    st.stop()

choice = st.selectbox("Model", ["ensemble", *available])
image_file = st.file_uploader("Upload an apple leaf image", type=["jpg", "jpeg", "png"])
if image_file:
    image = Image.open(image_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    if st.button("Classify leaf"):
        image_array = np.asarray(image)
        model_keys = available if choice == "ensemble" else [choice]
        outputs = []
        for key in model_keys:
            checkpoint_path = ROOT / "outputs" / "checkpoints" / f"{key}.pt"
            model, checkpoint = load_model(key, str(checkpoint_path), checkpoint_path.stat().st_mtime)
            transform = A.Compose([
                A.Resize(checkpoint["image_size"], checkpoint["image_size"]),
                A.Normalize(mean=checkpoint["mean"], std=checkpoint["std"]),
                ToTensorV2(),
            ])
            tensor = transform(image=image_array)["image"].unsqueeze(0)
            with torch.no_grad():
                outputs.append(torch.softmax(model(tensor), dim=1)[0].numpy())
        probabilities = np.mean(outputs, axis=0)
        best = int(probabilities.argmax())
        st.subheader(f"Prediction: {CLASSES[best].replace('_', ' ')}")
        st.bar_chart({name.replace("_", " "): float(probability)
                      for name, probability in zip(CLASSES, probabilities)})
