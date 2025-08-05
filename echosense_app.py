import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import pyttsx3
import cv2
from transformers import AutoProcessor, AutoModelForVision2Seq

st.set_page_config(page_title="EchoSense", layout="centered")
st.title("🧏 EchoSense: Offline Sign Language Interpreter")
st.markdown("Translate sign language gestures into spoken or written language — all on-device.")

@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("google/gemma-3n-vision")
    model = AutoModelForVision2Seq.from_pretrained("google/gemma-3n-vision")
    return processor, model

processor, model = load_model()

def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

upload_option = st.radio("Choose input mode:", ["📁 Upload Image", "📷 Use Webcam"])

image = None
if upload_option == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload sign language image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Sign", use_column_width=True)

elif upload_option == "📷 Use Webcam":
    capture = st.button("Capture Gesture")
    if capture:
        cam = cv2.VideoCapture(0)
        ret, frame = cam.read()
        cam.release()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            st.image(image, caption="Captured Sign", use_column_width=True)

if st.button("🔍 Translate Gesture"):
    if image:
        with st.spinner("Translating gesture..."):
            inputs = processor(images=image, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=20)
            translation = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            st.success(f"🗣️ Translation: **{translation}**")
            if st.checkbox("🔊 Speak Output"):
                speak(translation)
    else:
        st.warning("Please provide or capture an image first.")
