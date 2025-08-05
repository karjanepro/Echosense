# echosense_app.py

import time
import streamlit as st
import torch
from PIL import Image

from gtts import gTTS
from playsound import playsound
import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoProcessor, Gemma3nForConditionalGeneration

# Load environment variables
load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if hf_token:
    login(hf_token)

# Streamlit app configuration
st.set_page_config(page_title="EchoSense", layout="centered")
st.title("🧏 EchoSense: Offline Sign Language Interpreter")
st.markdown("Translate sign language gestures into spoken or written language — fully offline with **Gemma 3n**.")

# Load model + processor
@st.cache_resource
def load_model():
    model_id = "google/gemma-3n-e4b-it"  # or "gemma-3n-e2b-it" for lighter
    model = Gemma3nForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return processor, model

processor, model = load_model()

def display_sign_gifs(text):
    st.markdown("### 👋 Animated Sign Language GIFs:")
    for char in text.upper():
        if char.isalpha():
            gif_path = os.path.join("asl_gifs", f"{char}.gif")
            if os.path.exists(gif_path):
                st.markdown(f"**{char}**")
                st.image(gif_path, use_container_width=False, width=200)
            else:
                st.warning(f"Missing GIF for letter: {char}")

from PIL import Image
import numpy as np
import cv2




# Image input section
input_mode = st.radio("Choose input mode:", ["📁 Upload Image", "🔤 Input Text"])

if input_mode == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload a sign image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Sign", width=200)
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if st.button("🔍 Translate Gesture"):
            with st.spinner("Processing with Gemma 3n..."):
                # Build the chat prompt with image + text
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant who understands sign language."}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": "What is this person signing?"}
                        ]
                    }
                ]

                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)

                input_len = inputs["input_ids"].shape[-1]
                print(inputs["input_ids"].shape)
                with torch.inference_mode():
                    generation = model.generate(**inputs, max_new_tokens=60, do_sample=False)
                    output = generation[0][input_len:]

                translation = processor.decode(output, skip_special_tokens=True)
                st.session_state["translation"] = translation  # Save it
                st.success(f"🗣️ Translation: **{translation}**")
                
                if "translation" in st.session_state:
                    if st.button("🔊 Speak it"):
                        translation = st.session_state["translation"]
                        if translation.strip() == "":
                            st.warning("Please enter some text first.")
                        else:
                            try:
                                # Generate TTS and save to temp file
                                tts = gTTS(text=translation, lang="en")
                                file = "C:/users/weshore/echosense/sign.mp3"
                                tts.save(file)
                                playsound(file)
                                time.sleep(2)  # Reduce wait time
                                st.success("🔊 Spoken successfully!")
                                os.remove(file)
                            except Exception as e:
                                st.error(f"Error generating speech: {e}")

elif input_mode == "🔤 Input Text":
    st.subheader("Type Message to Show in Sign Language")
    text_input = st.text_input("Enter a phrase or word:")
    if text_input:
        if os.path.exists("asl_gifs/alphabet_full.gif"):
            st.markdown("#### Full ASL Alphabet Animation:")
            st.image("asl_gifs/alphabet_full.gif", width=200)
        display_sign_gifs(text_input)
