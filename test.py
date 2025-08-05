import streamlit as st
from gtts import gTTS
import os
from playsound import playsound

st.title("Text to Speech with gTTS and Streamlit")

text = st.text_input("Enter text to speak", "Hello, how are you?")

if st.button("🔊 Speak it"):
    if text.strip() == "":
        st.warning("Please enter some text first.")
    else:
        try:
            # Generate TTS and save to temp file
            tts = gTTS(text=text, lang="en")
            file = "C:/users/weshore/echosense/sign.mp3"
            tts.save(file)
            playsound(file)
            os.remove(file)
           

        except Exception as e:
            st.error(f"Error generating speech: {e}")
