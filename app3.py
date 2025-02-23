import streamlit as st
import cv2
import torch
import speech_recognition as sr
from moviepy.editor import VideoFileClip
from transformers import pipeline
from pathlib import Path
from openai import OpenAI
import os
import time
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()
# Get the OpenAI API key from the environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Function to extract audio from the video
def extract_audio_from_video(video_path):
    video = VideoFileClip(video_path)
    audio_path = "extracted_audio.wav"
    video.audio.write_audiofile(audio_path)
    return audio_path

# Function to transcribe audio using Google Speech Recognition
def transcribe_audio(audio_path, progress_callback):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        progress_callback(50)  # Update progress halfway
        try:
            text = recognizer.recognize_google(audio_data)
            progress_callback(100)  # Update progress to complete
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"API request failed: {e}"

# Function to correct transcription using Hugging Face model
def correct_transcription(transcription, progress_callback):
    corrector = pipeline("text2text-generation", model="facebook/bart-large-cnn")
    progress_callback(50)  # Midway through correction
    corrected_text = corrector(transcription)[0]['generated_text']
    progress_callback(100)  # Correction complete
    return corrected_text

# Function to generate audio from text using OpenAI API
def generate_audio_with_openai(text, voice="alloy"):
    speech_file_path = Path("speech.mp3")
    response = client.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=text
    )
    response.stream_to_file(speech_file_path)
    return speech_file_path

# Streamlit UI setup
def main():
    st.set_page_config(page_title="Video Text Extractor & TTS", layout="centered", initial_sidebar_state="collapsed")

    # Toggle between light and dark mode using a selectbox
    mode = st.selectbox("Toggle Light/Dark Mode", ["Light Mode", "Dark Mode"])
    
    # Define colors based on mode
    if mode == "Dark Mode":
        background_color = "#2E2E2E"
        text_color = "white"
        button_color = "#3e92cc"
    else:
        background_color = "#f9f9f9"
        text_color = "black"
        button_color = "#3e92cc"

    # Set CSS based on selected mode
    st.markdown(
        f"""
        <style>
        .main {{
            background-color: {background_color};
            color: {text_color};
        }}
        .css-18e3th9 {{
            background-color: {button_color};
            color: {text_color};
            border-radius: 5px;
        }}
        .step {{
            background-color: #d1e7dd;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 15px;
        }}
        .stButton>button {{
            color: {text_color};
            background-color: {button_color};
            border-radius: 8px;
            font-size: 16px;
            padding: 10px 20px;
            font-weight: bold;
        }}
        .stTextArea label {{
            font-weight: bold;
            font-size: 18px;
            color: {text_color};
        }}
        .stRadio label {{
            font-size: 16px;
            color: {text_color};
        }}
        .block-container p {{
            color: {text_color};
            font-size: 18px;
        }}
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown('<h1 style="color: #8A2BE2; font-size: 2.5em; font-weight: bold; text-align: center;">🎬 AI-Powered Video Text Extractor & TTS</h1>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

    if uploaded_file:
        with open("input_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        st.video("input_video.mp4")

        # Step 1: Extracting audio
        st.markdown('<div class="step"><h3>Step 1: Extracting Audio from Video...</h3></div>', unsafe_allow_html=True)
        progress_bar = st.progress(0)
        audio_path = extract_audio_from_video("input_video.mp4")
        progress_bar.progress(100)
        
        # Step 2: Transcribing audio
        st.markdown('<div class="step"><h3>Step 2: Transcribing Audio...</h3></div>', unsafe_allow_html=True)
        progress_bar.progress(0)
        transcription = transcribe_audio(audio_path, progress_bar.progress)
        
        if transcription:
            st.markdown('<div class="step"><h3>Original Transcription</h3></div>', unsafe_allow_html=True)
            original_text = st.text_area("Edit Original Text", transcription, height=200)

            st.markdown('<div class="step"><h3>Step 3: Correcting Transcription</h3></div>', unsafe_allow_html=True)
            corrected_text = st.text_area("Corrected Text", "", height=200)
            
            if st.button("Regenerate Summarization"):
                progress_bar.progress(0)
                corrected_text = correct_transcription(original_text, progress_bar.progress)
                st.text_area("Corrected Transcription", corrected_text, height=200)

            st.markdown('<div class="step"><h3>Step 4: Text-to-Speech Conversion with OpenAI</h3></div>', unsafe_allow_html=True)
            if st.button("Generate Audio with OpenAI"):
                st.write("### Generating audio...")
                progress_bar.progress(0)
                tts_audio_path = generate_audio_with_openai(original_text)
                progress_bar.progress(100)
                st.audio(tts_audio_path, format="audio/mp3")
                st.success("Audio generated successfully!")

if __name__ == "__main__":
    main()
