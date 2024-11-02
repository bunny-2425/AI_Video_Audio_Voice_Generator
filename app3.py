import streamlit as st
import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from moviepy.editor import VideoFileClip, AudioFileClip
from gtts import gTTS
import os
import speech_recognition as sr
from transformers import pipeline

# Load pre-trained gender classification model
class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)  # Output: male or female

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_model():
    model = GenderClassifier()
    model.load_state_dict(torch.load('E:\AI Voice\Training'))  # Load your pre-trained model
    model.eval()
    return model

def predict_gender(frame, model):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    image_tensor = transform(frame).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    return "male" if predicted.item() == 0 else "female"

# Function to extract audio from video
def extract_audio_from_video(video_path):
    video = VideoFileClip(video_path)
    audio_path = "extracted_audio.wav"
    video.audio.write_audiofile(audio_path)
    return audio_path

# Function to transcribe audio
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"API request failed: {e}"

# Function to correct transcription using Hugging Face model
def correct_transcription(transcription):
    corrector = pipeline("text2text-generation", model="facebook/bart-large-cnn")
    corrected_text = corrector(transcription)[0]['generated_text']
    return corrected_text

# Function to generate audio from text using gtts
def generate_audio_from_text(text, output_file, gender):
    tts = gTTS(text, lang='en')
    tts.save(output_file)

# Function to replace audio in video
def replace_audio_in_video(video_path, new_audio_path, output_path):
    video = VideoFileClip(video_path)
    new_audio = AudioFileClip(new_audio_path)
    final_video = video.set_audio(new_audio)
    final_video.write_videofile(output_path, audio_codec='aac')

# Streamlit UI
def main():
    st.title("Video Audio Replacement with Gender-Based TTS")
    
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        with open("input_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        st.video("input_video.mp4")
        
        # Load gender classification model
        model = load_model()
        
        # Step 1: Extract audio from the uploaded video
        audio_file = extract_audio_from_video("input_video.mp4")
        st.write("Audio extracted successfully.")
        
        # Step 2: Transcribe the audio
        transcription = transcribe_audio(audio_file)
        st.write("Transcription:")
        st.write(transcription)
        
        # Step 3: Correct the transcription
        corrected_transcription = correct_transcription(transcription)
        st.write("Corrected Transcription:")
        st.write(corrected_transcription)
        
        # Step 4: Detect gender by analyzing video frames
        video_capture = cv2.VideoCapture("input_video.mp4")
        frame_count = 0
        male_count = 0
        female_count = 0
        
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            
            # Resize frame for the model
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gender = predict_gender(frame, model)
            if gender == "male":
                male_count += 1
            else:
                female_count += 1
            
            frame_count += 1
        
        video_capture.release()
        st.write(f"Detected Males: {male_count}, Females: {female_count}")
        
        # Determine predominant gender
        predominant_gender = "male" if male_count > female_count else "female"
        st.write(f"Predominant Gender: {predominant_gender}")
        
        # Step 5: Generate audio from the corrected transcription based on gender
        output_audio = "output_audio.mp3"
        generate_audio_from_text(corrected_transcription, output_audio, predominant_gender)
        st.write(f"Audio generated and saved as {output_audio}.")
        
        # Step 6: Replace audio in the original video
        replace_audio_in_video("input_video.mp4", output_audio, "output_video.mp4")
        st.success("Audio replaced successfully! You can download the output video below:")
        
        # Provide a download link for the output video
        with open("output_video.mp4", "rb") as file:
            btn = st.download_button(
                label="Download Output Video",
                data=file,
                file_name="output_video.mp4",
                mime="video/mp4"
            )

if __name__ == "__main__":
    main()
