# AI_Voice_Generator

## Overview
The **AI Voice Generator** project takes you on an exciting journey of transforming videos into meaningful audio experiences. With the power of cutting-edge AI, you can:
1. Extract text from videos by processing their audio.
2. Summarize the extracted text for clarity and brevity.
3. Generate high-quality speech audio from the summarized text.

This tool combines the capabilities of **OpenAI**, **Hugging Face Transformers**, and **Google Speech Recognition** to create an all-in-one solution for video-to-text and text-to-speech tasks.

## Features
- **Video Processing**: Effortlessly upload videos in popular formats like `.mp4`, `.mov`, and `.avi`.
- **Audio Extraction**: Automatically extract audio from your uploaded video.
- **Speech-to-Text**: Convert audio into text with reliable Google Speech Recognition.
- **Text Summarization**: Summarize and refine the text using advanced Hugging Face models.
- **Text-to-Speech (TTS)**: Transform text into natural-sounding speech with OpenAI's API.
- **Streamlit Interface**: Enjoy a user-friendly UI that supports light/dark modes and tracks progress in real time.

## Technologies Used
This project is built on a robust stack of technologies:
- **Programming Language**: Python
- **Framework**: Streamlit
- **Libraries**:
  - `cv2` for handling video operations
  - `moviepy` for audio extraction
  - `speech_recognition` for speech-to-text conversion
  - `transformers` for text summarization
  - `openai` for generating speech from text

## Installation
Ready to dive in? Follow these steps to set up the project:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/AI_Voice_Generator.git
   cd AI_Voice_Generator
   ```

2. **Set Up a Virtual Environment**:
   Create and activate a virtual environment to isolate dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure OpenAI API Key**:
   Open the script and replace the placeholder API key (`"sk-proj-..."`) with your actual OpenAI API key.

## Usage
Let’s bring this project to life! Here’s how you can use it:

1. **Run the Application**:
   Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. **Upload Your Video**:
   Use the intuitive UI to upload your video file.

3. **Process the Video**:
   - Extract audio from the video.
   - Transcribe the audio into text.
   - Refine and summarize the text.

4. **Generate Speech**:
   Convert the summarized text into high-quality speech and listen to the output directly in the app.

## File Structure
Here’s a quick look at the project structure:
```
AI_Voice_Generator/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Required Python packages
├── README.md               # Project documentation
└── extracted_audio.wav     # Temporary audio file (auto-generated)
```

## Example Workflow
Imagine this:
1. You **upload a video** file.
2. The app **extracts the audio** seamlessly.
3. The audio is **transcribed into text**, ready for you to review.
4. The transcription is **summarized and refined** automatically.
5. The refined text is used to **generate natural-sounding audio**, giving you a complete audio transformation experience.

## Key Highlights
- **Real-Time Updates**: Monitor progress at every step with intuitive progress bars.
- **Light/Dark Modes**: Customize the interface to your liking.
- **Editable Text**: Modify the transcribed text before generating the final output.

## Future Enhancements
We’re just getting started! Here’s what’s on the horizon:
- Multi-language support for transcription and text-to-speech.
- Enhanced summarization with fine-tuned transformer models.
- Options for voice customization in TTS output.

## License
This project is licensed under the MIT License.

## Contributing
Have ideas for improvement? Join us! Contributions are always welcome. Submit issues or pull requests to help us grow.

## Contact
Got questions? Reach out at omunde2016@gmail.com
