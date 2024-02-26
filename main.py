import os
from typing import List
import logging
logging.basicConfig(filename='demo.log', encoding='utf-8', level=logging.ERROR)

# YOUTUBE_URL = "https://www.youtube.com/watch?v=szxiPMyuMGY"
YOUTUBE_URL = "https://www.youtube.com/watch?v=3j_daNHio4o"
VIDEO_NAME="demo"
VIDEO_NAME = "FULL MATCH: Real Madrid 2 - 3 Barça (2017) Messi grabs dramatic late win in #ElClásico!!"

def download_youtube_audio(url: str) -> str:
    """Download the audio from a YouTube video and save it as an MP3 file."""
    from pytube import YouTube
    video_url= YouTube(url)
    video_name = video_url.title
    video = video_url.streams.filter(only_audio=True).first()
    filename = video_name + ".mp3"
    video.download(filename=filename)
    return filename

download_youtube_audio(YOUTUBE_URL)

from IPython.display import Audio

# Load an audio file
audio_file = 'path/to/your/audio.mp3'

# Display an audio player in the Jupyter notebook
Audio(audio_file)

# Getting the title of the video


def save_text_to_file(text: str, file_name: str):
    """Save the transcribed text to a file."""
    try:
        with open(file_name, "w+") as file:
            file.write(text)
    except (IOError, OSError, FileNotFoundError, PermissionError) as e:
        logging.debug(f"Error in file operation: {e}")

def get_text(url: str, video_name: str, model_name: str = "medium", language: str = "English") -> None:
    """Load the medium multilingual Whisper model."""
    import whisper
    model = whisper.load_model(model_name)
    audio_path = download_youtube_audio(url, video_name)
    result = model.transcribe(audio_path, fp16=False, language=language)
    save_text_to_file(result["text"], video_name + ".txt")

get_text(url=YOUTUBE_URL, video_name=VIDEO_NAME)

import os
from dotenv import load_dotenv
load_dotenv()

# Access the OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')

import openai

prompt = """

"""

resp = openai.ChatCompletion.create(model="gpt-4",
                                    messages=[{"role":"user", "content":prompt}],
                                    max_tokens=1024, 
                                    temperature=0.1)

