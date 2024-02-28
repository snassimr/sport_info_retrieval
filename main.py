### Setup
import os
import logging

from service_utils import *
from app_utils import *

## Load env variables , to be adjusted
load_config_from_yaml("config.yaml")

### $$$ Local imports
import importlib
import service_utils
importlib.reload(service_utils)
from service_utils import *
import app_utils
importlib.reload(app_utils)
from app_utils import *

## Initialize logging
logging.basicConfig(filename='log.log', level=logging.INFO, 
                    format='%(asctime)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

### Set parameters
YOUTUBE_URL = "https://www.youtube.com/watch?v=3j_daNHio4o"
## Data
DATA_FOLDER  = 'data'
AUDIO_FOLDER = os.path.join(DATA_FOLDER, "audio")
TEXT_FOLDER  = os.path.join(DATA_FOLDER, "text")
## Audio-to-text
AUDIO_TO_TEXT_MODEL_FOLDER = os.environ.get("AUDIO_TO_TEXT_MODEL_FOLDER")
AUDIO_TO_TEXT_MODEL_NAME   = os.environ.get("AUDIO_TO_TEXT_MODEL_NAME")
AUDIO_TO_TEXT_MODEL_PATH   = os.path.join(AUDIO_TO_TEXT_MODEL_FOLDER, AUDIO_TO_TEXT_MODEL_NAME)

### Download youtube audio
VIDEO_NAME, AUDIO_FILE_PATH = download_youtube_audio(YOUTUBE_URL, AUDIO_FOLDER)

TEXT_FILE_PATH   = os.path.join(TEXT_FOLDER, VIDEO_NAME + '.txt')

print(f"{VIDEO_NAME} audio size is : {os.path.getsize(AUDIO_FILE_PATH)} bytes.")

### Get audio text transcript
text = get_audio_text(AUDIO_FILE_PATH, AUDIO_TO_TEXT_MODEL_PATH)

free_gpu_memory()

save_text(TEXT_FILE_PATH, text)

print(f"{VIDEO_NAME} text size is : {os.path.getsize(TEXT_FILE_PATH)} bytes.")

### Retrieve info
text = read_text(TEXT_FILE_PATH)

prompt = generate_prompt(text)

info = get_info(prompt, model = "gpt-4-1106-preview")

display_info(info, VIDEO_NAME)