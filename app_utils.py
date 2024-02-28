###################### sport_info_retrieval/app_utils.py

import importlib
import service_utils
importlib.reload(service_utils)
from service_utils import *

def download_youtube_audio(url: str, audio_folder : str) -> str:
    """
    Downloads the audio from a YouTube video and saves it as an MP3 file in the specified folder.

    :param url: The URL of the YouTube video.
    :type url: str
    :param audio_folder: The path to the directory where the audio file will be saved.
    :type audio_folder: str
    :return: A tuple containing the video's name and the path to the saved audio file.
    :rtype: tuple
    """
    import os
    from pytube import YouTube
    video_url= YouTube(url)
    video_name = video_url.title
    video = video_url.streams.filter(only_audio=True).first()
    audio_filepath = os.path.join(audio_folder, video_name + ".mp3")
    video.download(filename=audio_filepath)
    return video_name, audio_filepath

@runtime_logger
@gpu_memory_logger
def get_audio_text(audio_filepath: str, audio_to_text_model_path:str):
    """
    Transcribes audio to text using a specified model from Hugging Face Transformers.

    This function loads a speech-to-text model specified by the path to its pretrained weights and configurations.
    The function is decorated with `@runtime_logger` to log the execution time
    and `@gpu_memory_logger` to log the GPU memory usage during the transcription process.
    
    :param audio_filepath: The file path of the audio to be transcribed.
    :type audio_filepath: str
    :param audio_to_text_model_path: The path to the pretrained model and tokenizer for transcription.
    :type audio_to_text_model_path: str
    :return: The transcribed text from the audio file.
    :rtype: str
    """

    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        audio_to_text_model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(audio_to_text_model_path)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=False,
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(audio_filepath)
    return result['text']

@error_logger
def save_text(file_path: str, text: str) -> None:
    """
    Saves given text to a specified file.
    The function is decorated with `@error_logger` to log errors.

    :param file_path: The path to the file where the text will be saved.
    :type file_path: str
    :param text: The text content to be saved in the file.
    :type text: str
    :return: None
    """
    with open(file_path, "w+", encoding='utf-8') as file:
        file.write(text)

@error_logger
def read_text(file_path: str) -> str:
    """
    Read text from a file and return it as a string.
    The function is decorated with `@error_logger` to log errors.

    :param file_path: The path to the text file to be read.
    :type file_path: str
    :return: The content of the file as a string.
    :rtype: str
    """
    with open(file_path, "r", encoding='utf-8') as file:
        return file.read()
    
def generate_prompt(text : str) -> str:

    """
    Generates a detailed prompt for extracting information about a sport event from a given text.

    :param text: A string containing the description of a sport event.
    :type text: str
    :return: A formatted prompt string with detailed instructions for information extraction.
    :rtype: str
    """

    prompt = f"""

    Instruction
    ===========
    Based on information below output information about sport event is described in Content.
    Each Section in Outputs has Python dictionary format with keys and values.
    Combine all section-specific dictionaries into a single dictionary output.

    Content
    =======
    {text}

    Outputs
    =======
    1. Section : Basic game information 
    Instructions:
        a. What sport is described in the Content ?
        b. Which country hosts the competition according to the Content ?
        c. What competition is described in Content ?
        d. Which City/Town hosts the competition according to the Content ?
        e. Which sport facility hosts the competition according to the Content ?
        f. Set NA if you are not enable to find the information for a , b or c
    Dictionary format :
            Section Key : basic
                Key : Sport , Value : Answer to a.
                Key : Country , Value : Answer to b.
                Key : Competition , Value : Answer to c.
                Key : City , Value : Answer to d.
                Key : Facility, Value : Answer to e.
    2. Section : Teams and final result 
    Instructions :
            a. first command is the host the game
            c. Set NA if you are not enable to find the information for a or b
    Dictionary format :
            Section Key : result
                Key : Team_1 Value : Name of host team
                Key : Team_1 Score : Score of host team
                Key : Team_2 Name : Name of guest team
                Key : Team_2 Score : Score of guest team
    3. Section : Game timeline
        Instructions :
            a. List minutes during the game when the goals were scored
            b. List of the full names of players who scored the goals
            c. Set NA if you are not enable to match between minute and player for goal was scored
        Dictionary format :
            Section Key : timeline
                Key : Value from list in List of minutes during the game when the goals were scored , Value : List of players who scored the goals

    """

    return prompt

def get_info(prompt: str, model: str = "gpt-4-1106-preview") -> dict:
    """
    Evaluates and returns information from llm based on prompt using llama_index completion and OpenAI model

    :param prompt: Text prompt for the model.
    :type prompt: str
    :param model: llm model for text generation, default "gpt-4-1106-preview".
    :type model: str
    :return: A Python object evaluated from the model's response.
    :rtype: dict
    """

    import os
    from openai import OpenAI  
    from llama_index.llms.openai import OpenAI

    # Initialize the llm with specified parameters
    llm = OpenAI(model=model, temperature=0.0, max_tokens=512)

    # Get the response from the llm
    response = llm.complete(prompt)

    # Extract the text from response
    answer = response.text

    # Format text as Python dictionary
    info = eval(answer.strip('```python').strip('```').strip())

    return info

def display_info(info: dict, video_name: str):
    """
    Displays detailed information about a video in HTML format.

    :param info: A dictionary containing video information
    :type info: dict
    :param video_name: The title of the video to be displayed.
    :type video_name: str
    """

    from IPython.display import HTML

    # Constructing the HTML with the dynamic title
    html = f"""
    <div style="font-family: Arial, sans-serif;">
        <h2>Video : {video_name}</h2>
        <h3>Basic Information</h3>
        <ul>
    """

    for key, value in info['basic'].items():
        html += f"<li>{key}: {value}</li>"

    html += """
        </ul>
        <h3>Game Result</h3>
        <p>{Team_1} {Team_1_Score} - {Team_2_Score} {Team_2}</p>
        <h3>Timeline</h3>
        <ul>
    """.format(**info['result'])

    for minute, scorer in info['timeline'].items():
        html += f"<li>{minute}': {scorer}</li>"

    html += """
        </ul>
    </div>
    """

    # Displaying the HTML
    display(HTML(html))
