import os
import uuid
import logging
import time
import subprocess
import json
from tqdm import tqdm
from IPython.display import Audio
import torch
import torchaudio
from azure.storage.file import FileService
from azure.storage.fileshare import ShareServiceClient
import openai
import gradio as gr
from audiocraft.models import musicgen
from audiocraft.utils.notebook import display_audio
from audiocraft.data.audio import audio_write
import pydub
from pydub import effects
from pydub import AudioSegment
import shutil

# Constants
PASSWORD_CHECK = "Dhundhun@987!"

# Configure Vocals
output_no_vocals_folder = 'vocal_folder'
if not os.path.exists(output_no_vocals_folder):
    os.makedirs(output_no_vocals_folder)

# Configure Logging
LOG_FILE = "music_generation.log"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")

# Azure Storage Configuration
SHARE_NAME = "backgroundscore-sg"
DIRECTORY_NAME = "GeneratedMusic"
ACCOUNT_NAME = "amlzee5sbci1mu5120768980"
ACCOUNT_KEY = "Zgye3CcxQjnbp8r110Eb9BVtttSA5vGB8Gse+5vYWrjgOG4laKnk0ViIQmF9QZIB9Zy0XH24NSck+AStWlIGYg=="

# OpenAI Configuration
openai.api_type = "azure"
openai.api_base = "https://aoai-zee5-sb-sc1-musicstudio-0001.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = "4072d7aa8afc4848b8c0a3e4f6e9209a"

# MusicGen Model
MODEL_PATH = '/mnt/music-data/models/part7/part7/'
model = musicgen.MusicGen.get_pretrained(MODEL_PATH)
logging.info(f"Complete Model is loaded from path: {MODEL_PATH}")

# Default Generation Parameters
DEFAULT_DIVERSITY = 512
DEFAULT_TEMPERATURE = 1.05
DEFAULT_GUIDANCE = 2.5
DEFAULT_DURATION = 15

model.set_generation_params(
    use_sampling=True,
    duration=DEFAULT_DURATION
)

# Dictionary to map model choices to prompts
MODEL_PROMPTS = {
    "Grand Piano": "individual grand piano",
    "Acoustic Guitar": "individual acoustic guitar",
    # Add more instruments as needed
}

def process_audio(input_path, output_path, duration):
    # Load the music track
    accompaniment = pydub.AudioSegment.from_wav(input_path)
    duration_ms = duration * 1000
    accompaniment = accompaniment[:duration_ms]
    sr = accompaniment.frame_rate
    accompaniment = accompaniment.set_frame_rate(22000)
    accompaniment = effects.normalize(accompaniment, headroom=1.2)
    # Normalize the accompaniment
    accompaniment = effects.normalize(accompaniment)
    # Compress the accompaniment
    accompaniment = effects.compress_dynamic_range(accompaniment, threshold=-20.0, ratio=6.0, attack=5.0, release=60.0)
    # Boost the treble frequencies
    accompaniment = effects.low_pass_filter(accompaniment, cutoff=4000)
    # Save the enhanced track
    accompaniment.export(output_path, format='wav')
    logging.info(f'The file {output_path} is normalized.')

def extract_no_vocals(input_audio_path):
    # Execute Demucs command
    demucs_executable = shutil.which("/anaconda/envs/dlwp/bin/demucs")
    if demucs_executable is not None:
        subprocess.run([demucs_executable, f"--two-stems=vocals", input_audio_path])
    else:
        logging.info("Error: demucs executable not found in PATH.")
    logging.info(f'Vocals extracted and saved')

def music_process(file_path,output_audio_path, duration):
    extract_no_vocals(file_path)
    logging.info(f'separated/htdemucs/{os.path.splitext(os.path.basename(file_path))[0]}/no_vocals.wav')
    process_audio(f'separated/htdemucs/{os.path.splitext(os.path.basename(file_path))[0]}/no_vocals.wav', output_audio_path, duration)

def music_completion2(prompt_string):
    openai.api_type = "azure"
    openai.api_base = "https://aoai-zee5-sb-sc1-musicstudio-0001.openai.azure.com/"
    openai.api_version = "2023-07-01-preview"
    openai.api_key = "4072d7aa8afc4848b8c0a3e4f6e9209a"

    message_text = [{"role": "system", "content": "Input 1: Create a straightforward music generation prompt for the AI model, focusing on the preferred genre and style. Direct the AI to compose a piece without specifying its duration. Keep the prompt general and versatile, excluding specific details like movie names.\nInput 2: Take the output from previous step and generate a comma-separated set of instructions for AI music generation based on a given input. Include at least 6 meaningful words (adjectives, verbs, etc.) to convey the context. Guide the music generation with details like emotion, genre, moods, tempo (high, medium, or low), pitch (high, medium, or low), and other relevant adjectives. Avoid adding any names/nouns or keys or instruments. Dont suggest any instrument in the output."},{"role": "user","content": "Generate me music in which a hero is standing and staring at the Indian flag and is having a patriotic feeling. He is angry and wants to take revenge against the corrupt politicians."},{ "role": "assistant","content": "victorious, struggle, high tempo, medium pitch, anger, revenge, patriotic, passionate"},{"role": "user", "content": "Give me Linkin Park style music."},{"role": "assistant","content": "alternative rock, high tempo, intense, powerful, emotional, rebellious"},{ "role": "user","content": "Give me catchy musical logo."},{"role": "assistant","content": "upbeat, memorable, high tempo, high pitch, lively"},{ "role": "user","content": "music suitable for Indian saree brand"},{"role": "assistant","content": "elegant, enchanting, uplifting, classical, contemporary, medium tempo, medium pitch"}]
    new_user_message = {"role": "user", "content": prompt_string.lower()}
    message_text.append(new_user_message)

    completion = openai.ChatCompletion.create(
        # The above code snippet appears to be incomplete and contains syntax errors. It seems to be
        # a Python script, but it is missing some necessary components such as function definitions,
        # variable assignments, and proper indentation.
        engine="test-gpt4",
        messages = message_text,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
        )
    return completion['choices'][0]['message']['content']

def music_completion(prompt_string):
    openai.api_type = "azure"
    openai.api_base = "https://aoai-zee5-sb-sc1-musicstudio-0001.openai.azure.com/"
    openai.api_version = "2023-07-01-preview"
    openai.api_key = "4072d7aa8afc4848b8c0a3e4f6e9209a"

    message_text = [{"role":"system","content":"Input 1: Create a straightforward music generation prompt for the AI model, focusing on the preferred genre and style. Direct the AI to compose a piece without specifying its duration. Keep the prompt general and versatile, excluding specific details like movie names.\nInput 2: Take the output from previous step and generate a comma-separated set of instructions for AI music generation based on a given input. Include at least 6 meaningful words (adjectives, verbs, etc.) to convey the context. Guide the music generation with details like emotion, genre, moods, tempo (high, medium, or low), pitch (high, medium, or low), instruments (Indian and/or Western), and other relevant adjectives. Avoid adding any names/nouns or keys. Dont use flute in music."},{"role":"user","content":"Generate me music in which a hero is standing and staring at the Indian flag and is having a patriotic feeling. He is angry and wants to take revenge against the corrupt politicians."},{"role":"assistant","content":"victorious, struggle, high tempo, medium pitch, anger, revenge, patriotic, drums, electric guitar, passionate"},{"role":"user","content":"Give me Linkin Park style music."},{"role":"assistant","content":"alternative rock, high tempo, intense, powerful, emotional, rebellious, electric guitar, drums"},{"role":"user","content":"Give me catchy musical logo."},{"role":"assistant","content":"upbeat, memorable, high tempo, high pitch, lively, synthesizer, drums"},{"role":"user","content":"music suitable for Indian saree brand"},{"role":"assistant","content":"elegant, enchanting, uplifting, classical, contemporary, fusion, sitar, tabla, medium tempo, medium pitch"}]
    new_user_message = {"role": "user", "content": prompt_string.lower()}
    message_text.append(new_user_message)

    completion = openai.ChatCompletion.create(
        engine="test-gpt4",
        messages = message_text,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
        )
    return completion['choices'][0]['message']['content']

def upload_to_azure_file(file_path, share_name, directory_name, file_name, account_name, account_key):
    try:
        file_service = FileService(account_name=account_name, account_key=account_key)
        file_service.create_share(share_name)
        file_service.create_file_from_path(
            share_name=share_name,
            directory_name=directory_name,
            file_name=file_name,
            local_file_path=file_path
        )
        logging.info(f"File '{file_name}' has been successfully uploaded to Azure File Storage.")
    except Exception as e:
        logging.error(f"Error uploading file '{file_name}' to Azure: {str(e)}")

def write_wav(output, upload_to_azure_flag):
    file_names = []
    try:
        for idx, one_wav in enumerate(output):
            sing_id = uuid.uuid4()
            filename = f'sample_files/{sing_id}'
            audio_write(filename, one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
            
            if upload_to_azure_flag:
                upload_to_azure_file(f'{filename}.wav', SHARE_NAME, DIRECTORY_NAME, f'{sing_id}.wav', ACCOUNT_NAME, ACCOUNT_KEY)
            
            file_names.append(f'sample_files/{sing_id}.wav')
        logging.info(f"Generated File Names: {file_names}")
        
        return file_names
    except Exception as e:
        logging.error(f"Error while writing or uploading the file: {str(e)}")
        return None

def generate_music(prompt_text, melody_audio=None, duration=DEFAULT_DURATION, model_choice="A"):
    model.set_generation_params(
        use_sampling=True,
        duration=duration
    )
    # Log inputs
    logging.info(f"Prompt Text: {prompt_text}")
    logging.info(f"Duration: {duration}")

    # Get the prompt for the specified model_choice from the dictionary
    if model_choice in MODEL_PROMPTS:
        prompt = [f"{MODEL_PROMPTS[model_choice]}, {music_completion2(prompt_text)}"]
        logging.warning(f"Using the Nano model.")
        logging.info(f"Model Choice: {model_choice}")
    else:
        logging.warning(f"Using the Complete model.")
        logging.info(f"Model Choice: FMM")
        prompt = [music_completion(prompt_text)]
    logging.info(f"Chosen Prompt: {prompt[0]}")
    
    prompt *= 3
    
    if melody_audio:
        logging.info(f"Refernce Melody uploaded at: {melody_audio}")
        melody_waveform, sr = torchaudio.load(melody_audio)
        music_process(melody_audio, melody_audio, duration)
        melody_waveform = melody_waveform.unsqueeze(0).repeat(3, 1, 1)
        output = model.generate_with_chroma(
            descriptions=prompt,
            melody_wavs=melody_waveform,
            melody_sample_rate=sr,
            progress=True,
            return_tokens=True
        )
    else:
        output = model.generate(
            descriptions=prompt,
            progress=True,
            return_tokens=True
        )
    
    return write_wav(output[0], 1), prompt

def generate_and_display(prompt_text, melody_audio, duration, model_choice):
    # Log separator for a new generation
    logging.info("--- New Generation ---")
    generated_paths, prompt = generate_music(prompt_text, melody_audio, duration, model_choice)
    print(duration)
    if len(generated_paths) >= 3:
        output_music1.value = generated_paths[0]
        output_music2.value = generated_paths[1]
        output_music3.value = generated_paths[2]
        return generated_paths[0], generated_paths[1], generated_paths[2], prompt[0]
    else:
        print("Error: Not enough generated paths.")

# Gradio Interface
with gr.Blocks() as demo:
    with gr.Tab("FMM"):
        # Define the Gradio interface for the first tab
        input_prompt = gr.Textbox(label="Enter Music Description", placeholder="Describe your music...", lines=2)
        input_melody = gr.Audio(sources="upload", type="filepath")
        duration_slider = gr.Slider(minimum=5, maximum=60, step=1, value=DEFAULT_DURATION, label="Select duration:")
        submit_button = gr.Button("Submit")

        prompt_output = gr.Textbox(label="Actual Prompt for Music", placeholder="The prompt that model takes will appear here...", lines=1)
        output_music1 = gr.Audio(type='filepath', label="Generated Music 1")
        output_music2 = gr.Audio(type='filepath', label="Generated Music 2")
        output_music3 = gr.Audio(type='filepath', label="Generated Music 3")

        submit_button.click(generate_and_display, inputs=[input_prompt, input_melody, duration_slider, input_prompt], outputs=[output_music1, output_music2, output_music3, prompt_output])

    with gr.Tab("Nano Models"):
        # Define the Gradio interface for the second tab
        model_choice_dropdown = gr.Dropdown(label="Select Model", choices=["Grand Piano", "Acoustic Guitar"], value="Grand Piano")
        input_prompt2 = gr.Textbox(label="Enter Music Description", placeholder="Describe your music...", lines=2)
        input_melody2 = gr.Audio(sources="upload", type="filepath")
        duration_slider2 = gr.Slider(minimum=5, maximum=60, step=1, value=DEFAULT_DURATION, label="Select duration:")
        submit_button2 = gr.Button("Submit")

        prompt_output2 = gr.Textbox(label="Actual Prompt for Music", placeholder="The prompt that model takes will appear here...", lines=1)
        output_music4 = gr.Audio(type='filepath', label="Generated Music 1")
        output_music5 = gr.Audio(type='filepath', label="Generated Music 2")
        output_music6 = gr.Audio(type='filepath', label="Generated Music 3")

        submit_button2.click(generate_and_display, inputs=[input_prompt2, input_melody2, duration_slider2, model_choice_dropdown], outputs=[output_music4, output_music5, output_music6, prompt_output2])

demo.launch(share=True, auth=("dhuner", PASSWORD_CHECK))
