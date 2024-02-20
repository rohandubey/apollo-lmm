import os
import logging
import json
from audiocraft.models import musicgen
import torchaudio
from audiocraft.data.audio import audio_write
import uuid
import time
from moviepy.editor import VideoFileClip
import subprocess
import shutil
import pydub
from pydub import effects
from azure.storage.file import FileService
from azure.storage.fileshare import ShareServiceClient
import openai
from pydub import AudioSegment

logging.basicConfig(level=logging.INFO)

output_no_vocals_folder = 'vocal_folder'

# Example usage
share_name = "backgroundscore-sg"
account_name = "amlzee5sbci1mu5120768980"
account_key = "Zgye3CcxQjnbp8r110Eb9BVtttSA5vGB8Gse+5vYWrjgOG4laKnk0ViIQmF9QZIB9Zy0XH24NSck+AStWlIGYg=="


def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "part6")
    model = musicgen.MusicGen.get_pretrained(model_path)
    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info("--- New Generation ---")
    data = json.loads(raw_data)["data"]
    logging.info("Generation starting")
    file_path = data['file_path']
    duration = data['duration']
    prompt = music_completion(data['prompt'])

    return generate_background_music(file_path, duration, prompt), prompt

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

def generate_background_music(file_path=None, duration=10, prompt="happy, serene"):
    duration = duration

    model.set_generation_params(
        use_sampling=True,
        duration=duration,
    )
    start_time = time.time()

    if file_path:
        logging.info(f"\n🚀 Starting Music Generation 🚀")
        logging.info(f"file_path : \t{file_path}, prompt: \t {prompt} , duration : \t {duration}")
        # process the url input and get the file name
        # Split the URL string by '/'
        url_parts = file_path.split('/')

        # Extract the last part of the split string, which should be the .wav file name
        file_name = url_parts[-1]

        # Create and output file - N
        # Azure Storage
        local_input_path = f'storage/{file_name}'
        download_from_azure_file(share_name, "FMM_inputs", file_name, local_input_path, account_name, account_key)
        local_input_path2 = f"storage/{file_name.split('.')[0]}.wav"
        
        music_process(local_input_path, local_input_path2, duration)
        logging.info(f"Audio file downloaded: \t\t{local_input_path2}")
        melody_waveform, sr = torchaudio.load(local_input_path2)
        melody_waveform = melody_waveform.unsqueeze(0).repeat(3, 1, 1)

        prompt = [prompt]*3

        logging.info(f"🎵 Music Generation in Progress... 🎵")
        output = model.generate_with_chroma(
            descriptions=prompt,
            melody_wavs=melody_waveform,
            melody_sample_rate=sr,
            progress=True, return_tokens=True
        )
        # Delete the local file after uploading to Azure
        if os.path.exists(f'separated/htdemucs/{os.path.splitext(os.path.basename(local_input_path))[0]}'):
            shutil.rmtree(f'separated/htdemucs/{os.path.splitext(os.path.basename(local_input_path))[0]}')
        if os.path.exists(local_input_path2):
            os.remove(local_input_path2)
        if os.path.exists(local_input_path):
            os.remove(local_input_path)
    else:
        logging.info(f"\n🚀 Starting Music Generation 🚀")
        logging.info(f"Prompt:          {prompt}")
        logging.info(f"No audio file Given.")

        prompt = [prompt]*3

        logging.info(f"🎵 Music Generation in Progress... 🎵")
        output = model.generate(
            descriptions=prompt,
            progress=True, return_tokens=True
        )
    output_files = write_wav(output[0], 1)
    end_time = time.time()
    logging.info(f"Elapsed time: {end_time-start_time}")

    return output_files


def upload_to_azure_file(file_path, share_name, directory_name, file_name, account_name, account_key):
    # Create a FileService object
    file_service = FileService(account_name=account_name, account_key=account_key)
    # Create a file share if it doesn\'t exist
    file_service.create_share(share_name)
    # Upload the file to Azure File Storage
    file_service.create_file_from_path(share_name=share_name,directory_name=directory_name, file_name=file_name, local_file_path=file_path )
    # Indicate that the file has been uploaded
    logging.info(f"File '{file_name}' has been successfully uploaded to Azure File Storage.")

def write_wav(output, upload_to_azure_flag):
    file_names = []
    try:
        for idx, one_wav in enumerate(output):
            # Generate a unique identifier for each audio file
            sing_id = uuid.uuid4()
            filename = f'storage/{sing_id}'
            
            # Write the audio file using the specified strategy
            audio_write(filename, one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
            
            # Upload to Azure if upload_to_azure_flag is True
            if upload_to_azure_flag:
                add_watermark_interval(f'{filename}.wav')
                upload_to_azure_file(f'{filename}.wav', share_name, "FMM_outputs", f'{sing_id}.wav', account_name, account_key)
                
                # Delete the local file after uploading to Azure
                os.remove(f'{filename}.wav')
                
            file_names.append(f'{sing_id}.wav')
        
        return file_names
    except Exception as e:
        logging.info("Error while writing or uploading the file:", e)
        return None

def detect_media_type(file_path):
    _, extension = os.path.splitext(file_path)
    audio_extensions = ['.mp3', '.wav', '.ogg', '.flac']

    if extension.lower() in audio_extensions:
        return 0
    else:
        return 1


def adjust_volume(segment, target_volume):
    current_volume = segment.dBFS
    adjustment = target_volume - current_volume
    return segment + adjustment


def extract_no_vocals(input_audio_path, output_no_vocals_folder):
    # Execute Demucs command

    demucs_executable = shutil.which("demucs")
    if demucs_executable is not None:
        subprocess.run([demucs_executable, f"--two-stems=vocals", input_audio_path])
    else:
        logging.info("Error: demucs executable not found in PATH.")
    logging.info(f'Vocals extracted and saved as {output_no_vocals_folder}')


def extract_audio_from_video(video_path, output_audio_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_audio_path)
    logging.info(f'video processed at {output_audio_path}')


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


def music_process(file_path,output_audio_path, duration):
    if detect_media_type(file_path) == 1:
        extract_audio_from_video(file_path, output_audio_path)
        logging.info(f'Audio extracted from {file_path} and saved as {output_audio_path}')
        file_path_n = output_audio_path
    else:
        logging.info(f'The file {file_path} is already an audio file.')
        file_path_n = file_path
    extract_no_vocals(file_path_n, output_no_vocals_folder)
    logging.info(f'separated/htdemucs/{os.path.splitext(os.path.basename(file_path))[0]}/no_vocals.wav')
    process_audio(f'separated/htdemucs/{os.path.splitext(os.path.basename(file_path))[0]}/no_vocals.wav', output_audio_path, duration)

def download_from_azure_file(share_name, directory_name, file_name, destination_path, account_name, account_key):
    # Create a ShareServiceClient object
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
    service_client = ShareServiceClient.from_connection_string(connection_string)

    # Get a reference to the share
    share_client = service_client.get_share_client(share_name)

    # Check if the file exists
    file_client = share_client.get_directory_client(directory_name).get_file_client(file_name)
    os.makedirs("storage", exist_ok=True)
    os.makedirs("seperated/htdemucs", exist_ok=True)

    # Download the file to the specified destination path
    with open(destination_path, "wb") as file:
        file_data = file_client.download_file().readall()
        file.write(file_data)

    # Indicate that the file has been successfully downloaded
    logging.info(f"File '{file_name}' has been successfully downloaded to '{destination_path}'.")

def add_watermark_interval(input_output_audio_path, start_time_ms=2000, interval_ms=4000, watermark_volume_reduction=12):
    download_from_azure_file(share_name, "FMM_inputs", 'DhunVO.wav', 'DhunVO.wav', account_name, account_key)
    watermark_audio_path = 'DhunVO.wav'
    input_audio = AudioSegment.from_file(input_output_audio_path)
    watermark_audio = AudioSegment.from_file(watermark_audio_path)
    num_intervals = (len(input_audio) - start_time_ms) // interval_ms
    if num_intervals <= 0:
        raise ValueError("Input audio is too short to add the watermark with the specified interval.")
    watermark_audio = watermark_audio - watermark_volume_reduction
    for i in range(num_intervals):
        start_time = start_time_ms + i * interval_ms
        input_audio = input_audio.overlay(watermark_audio, position=start_time)
    input_audio.export(input_output_audio_path, format="wav")
    logging.info(f"Watermark added to '{input_output_audio_path}' at 4-seconds intervals with reduced volume.")
