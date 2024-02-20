import os
import logging
import json
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
import torch
import uuid
from azure.storage.file import FileService
from azure.storage.fileshare import ShareServiceClient
import openai
from pydub import AudioSegment


import os
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
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "run_6")
    model = AudioGen.get_pretrained(model_path)
    download_from_azure_file(share_name, "FMM_inputs", 'DhunVO.wav', 'DhunVO.wav', account_name, account_key)
    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info("Request received")
    data = json.loads(raw_data)["data"]
    logging.info("Request processed")
    duration = data['duration']
    prompt = music_completion(data['prompt'])

    return generate_sfx(prompt, duration)


def generate_sfx(prompt,duration):
    prompt = music_completion(prompt)
    print(prompt)
    prompt = [prompt]*3
    model.set_generation_params(duration=duration) 
    wav = model.generate(prompt)  # generates 3 samples.
    return write_wav(wav,1)

def music_completion(prompt_string):
  openai.api_type = "azure"
  openai.api_base = "https://aoai-zee5-sb-sc1-musicstudio-0001.openai.azure.com/"
  openai.api_version = "2023-07-01-preview"
  openai.api_key = "4072d7aa8afc4848b8c0a3e4f6e9209a"

  message_text = [{"role":"system","content":"You are an AI that takes in an input texts and understands the required sound/SFX/foley that is needed to be generated and pass just the required text to the user. Give output in lowercase and never use a proper noun."},{"role":"user","content":"i want barking dog sound"},{"role":"assistant","content":"barking dog"},{"role":"user","content":"I want a sound for a rainy scene and people are walking"},{"role":"assistant","content":"pedestrian walking in rain"}]
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
    # Create a FileService object
    file_service = FileService(account_name=account_name, account_key=account_key)
    # Create a file share if it doesn\'t exist
    file_service.create_share(share_name)
    # Upload the file to Azure File Storage
    file_service.create_file_from_path(share_name=share_name,directory_name=directory_name, file_name=file_name, local_file_path=file_path )
    # Indicate that the file has been uploaded
    print(f"File '{file_name}' has been successfully uploaded to Azure File Storage.")

def write_wav(output, upload_to_azure_flag):
    file_names = []
    try:
        for idx, one_wav in enumerate(output):
            # Generate a unique identifier for each audio file
            sing_id = uuid.uuid4()
            filename = f'sample_files/{sing_id}'
            
            # Write the audio file using the specified strategy
            audio_write(filename, one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
            
            # Upload to Azure if upload_to_azure_flag is True
            if upload_to_azure_flag:
                add_watermark_interval(f'{filename}.wav')
                upload_to_azure_file(f'{filename}.wav', share_name, "SFX_outputs", f'{sing_id}.wav', account_name, account_key)
            
            file_names.append(f'{sing_id}.wav')
        
        return file_names
    except Exception as e:
        print("Error while writing or uploading the file:", e)
        return None

def download_from_azure_file(share_name, directory_name, file_name, destination_path, account_name, account_key):
    # Create a ShareServiceClient object
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
    service_client = ShareServiceClient.from_connection_string(connection_string)

    # Get a reference to the share
    share_client = service_client.get_share_client(share_name)

    # Check if the file exists
    file_client = share_client.get_directory_client(directory_name).get_file_client(file_name)

    # Download the file to the specified destination path
    with open(destination_path, "wb") as file:
        file_data = file_client.download_file().readall()
        file.write(file_data)

def add_watermark_interval(input_output_audio_path, start_time_ms=1000, watermark_volume_reduction=12):
    watermark_audio_path = 'DhunVO.wav'
    input_audio = AudioSegment.from_file(input_output_audio_path)
    watermark_audio = AudioSegment.from_file(watermark_audio_path)

    if len(input_audio) <= start_time_ms:
        raise ValueError("Input audio is too short to add the watermark at the specified start time.")
        logging.info("Input audio is too short to add the watermark at the specified start time.")
    
    watermark_audio = watermark_audio - watermark_volume_reduction
    input_audio = input_audio.overlay(watermark_audio, position=start_time_ms)
    
    input_audio.export(input_output_audio_path, format="wav")
    logging.info(f"Watermark added to '{input_output_audio_path}' at {start_time_ms / 1000} seconds with reduced volume.")
