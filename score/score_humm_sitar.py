import os
import logging
import json
import sys
from pydub import AudioSegment
from azure.storage.file import FileService
from azure.storage.fileshare import ShareServiceClient
sys.path.append(os.path.join(os.getenv("AZUREML_MODEL_DIR"), "RVC-CLI-Indian/"))
from rvc.infer import infer


share_name = "backgroundscore-sg"
directory_name = "HummIO"
account_name = "amlzee5sbci1mu5120768980"
account_key = "Zgye3CcxQjnbp8r110Eb9BVtttSA5vGB8Gse+5vYWrjgOG4laKnk0ViIQmF9QZIB9Zy0XH24NSck+AStWlIGYg=="

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model_path
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "RVC-CLI-Indian/")
    print(os.getenv("AZUREML_MODEL_DIR"))
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
    file_path = data['file_path']

    download_from_azure_file(share_name, "humm_input", file_path, os.path.join(os.getenv("AZUREML_MODEL_DIR"), "RVC-CLI-Indian/in", file_path), account_name, account_key)
    infer(12, os.path.join(os.getenv("AZUREML_MODEL_DIR"), "RVC-CLI-Indian/in", file_path), os.path.join(os.getenv("AZUREML_MODEL_DIR"), "RVC-CLI-Indian/out", "gen_" + file_path), os.path.join(os.getenv("AZUREML_MODEL_DIR"), "RVC-CLI-Indian/models/Sitar.pth"),  os.path.join(os.getenv("AZUREML_MODEL_DIR"), "RVC-CLI-Indian/models/index/added_IVF2736_Flat_nprobe_1_Sitar_v2.index"), "cuda:0", "rmvpe")
    # another pipeline without watermark.
    # add_watermark_interval(os.path.join(os.getenv("AZUREML_MODEL_DIR"), "RVC-CLI-Indian/out", "gen_" + file_path))
    upload_to_azure_file(os.path.join(os.getenv("AZUREML_MODEL_DIR"), "RVC-CLI-Indian/out", "gen_" + file_path), share_name, "humm_output", "gen_" + file_path, account_name, account_key)
    return "gen_" + file_path

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

    # Indicate that the file has been successfully downloaded
    print(f"File '{file_name}' has been successfully downloaded to '{destination_path}'.")

def upload_to_azure_file(file_path, share_name, directory_name, file_name, account_name, account_key):
    # Create a FileService object
    file_service = FileService(account_name=account_name, account_key=account_key)
    # Create a file share if it doesn\'t exist
    file_service.create_share(share_name)
    # Upload the file to Azure File Storage
    file_service.create_file_from_path(share_name=share_name,directory_name=directory_name, file_name=file_name, local_file_path=file_path )
    # Indicate that the file has been uploaded
    print(f"File '{file_name}' has been successfully uploaded to Azure File Storage.")

def add_watermark_interval(input_output_audio_path, start_time_ms=2000, interval_ms=4000, watermark_volume_reduction=12):
    download_from_azure_file(share_name, "FMM_inputs", 'DhunVO.wav', os.path.join(os.getenv("AZUREML_MODEL_DIR"), "RVC-CLI-Indian/out", 'DhunVO.wav'), account_name, account_key)
    watermark_audio_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "RVC-CLI-Indian/out", 'DhunVO.wav')
    input_audio = AudioSegment.from_file(input_output_audio_path)
    watermark_audio = AudioSegment.from_file(watermark_audio_path)
    num_intervals = (len(input_audio) - start_time_ms) // interval_ms
    if num_intervals <= 0:
        raise ValueError("Input audio is too short to add the watermark with the specified interval.")
    watermark_audio = watermark_audio - watermark_volume_reduction
    for i in range(1):
        start_time = start_time_ms + i * interval_ms
        input_audio = input_audio.overlay(watermark_audio, position=start_time)
    input_audio.export(input_output_audio_path, format="wav")
    print(f"Watermark added to '{input_output_audio_path}' at 4-seconds intervals with reduced volume.")