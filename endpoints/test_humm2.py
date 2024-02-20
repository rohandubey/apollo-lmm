import urllib.request
import json
import os
import ssl

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

# Request data goes here
# The example below assumes JSON formatting which may be updated
# depending on the format your endpoint expects.
# More information can be found here:
# https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
data = {"data":{"file_path":"longhumm.mp3"}}

body = str.encode(json.dumps(data))

url = 'https://humming-model2.centralindia.inference.ml.azure.com/score'
# Replace this with the primary/secondary key or AMLToken for the endpoint
api_key = 'JIm5LEffhS6cAGF8STO9gr3pxPQQtRvp'
if not api_key:
    raise Exception("A key should be provided to invoke the endpoint")

# The azureml-model-deployment header will force the request to go to a specific deployment.
# Remove this header to have the request observe the endpoint traffic rules
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'humming-model2nw-1' }

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = json.loads(response.read())
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(error.read().decode("utf8", 'ignore'))
    
message_text = [{"role":"system","content":"Input 1: Create a straightforward music generation prompt for the AI model, focusing on the preferred genre and style. Direct the AI to compose a piece without specifying its duration. Keep the prompt general and versatile, excluding specific details like movie names.\nInput 2: Take the output from previous step and generate a comma-separated set of instructions for AI music generation based on a given input. Include at least 6 meaningful words (adjectives, verbs, etc.) to convey the context. Guide the music generation with details like emotion, genre, moods, tempo (high, medium, or low), pitch (high, medium, or low), and other relevant adjectives. Avoid adding any names/nouns or keys OR instruments. Dont suggest any instrument in the output."},{"role":"user","content":"Generate me music in which a hero is standing and staring at the Indian flag and is having a patriotic feeling. He is angry and wants to take revenge against the corrupt politicians."},{"role":"assistant","content":"victorious, struggle, high tempo, medium pitch, anger, revenge, patriotic, drums, electric guitar, passionate"},{"role":"user","content":"Give me Linkin Park style music."},{"role":"assistant","content":"alternative rock, high tempo, intense, powerful, emotional, rebellious, electric guitar, drums"},{"role":"user","content":"Give me catchy musical logo."},{"role":"assistant","content":"upbeat, memorable, high tempo, high pitch, lively, synthesizer, drums"},{"role":"user","content":"music suitable for Indian saree brand"},{"role":"assistant","content":"elegant, enchanting, uplifting, classical, contemporary, fusion, sitar, tabla, medium tempo, medium pitch"}]
 