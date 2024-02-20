import urllib.request
import json
import os
import ssl

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

data = {"data":{"file_path":"no_vocals (1) (1).webm"}}

body = str.encode(json.dumps(data))

url = 'https://humming-model.centralindia.inference.ml.azure.com/score'
# Replace this with the primary/secondary key or AMLToken for the endpoint
api_key = 'ynPXD9iaZKumuGLi3LKTGnIIudv7LmfR'
if not api_key:
    raise Exception("A key should be provided to invoke the endpoint")

headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'humming-model-nw-1' }

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = json.loads(response.read())
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    print(error.info())
    print(error.read().decode("utf8", 'ignore'))