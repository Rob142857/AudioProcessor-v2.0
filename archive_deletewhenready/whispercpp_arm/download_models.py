import os
import sys
import urllib.request
import shutil

WHISPER_MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"
PUNCTUATION_MODEL_URL = "https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large/resolve/main/model.onnx"
MODELS_DIR = "models"

os.makedirs(MODELS_DIR, exist_ok=True)

models = [
    (WHISPER_MODEL_URL, os.path.join(MODELS_DIR, "ggml-large-v3-turbo.bin")),
    (PUNCTUATION_MODEL_URL, os.path.join(MODELS_DIR, "punctuation.onnx")),
]

def download(url, dest):
    print(f"Downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as response, open(dest, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    print(f"Downloaded: {dest}")

if __name__ == "__main__":
    for url, dest in models:
        if not os.path.exists(dest):
            download(url, dest)
        else:
            print(f"Already exists: {dest}")
    print("All models downloaded.")
