import os
import urllib.request
import shutil

MODELS_DIR = "models"
MODEL_NAME = "ggml-large-v3-turbo.bin"
MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin"

os.makedirs(MODELS_DIR, exist_ok=True)
model_path = os.path.join(MODELS_DIR, MODEL_NAME)

def download(url, dest):
    print(f"Downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as response, open(dest, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    print(f"Downloaded: {dest}")

if __name__ == "__main__":
    if not os.path.exists(model_path):
        download(MODEL_URL, model_path)
    else:
        print(f"Already exists: {model_path}")
