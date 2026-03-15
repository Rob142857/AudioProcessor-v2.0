import os
import subprocess
import sys
from pathlib import Path

# CONFIGURATION
WHISPER_CPP_PATH = "./whisper.cpp"  # Path to whisper.cpp binary
MODEL_PATH = "./models/ggml-large.bin"  # Path to large model
PUNCTUATION_MODEL = "./punctuation.onnx"  # Path to ONNX punctuation model

# Choose file or folder
INPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else None
if not INPUT_PATH:
    print("Please provide a file or folder to transcribe.")
    sys.exit(1)

input_path = Path(INPUT_PATH)
files = []
if input_path.is_file():
    files = [input_path]
elif input_path.is_dir():
    files = list(input_path.glob("*.wav")) + list(input_path.glob("*.mp3")) + list(input_path.glob("*.m4a"))
else:
    print("Invalid input path.")
    sys.exit(1)

os.makedirs("output", exist_ok=True)

def run_whisper_cpp(audio_file):
    out_txt = f"output/{audio_file.stem}.txt"
    cmd = [WHISPER_CPP_PATH, "-m", MODEL_PATH, "-f", str(audio_file), "-otxt", "-of", out_txt]
    print(f"Transcribing {audio_file}...")
    subprocess.run(cmd, check=True)
    return out_txt

def restore_punctuation(text_file):
    # Placeholder: integrate ONNX or pure Python punctuation restoration here
    # For now, just copy the file
    punct_file = text_file.replace(".txt", "_punct.txt")
    with open(text_file, "r", encoding="utf-8") as fin, open(punct_file, "w", encoding="utf-8") as fout:
        for line in fin:
            fout.write(line)  # TODO: Replace with actual punctuation restoration
    return punct_file

def apply_guardrails(text_file):
    # Placeholder: mute copyright, filter swearing
    guard_file = text_file.replace(".txt", "_guard.txt")
    with open(text_file, "r", encoding="utf-8") as fin, open(guard_file, "w", encoding="utf-8") as fout:
        for line in fin:
            # Example: replace offensive words
            clean_line = line.replace("damn", "[muted]").replace("hell", "[muted]")
            fout.write(clean_line)
    return guard_file

for audio_file in files:
    txt_file = run_whisper_cpp(audio_file)
    punct_file = restore_punctuation(txt_file)
    guard_file = apply_guardrails(punct_file)
    print(f"Output: {guard_file}")
