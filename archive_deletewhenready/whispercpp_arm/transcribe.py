import os
import sys
import subprocess
from pathlib import Path
import onnxruntime as ort
import numpy as np

# CONFIGURATION
WHISPER_CPP_PATH = "./whisper.cpp"  # Path to whisper.cpp binary
MODEL_PATH = "./models/ggml-large.bin"  # Path to large model
PUNCTUATION_MODEL_PATH = "./models/punctuation.onnx"  # Path to ONNX punctuation model

# Offensive words list (expand as needed)
OFFENSIVE_WORDS = ["damn", "hell", "shit", "fuck"]

# Helper: run whisper.cpp on a file
def run_whisper_cpp(audio_file):
    out_txt = f"output/{audio_file.stem}.txt"
    cmd = [WHISPER_CPP_PATH, "-m", MODEL_PATH, "-f", str(audio_file), "-otxt", "-of", out_txt]
    print(f"Transcribing {audio_file}...")
    subprocess.run(cmd, check=True)
    return out_txt

# Helper: restore punctuation using ONNX model
# (Assumes model expects a string input and returns a string output)
def restore_punctuation(text_file):
    with open(text_file, "r", encoding="utf-8") as fin:
        text = fin.read()
    # Placeholder: actual ONNX punctuation restoration logic
    # Example: use Hugging Face model with ONNXRuntime
    # session = ort.InferenceSession(PUNCTUATION_MODEL_PATH)
    # result = session.run(None, {"input": np.array([text])})
    # punct_text = result[0][0]
    punct_text = text  # TODO: Replace with actual punctuation restoration
    punct_file = text_file.replace(".txt", "_punct.txt")
    with open(punct_file, "w", encoding="utf-8") as fout:
        fout.write(punct_text)
    return punct_file

# Helper: apply AI guardrails (mute copyright, filter swearing)
def apply_guardrails(text_file):
    guard_file = text_file.replace(".txt", "_guard.txt")
    with open(text_file, "r", encoding="utf-8") as fin, open(guard_file, "w", encoding="utf-8") as fout:
        for line in fin:
            clean_line = line
            for word in OFFENSIVE_WORDS:
                clean_line = clean_line.replace(word, "[muted]")
            # TODO: Add copyright mute logic if needed
            fout.write(clean_line)
    return guard_file

# Main orchestration
def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py --input <file_or_folder>")
        sys.exit(1)
    input_path = Path(sys.argv[2]) if sys.argv[1] == "--input" else Path(sys.argv[1])
    files = []
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = list(input_path.glob("*.wav")) + list(input_path.glob("*.mp3")) + list(input_path.glob("*.m4a"))
    else:
        print("Invalid input path.")
        sys.exit(1)
    os.makedirs("output", exist_ok=True)
    for audio_file in files:
        txt_file = run_whisper_cpp(audio_file)
        punct_file = restore_punctuation(txt_file)
        guard_file = apply_guardrails(punct_file)
        print(f"Output: {guard_file}")

if __name__ == "__main__":
    main()
