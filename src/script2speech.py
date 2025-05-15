import torch
from TTS.api import TTS
import re
print(torch.cuda.is_available())  # Should output True
print(torch.cuda.get_device_name(0))
with open('output.txt', 'r') as file:
    text = file.readlines()
    lines = [re.sub(r'^(Host|Guest):\s*', '', line).strip() 
             for line in text if line.strip()]
    
# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

counter = 0
for text in lines:
    VOICE_FILE = 'male.mp3' if counter % 2 == 0 else 'female.mp3'
    counter += 1
    tts.tts_to_file(text=text, speaker_wav=f"C:\\Doc2Podcast\\{VOICE_FILE}", language="en", file_path=f"output_new_{counter}.wav")
    