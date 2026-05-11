from fastapi import FastAPI, UploadFile, File
import shutil
import os
import subprocess
from faster_whisper import WhisperModel

app = FastAPI()

UPLOAD_FOLDER = "../uploads"
AUDIO_FOLDER = "../audio"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Load Whisper model
model = WhisperModel("base", device="cpu", compute_type="int8")

@app.get("/")
def home():
    return {"message": "Backend running successfully"}

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):

    # Save uploaded video
    video_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Create audio filename
    audio_filename = os.path.splitext(file.filename)[0] + ".mp3"
    audio_path = os.path.join(AUDIO_FOLDER, audio_filename)

    # Extract audio using FFmpeg
    command = [
        "ffmpeg",
        "-i", video_path,
        "-q:a", "0",
        "-map", "a",
        audio_path,
        "-y"
    ]

    subprocess.run(command)

    # Transcribe audio
    segments, info = model.transcribe(audio_path)

    transcript = ""

    for segment in segments:
        transcript += segment.text + " "

    return {
        "video": file.filename,
        "audio": audio_filename,
        "transcript": transcript,
        "message": "Transcription completed successfully"
    }