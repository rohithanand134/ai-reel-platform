from fastapi import FastAPI, UploadFile, File
import shutil
import os
import subprocess
from faster_whisper import WhisperModel

app = FastAPI()

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Absolute folders
UPLOAD_FOLDER = os.path.abspath(os.path.join(BASE_DIR, "../uploads"))
AUDIO_FOLDER = os.path.abspath(os.path.join(BASE_DIR, "../audio"))
SUBTITLE_FOLDER = os.path.join(BASE_DIR, "subtitles")
PROCESSED_FOLDER = os.path.abspath(os.path.join(BASE_DIR, "../processed"))

# Create folders if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(SUBTITLE_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load Whisper model
model = WhisperModel("base", device="cpu", compute_type="int8")


@app.get("/")
def home():
    return {"message": "Backend running successfully"}


def format_time(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds - int(seconds)) * 1000)

    return f"{hrs:02}:{mins:02}:{secs:02},{millisecs:03}"


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):

    # Save uploaded video
    video_path = os.path.join(UPLOAD_FOLDER, file.filename)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print("Video path:", video_path)

    # Audio filename
    audio_filename = os.path.splitext(file.filename)[0] + ".mp3"
    audio_path = os.path.join(AUDIO_FOLDER, audio_filename)

    # Extract audio
    audio_command = [
        "ffmpeg",
        "-i", video_path,
        "-q:a", "0",
        "-map", "a",
        audio_path,
        "-y"
    ]

    subprocess.run(audio_command)

    # Transcribe
    segments, info = model.transcribe(audio_path)

    transcript = ""

    # Subtitle filename
    subtitle_filename = os.path.splitext(file.filename)[0] + ".srt"
    subtitle_path = os.path.join(SUBTITLE_FOLDER, subtitle_filename)

    # Generate subtitles
    with open(subtitle_path, "w", encoding="utf-8") as srt_file:

        for index, segment in enumerate(segments):

            start_time = format_time(segment.start)
            end_time = format_time(segment.end)

            text = segment.text.strip()

            transcript += text + " "

            srt_file.write(f"{index + 1}\n")
            srt_file.write(f"{start_time} --> {end_time}\n")
            srt_file.write(f"{text}\n\n")

    print("Subtitle path:", subtitle_path)
    print("Subtitle exists:", os.path.exists(subtitle_path))

    # Processed video output
    output_video_filename = (
        os.path.splitext(file.filename)[0] + "_subtitled.mp4"
    )

    output_video_path = os.path.join(
        PROCESSED_FOLDER,
        output_video_filename
    )

    print("Output path:", output_video_path)

    # Windows-safe subtitle path
    ffmpeg_subtitle_path = subtitle_path.replace("\\", "/")
    ffmpeg_subtitle_path = ffmpeg_subtitle_path.replace(":", "\\:")

    # Burn subtitles
    subtitle_command = [
        "ffmpeg",
        "-i", video_path,
        "-vf",
        f"subtitles='{ffmpeg_subtitle_path}'",
        output_video_path,
        "-y"
    ]

    print("Running subtitle command...")
    print(subtitle_command)

    subprocess.run(subtitle_command)

    return {
        "video": file.filename,
        "audio": audio_filename,
        "subtitle": subtitle_filename,
        "processed_video": output_video_filename,
        "transcript": transcript,
        "message": "Video processed successfully"
    }