import shutil
import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from stt import transcribe_streaming_v2

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/", StaticFiles(directory="frontend/build", html=True), name="frontend")

@app.get("/")
async def read_index():
    return FileResponse('frontend/build/index.html')

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Endpoint to transcribe an audio file (legacy, not real-time)."""
    with open(file.filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    transcription = transcribe_streaming_v2(file.filename)
    return {"transcription": transcription}

# --- WebSocket endpoint for real-time streaming ---
import asyncio
from typing import List
from google.cloud.speech_v2.types import cloud_speech as cloud_speech_types

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()
    audio_chunks: List[bytes] = []
    try:
        while True:
            data = await websocket.receive_bytes()
            audio_chunks.append(data)
            # Optionally, send partial transcript back
            if len(audio_chunks) >= 5:  # Example: process every 5 chunks
                transcript = transcribe_streaming_v2(audio_chunks)
                await websocket.send_text(transcript)
                audio_chunks.clear()
    except Exception as e:
        await websocket.close()
        print(f"WebSocket closed: {e}")


import os
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
