import streamlit as st
from stt import transcribe_streaming
from st_audiorec import st_audiorec
import os

st.title("Real-Time Speech-to-Text")
st.write("Click the microphone to start and stop recording, then see the transcription below.")

# -- 1. Audio Recorder --
# Use the st_audiorec component to record audio
wav_audio_data = st_audiorec()

# -- 2. Transcription --
if wav_audio_data is not None:
    st.audio(wav_audio_data, format='audio/wav')
    
    # Save the recorded audio to a temporary file
    temp_file_path = "temp_audio.wav"
    with open(temp_file_path, "wb") as f:
        f.write(wav_audio_data)

    # --- Corrected Transcription and UI Update ---
    transcript_container = st.empty()
    transcript_container.info("Transcribing...")

    transcript_generator = transcribe_streaming(temp_file_path)
    
    full_transcript = ""
    for chunk in transcript_generator:
        full_transcript += chunk
        transcript_container.text_area("Transcription", full_transcript, height=200)

    # Clean up the temporary file
    os.remove(temp_file_path)

# -- 3. File Uploader (as a fallback) --
st.header("Or, upload an audio file")
uploaded_file = st.file_uploader("Choose a WAV file", type="wav")

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    # Save the uploaded file to a temporary file
    temp_upload_path = uploaded_file.name
    with open(temp_upload_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # --- Corrected Transcription and UI Update ---
    file_transcript_container = st.empty()
    file_transcript_container.info("Transcribing file...")

    file_transcript_generator = transcribe_streaming(temp_upload_path)
    
    full_transcript = ""
    for chunk in file_transcript_generator:
        full_transcript += chunk
        file_transcript_container.text_area("File Transcription", full_transcript, height=200)
    
    # Clean up the temporary file
    os.remove(temp_upload_path)
