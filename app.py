import streamlit as st
from stt import transcribe_with_gemini
from st_audiorec import st_audiorec
import os

st.title("Cymbal Bank - Voice Transcription")
st.write("Record your audio or upload a file to have it transcribed by Gemini.")

# -- Audio Recorder --
wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    st.audio(wav_audio_data, format='audio/wav')
    
    temp_file_path = "temp_audio.wav"
    with open(temp_file_path, "wb") as f:
        f.write(wav_audio_data)

    with st.spinner('Transcribing with Gemini...'):
        transcript = transcribe_with_gemini(temp_file_path)
        st.text_area("Transcription", transcript, height=200)

    os.remove(temp_file_path)

# -- File Uploader --
st.header("Or, upload an audio file")
uploaded_file = st.file_uploader("Choose a WAV file", type="wav")

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    temp_upload_path = uploaded_file.name
    with open(temp_upload_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner('Transcribing file with Gemini...'):
        transcript = transcribe_with_gemini(temp_upload_path)
        st.text_area("File Transcription", transcript, height=200)
        
    os.remove(temp_upload_path)
