import streamlit as st
from stt import transcribe_with_gemini, query_rag_with_vertex
from st_audiorec import st_audiorec
import os

st.title("Cymbal Bank - Voice Query")
st.write("Record your question to query our system.")

def handle_audio(file_path: str):
    """A helper function to handle the audio processing pipeline."""
    
    # -- Step 1: Transcribe the audio --
    with st.spinner('Transcribing your question...'):
        transcript = transcribe_with_gemini(file_path)
    
    if "Error" in transcript:
        st.error(transcript)
        return

    st.info("Your transcribed question:")
    st.text_area("Question", transcript, height=100)

    # -- Step 2: Query the RAG system and stream the response --
    st.success("Answer from our system:")
    answer_container = st.empty()
    
    with st.spinner('Searching for an answer...'):
        rag_response_stream = query_rag_with_vertex(transcript)
        
        full_response = ""
        for chunk in rag_response_stream:
            full_response += chunk
            answer_container.markdown(full_response)
    
    if "Error" in full_response:
        st.error(full_response)
        return

# -- Audio Recorder --
wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    st.audio(wav_audio_data, format='audio/wav')
    
    temp_file_path = "temp_audio.wav"
    with open(temp_file_path, "wb") as f:
        f.write(wav_audio_data)

    handle_audio(temp_file_path)
    os.remove(temp_file_path)
