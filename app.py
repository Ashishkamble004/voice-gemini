import streamlit as st
from stt import transcribe_with_vertex, query_rag_with_vertex
from st_audiorec import st_audiorec
import os

st.title("Cymbal Bank - Voice Query")
st.write("Record your question to query our system.")

def handle_audio(file_path: str):
    """A helper function to handle the audio processing pipeline."""
    
    # -- Step 1: Transcribe the audio using Vertex AI --
    with st.spinner('Transcribing your question...'):
        transcript = transcribe_with_vertex(file_path)
    
    if "Error" in transcript:
        st.error(transcript)
        return

    st.info("Your transcribed question:")
    st.text_area("Question", transcript, height=100)

    # -- Step 2: Query the RAG system with the transcript --
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

# Initialize session state for controlling the recorder
if 'show_recorder' not in st.session_state:
    st.session_state.show_recorder = False

# Button to start recording
if not st.session_state.show_recorder:
    if st.button("Start Recording"):
        st.session_state.show_recorder = True
        st.rerun()
else:
    st.write("Click the microphone to record your question.")
    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        st.audio(wav_audio_data, format='audio/wav')
        
        temp_file_path = "temp_audio.wav"
        with open(temp_file_path, "wb") as f:
            f.write(wav_audio_data)

        handle_audio(temp_file_path)
        os.remove(temp_file_path)

        # Reset for next query
        st.session_state.show_recorder = False
        st.rerun()
