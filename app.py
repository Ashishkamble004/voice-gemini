import streamlit as st
from stt import transcribe_with_vertex, query_rag_with_vertex
from st_audiorec import st_audiorec
import os

st.title("Cymbal Bank - Voice Query")

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

# --- UI Flow ---

# Initialize session state to control the recording widget visibility
if 'show_recorder' not in st.session_state:
    st.session_state.show_recorder = False

# Show a button to start the recording process
if not st.session_state.show_recorder:
    if st.button("Record Your Question"):
        st.session_state.show_recorder = True
        st.rerun()
else:
    st.write("Click the microphone to start/stop recording.")
    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        st.audio(wav_audio_data, format='audio/wav')
        
        temp_file_path = "temp_audio.wav"
        with open(temp_file_path, "wb") as f:
            f.write(wav_audio_data)

        handle_audio(temp_file_path)
        os.remove(temp_file_path)

        # Hide the recorder and show the button again for the next query
        st.session_state.show_recorder = False
        st.rerun()
