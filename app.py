import streamlit as st
from stt import transcribe_streaming
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import queue
import threading

st.title("Real-Time Speech-to-Text")
st.write("Click 'Start' to begin recording from your microphone and see the live transcription.")

# Use a session state to track the recording status and transcript
if "recording" not in st.session_state:
    st.session_state.recording = False
    st.session_state.transcript = ""

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_queue = queue.Queue()

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # The audio frames are in pcm_s16 format, which is what we need.
        self.audio_queue.put(frame.to_ndarray().tobytes())
        return frame

def stream_transcription(audio_queue: queue.Queue, transcript_container):
    """Continuously transcribes audio chunks from the queue."""
    
    def audio_chunk_generator():
        """Yields audio chunks from the queue."""
        while st.session_state.recording:
            try:
                yield audio_queue.get(timeout=1)
            except queue.Empty:
                break # Exit if no audio for 1s

    transcript_chunks = transcribe_streaming(audio_chunk_generator())
    
    full_transcript = ""
    for chunk in transcript_chunks:
        full_transcript += chunk
        transcript_container.markdown(full_transcript)
    
    st.session_state.transcript = full_transcript

# -- Main App UI --
webrtc_ctx = webrtc_streamer(
    key="speech-to-text",
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    sendback_audio=False,
)

transcript_container = st.empty()
transcript_container.markdown(st.session_state.transcript)

if not webrtc_ctx.state.playing and not st.session_state.recording:
    if st.button("Start Recording"):
        st.session_state.recording = True
        st.rerun()

elif webrtc_ctx.state.playing and st.session_state.recording:
    st.write("🔴 Recording... Speak into your microphone.")
    
    # Start the transcription thread
    if "transcription_thread" not in st.session_state:
        thread = threading.Thread(
            target=stream_transcription,
            args=(webrtc_ctx.audio_processor.audio_queue, transcript_container),
        )
        st.session_state.transcription_thread = thread
        thread.start()

    if st.button("Stop Recording"):
        st.session_state.recording = False
        st.rerun()

# -- File Uploader --
st.header("Or, upload an audio file")
uploaded_file = st.file_uploader("Choose a WAV file", type="wav")

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    with st.spinner('Transcribing file...'):
        # Save the uploaded file to a temporary file to get a valid path
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # The `transcribe_streaming` function now handles file paths directly
        file_transcript_generator = transcribe_streaming(uploaded_file.name)
        full_transcript = "".join(list(file_transcript_generator))
        st.text_area("File Transcription", full_transcript, height=200)
