import streamlit as st
import tempfile
from stt import transcribe_streaming_v2
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import threading

st.title("Real-Time Speech-to-Text")

st.write("Record or upload an audio file and get the transcript using Google Cloud Speech-to-Text.")

# Real-time microphone streaming
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = b""
        self.transcript = ""
        self.lock = threading.Lock()
        self.streaming = False
        self.partial_transcript = ""

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        pcm = frame.to_ndarray().tobytes()
        with self.lock:
            self.audio_buffer += pcm
        return frame

    def start_streaming_transcription(self, update_callback):
        self.streaming = True
        def audio_chunk_generator():
            idx = 0
            while self.streaming:
                with self.lock:
                    chunk = self.audio_buffer[idx:idx+25600]
                if not chunk:
                    break
                yield chunk
                idx += 25600
        def run():
            transcript = ""
            for partial in transcribe_streaming_v2(audio_chunk_generator(), partials=True):
                transcript += partial
                self.partial_transcript = transcript
                update_callback(transcript)
        threading.Thread(target=run, daemon=True).start()

    def stop_streaming(self):
        self.streaming = False

webrtc_ctx = webrtc_streamer(
    key="speech-to-text",
    audio_receiver_size=256,
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=AudioProcessor,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    }
)

if webrtc_ctx and webrtc_ctx.state.playing:
    st.write("Recording... Speak into your microphone.")
    transcript_placeholder = st.empty()
    def update_transcript(text):
        transcript_placeholder.write(text)
    if st.button("Start Immediate Transcription"):
        webrtc_ctx.audio_processor.start_streaming_transcription(update_transcript)
    if st.button("Stop Transcription"):
        webrtc_ctx.audio_processor.stop_streaming()
    transcript_placeholder.write(webrtc_ctx.audio_processor.partial_transcript)

# Audio file upload (legacy)
uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3", "ogg", "webm"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    st.audio(tmp_file_path)
    with st.spinner("Transcribing..."):
        transcript = transcribe_streaming_v2(tmp_file_path)
    st.success("Transcription complete!")
    st.write(transcript)
