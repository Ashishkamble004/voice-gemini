import streamlit as st
import tempfile
from stt import transcribe_streaming_v2
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av

st.title("Real-Time Speech-to-Text")

st.write("Record or upload an audio file and get the transcript using Google Cloud Speech-to-Text.")

# Real-time microphone streaming
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = []
        self.transcript = ""

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Collect audio frames
        pcm = frame.to_ndarray().tobytes()
        self.audio_frames.append(pcm)
        return frame

    def get_transcript(self):
        # Save audio to temp file and transcribe
        if self.audio_frames:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(b"".join(self.audio_frames))
                tmp_file_path = tmp_file.name
            self.transcript = transcribe_streaming_v2(tmp_file_path)
            return self.transcript
        return ""

webrtc_ctx = webrtc_streamer(key="speech-to-text", audio_receiver_size=256, 
                             media_stream_constraints={"audio": True, "video": False},
                             audio_processor_factory=AudioProcessor)

if webrtc_ctx and webrtc_ctx.state.playing:
    st.write("Recording... Speak into your microphone.")
    if st.button("Transcribe Audio"):
        transcript = webrtc_ctx.audio_processor.get_transcript()
        st.write(transcript)

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
