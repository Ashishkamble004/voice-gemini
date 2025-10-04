import os

from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech as cloud_speech_types

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT","general-ak")

def transcribe_streaming_v2(audio_bytes: bytes) -> str:
    """
    Transcribes audio from bytes using Google Cloud Speech-to-Text API with proper chunking.
    Args:
        audio_bytes: bytes object containing audio data
    Returns:
        str: The transcription result.
    """
    client = SpeechClient(client_options={"api_endpoint": "asia-southeast1-speech.googleapis.com"})
    MAX_CHUNK_SIZE = 25600
    stream = [
        audio_bytes[start : start + MAX_CHUNK_SIZE]
        for start in range(0, len(audio_bytes), MAX_CHUNK_SIZE)
    ]
    audio_requests = (
        cloud_speech_types.StreamingRecognizeRequest(audio=audio) for audio in stream
    )
    recognition_config = cloud_speech_types.RecognitionConfig(
        auto_decoding_config=cloud_speech_types.AutoDetectDecodingConfig(),
        language_codes=["en-US"],
        model="chirp_3",
    )
    streaming_config = cloud_speech_types.StreamingRecognitionConfig(
        config=recognition_config
    )
    config_request = cloud_speech_types.StreamingRecognizeRequest(
        recognizer=f"projects/{PROJECT_ID}/locations/asia-southeast1/recognizers/_",
        streaming_config=streaming_config,
    )
    def requests(config_request, audio_requests):
        yield config_request
        yield from audio_requests
    responses_iterator = client.streaming_recognize(
        requests=requests(config_request, audio_requests)
    )
    transcript = ""
    for response in responses_iterator:
        for result in response.results:
            transcript += result.alternatives[0].transcript
    return transcript