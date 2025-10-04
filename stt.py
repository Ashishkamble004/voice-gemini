import os
from typing import Union

from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech as cloud_speech_types

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT","general-ak")

def transcribe_streaming_v2(audio_input: Union[bytes, str]) -> str:
    """
    Transcribes audio from bytes or file path using Google Cloud Speech-to-Text API with proper chunking.
    Args:
        audio_input: bytes object containing audio data or string file path
    Returns:
        str: The transcription result.
    """
    # Handle both file path and bytes
    if isinstance(audio_input, str):
        with open(audio_input, 'rb') as f:
            audio_bytes = f.read()
    else:
        audio_bytes = audio_input
    
    # If audio_bytes is empty, return empty string
    if not audio_bytes:
        return ""
    
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
        model="chirp",
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
    
    try:
        responses_iterator = client.streaming_recognize(
            requests=requests(config_request, audio_requests)
        )
        transcript = ""
        for response in responses_iterator:
            for result in response.results:
                if result.alternatives:
                    transcript += result.alternatives[0].transcript
        return transcript
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""