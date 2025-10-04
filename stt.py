import os

from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech as cloud_speech_types

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT","general-ak")

from typing import Generator

def transcribe_streaming_v2(
    audio_chunk_generator,
) -> Generator[str, None, None]:
    """
    Transcribes audio from a generator of audio chunks using Google Cloud Speech-to-Text API.
    Args:
        audio_chunk_generator: generator yielding bytes objects (≤25600 bytes each)
    Returns:
        str: The transcription result.
    """
    client = SpeechClient(client_options={"api_endpoint": "asia-southeast1-speech.googleapis.com"})
    audio_requests = (
        cloud_speech_types.StreamingRecognizeRequest(audio=audio) for audio in audio_chunk_generator
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
    for response in responses_iterator:
        for result in response.results:
            # Yield each partial transcript as soon as it's available
            yield result.alternatives[0].transcript