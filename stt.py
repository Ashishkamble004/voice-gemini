import os
from typing import Generator, Union
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "general-ak")
LOCATION = "asia-southeast1"
RECOGNIZER_ID = "_"  # Using the default recognizer

def transcribe_streaming(
    stream: Union[str, Generator[bytes, None, None]],
) -> Generator[str, None, None]:
    """
    Transcribes a streaming audio source from a file or a generator.

    Args:
        stream: The source of the audio stream. Can be a file path (str) or
                a generator that yields audio chunks (bytes).

    Yields:
        The transcribed text chunks as they are recognized.
    """
    client = SpeechClient(
        client_options={"api_endpoint": f"{LOCATION}-speech.googleapis.com"}
    )

    # -- 1. Set up the recognition config --
    recognition_config = cloud_speech.RecognitionConfig(
        explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
            encoding="LINEAR16",
            sample_rate_hertz=48000,
            audio_channel_count=1,
        ),
        language_codes=["en-US"],
        model="chirp_3",
    )
    streaming_config = cloud_speech.StreamingRecognitionConfig(
        config=recognition_config
    )
    recognizer_path = f"projects/{PROJECT_ID}/locations/{LOCATION}/recognizers/{RECOGNIZER_ID}"
    
    config_request = cloud_speech.StreamingRecognizeRequest(
        recognizer=recognizer_path, streaming_config=streaming_config
    )

    # -- 2. Create the audio stream generator --
    def audio_stream_generator(audio_source):
        # If the source is a file path, read and chunk it
        if isinstance(audio_source, str):
            with open(audio_source, "rb") as f:
                content = f.read()
            for i in range(0, len(content), 25600):
                yield cloud_speech.StreamingRecognizeRequest(audio=content[i : i + 25600])
        # If the source is already a generator, wrap the chunks
        else:
            for chunk in audio_source:
                yield cloud_speech.StreamingRecognizeRequest(audio=chunk)

    # -- 3. Define the full request generator --
    def request_generator(config, audio_stream):
        yield config
        yield from audio_stream

    # -- 4. Perform the transcription --
    requests = request_generator(config_request, audio_stream_generator(stream))
    responses = client.streaming_recognize(requests=requests)

    for response in responses:
        for result in response.results:
            if result.alternatives:
                yield result.alternatives[0].transcript