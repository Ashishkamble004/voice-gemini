import os
from google import genai
from google.genai import types
from google.cloud import texttospeech

def initialize_vertexai():
    """Initializes the Google GenAI client with Vertex AI."""
    return genai.Client(
        vertexai=True,
        project="general-ak",
        location="global",
    )

def transcribe_with_vertex(audio_file_path: str) -> str:
    """Transcribes an audio file using Gemini 2.5 Flash."""
    client = initialize_vertexai()
    
    print(f"Transcribing file: {audio_file_path}")
    
    # Read the audio file bytes
    with open(audio_file_path, "rb") as f:
        audio_bytes = f.read()

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                types.Part.from_text(text="Transcribe this audio.")
            ]
        ),
    ]

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )
        return response.text
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        return f"Error during transcription: {e}"

def query_rag_with_vertex(prompt: str):
    """Sends a prompt to Gemini 2.5 Flash  with RAG and streams the response."""
    client = initialize_vertexai()

    print(f"Querying RAG with prompt: {prompt}")

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt)
            ]
        ),
    ]
    tools = [
        types.Tool(
            retrieval=types.Retrieval(
                vertex_rag_store=types.VertexRagStore(
                    rag_resources=[
                        types.VertexRagStoreRagResource(
                            rag_corpus="projects/general-ak/locations/us-east4/ragCorpora/2305843009213693952"
                        )
                    ],
                )
            )
        )
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=0.6,
        top_p=0.95,
        max_output_tokens=1000,
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_ONLY_HIGH"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_ONLY_HIGH"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_ONLY_HIGH"
            ),
        ],
        tools=tools,
        thinking_config=types.ThinkingConfig(
            thinking_budget=0,
        ),
    )

    try:
        for chunk in client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=contents,
            config=generate_content_config,
        ):
            if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                continue
            yield chunk.text
    except Exception as e:
        print(f"An error occurred during RAG query: {e}")
        yield f"Error querying RAG system: {e}"

def text_to_speech(text: str) -> bytes:
    """Converts text to speech using Google Cloud Text-to-Speech."""
    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=text)

    # Configure voice - using Chirp3 HD model for high-quality TTS
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Chirp3-HD-Female",  # Chirp3 HD female voice
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
    )

    # Configure audio
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
    )

    try:
        response = client.synthesize_speech(
            input=input_text, voice=voice, audio_config=audio_config
        )
        return response.audio_content
    except Exception as e:
        print(f"TTS error: {e}")
        return b""

def add_wav_header(audio_bytes, sample_rate=24000):
    """Add WAV header to raw LINEAR16 audio bytes."""
    import struct
    riff = b'RIFF'
    riff_size = len(audio_bytes) + 36
    wave = b'WAVE'
    fmt = b'fmt '
    fmt_size = 16
    audio_format = 1  # PCM
    num_channels = 1
    byte_rate = sample_rate * num_channels * 2  # 16-bit
    block_align = num_channels * 2
    bits_per_sample = 16
    data = b'data'
    data_size = len(audio_bytes)
    header = struct.pack('<4sL4s4sLHHLLHH4sL', riff, riff_size, wave, fmt, fmt_size, audio_format, num_channels, sample_rate, byte_rate, block_align, bits_per_sample, data, data_size)
    return header + audio_bytes

from pydub import AudioSegment
AudioSegment.converter = "/usr/bin/ffmpeg"
import io

def streaming_text_to_speech(text_chunks):
    """Streams text-to-speech audio for a sequence of text chunks."""
    client = texttospeech.TextToSpeechClient()

    # Assuming a standard US English voice, modify as needed
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Chirp3-HD-Female",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
    )

    # Configure audio - Chirp voices work well with MP3
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
    )

    # Collect audio chunks
    combined_audio = AudioSegment.empty()

    try:
        for text in text_chunks:
            if not text.strip():
                continue
            input_text = texttospeech.SynthesisInput(text=text)

            # Perform the text-to-speech request
            response = client.synthesize_speech(
                input=input_text, voice=voice, audio_config=audio_config
            )

            # Append the received audio content using pydub
            audio_segment = AudioSegment.from_mp3(io.BytesIO(response.audio_content))
            combined_audio += audio_segment
        
        # Export the combined audio to an in-memory file
        buffer = io.BytesIO()
        combined_audio.export(buffer, format="mp3")
        return buffer.getvalue()

    except Exception as e:
        return str(e)