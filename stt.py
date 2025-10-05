import os
from google import genai
from google.genai import types

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