import os
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Tool
from google.genai import types

def initialize_vertexai():
    """Initializes the Vertex AI SDK."""
    # This will use the project and location from your gcloud config
    # or the environment your app is running in.
    vertexai.init(project="general-ak", location="us-central1")

def transcribe_with_vertex(audio_file_path: str) -> str:
    """Transcribes an audio file using a Gemini model in Vertex AI."""
    initialize_vertexai()
    
    print(f"Transcribing file: {audio_file_path}")
    
    # Read the audio file bytes
    with open(audio_file_path, "rb") as f:
        audio_bytes = f.read()

    # Prepare the audio part for the model
    audio_part = Part.from_data(
        data=audio_bytes,
        mime_type="audio/wav"
    )

    model = GenerativeModel(model_name="gemini-2.5-flash")
    prompt = "Transcribe this audio."

    try:
        response = model.generate_content([prompt, audio_part])
        return response.text
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        return f"Error during transcription: {e}"

def query_rag_with_vertex(prompt: str):
    """Sends a prompt to a RAG-enabled model in Vertex AI and streams the response."""
    initialize_vertexai()

    print(f"Querying RAG with prompt: {prompt}")

    system_prompt = (
        "You are a helpful assistant for Cymbal Bank. "
        "Please answer the user's question based on the documents provided. "
        "If the information is not in the documents, say that you cannot find the answer."
    )

    model = GenerativeModel(
        "gemini-2.5-flash",
        system_instruction=system_prompt
    )
    
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

    try:
        response_stream = model.generate_content(
            prompt,
            tools=tools,
            stream=True,
        )
        
        for chunk in response_stream:
            if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                yield chunk.text
    except Exception as e:
        print(f"An error occurred during RAG query: {e}")
        yield f"Error querying RAG system: {e}"