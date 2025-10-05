import os
import google.generativeai as genai
from google.cloud import aiplatform
from google.generativeai import types
from typing import Generator

# --- Authentication for Transcription (Gemini API Key) ---
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    import streamlit as st
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    except Exception:
        print("Could not find GOOGLE_API_KEY for transcription.")
        pass

def transcribe_with_gemini(audio_file_path: str) -> str:
    """Transcribes an audio file using the Gemini 2.5 Flash model."""
    print(f"Uploading file for transcription: {audio_file_path}")
    audio_file = genai.upload_file(path=audio_file_path)
    print(f"Completed upload: {audio_file.name}")

    model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")
    prompt = "Transcribe this audio."

    try:
        response = model.generate_content([prompt, audio_file], request_options={"timeout": 600})
        genai.delete_file(audio_file.name)
        return response.text
    except Exception as e:
        genai.delete_file(audio_file.name)
        print(f"An error occurred during transcription: {e}")
        return f"Error during transcription: {e}"

def query_rag_with_vertex(prompt: str) -> Generator[str, None, None]:
    """Sends a prompt to a RAG-enabled model in Vertex AI and streams the response."""
    print(f"Querying RAG with prompt: {prompt}")

    # --- RAG Client Initialization ---
    client = genai.Client(
        vertexai=True,
        project="general-ak",
        location="global",
    )

    # --- Fixed System Prompt ---
    system_prompt = (
        "You are a helpful assistant for Cymbal Bank. "
        "Please answer the user's question based on the documents provided. "
        "If the information is not in the documents, say that you cannot find the answer."
    )

    # The model is now initialized with the fixed system prompt
    model = genai.GenerativeModel(
        "gemini-2.5-flash",
        system_instruction=system_prompt
    )
    
    tools = [
        types.Tool(
            retrieval=types.Retrieval(
                vertex_rag_store=types.VertexRagStore(
                    rag_resources=[
                        types.VertexRagStoreRagResource(
                            rag_corpus="projects/general-ak/locations/us-east4/ragCorpora/6917529027641081856"
                        )
                    ],
                )
            )
        )
    ]

    try:
        # Stream the response from the model
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