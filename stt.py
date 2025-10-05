import os
import google.generativeai as genai
from google.cloud import aiplatform
from google.generativeai import types

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
    """Transcribes an audio file using the Gemini 1.5 Flash model."""
    print(f"Uploading file for transcription: {audio_file_path}")
    audio_file = genai.upload_file(path=audio_file_path)
    print(f"Completed upload: {audio_file.name}")

    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
    prompt = "Transcribe this audio."

    try:
        response = model.generate_content([prompt, audio_file], request_options={"timeout": 600})
        genai.delete_file(audio_file.name)
        return response.text
    except Exception as e:
        genai.delete_file(audio_file.name)
        print(f"An error occurred during transcription: {e}")
        return f"Error during transcription: {e}"

def query_rag_with_vertex(prompt: str) -> str:
    """Sends a prompt to a RAG-enabled model in Vertex AI."""
    print(f"Querying RAG with prompt: {prompt}")

    # --- Authentication for RAG (Vertex AI Service Account) ---
    # This assumes you are authenticated via gcloud auth application-default login
    # or running in a GCP environment with a service account.
    aiplatform.init(project="general-ak", location="global")
    
    client = aiplatform.gapic.GenerativeServiceClient()

    model = "gemini-2.5-flash"
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)]
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

    try:
        response = client.generate_content(
            model=f"projects/general-ak/locations/global/endpoints/{model}",
            contents=contents,
            tools=tools,
        )
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        print(f"An error occurred during RAG query: {e}")
        return f"Error querying RAG system: {e}"