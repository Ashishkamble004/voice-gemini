import os
import google.generativeai as genai

# --- Authentication ---
# Make sure to set the GOOGLE_API_KEY environment variable.
# You can get a key from https://aistudio.google.com/app/apikey
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    # This is a fallback for Streamlit Community Cloud secrets
    import streamlit as st
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    except Exception:
        print("Could not find GOOGLE_API_KEY in environment variables or Streamlit secrets.")
        # Handle the error gracefully in the app if needed
        pass

def transcribe_with_gemini(audio_file_path: str) -> str:
    """
    Transcribes an audio file using the Gemini 1.5 Flash model.

    Args:
        audio_file_path: The path to the audio file to transcribe.

    Returns:
        The transcribed text, or an error message if transcription fails.
    """
    print(f"Uploading file: {audio_file_path}")
    
    # Upload the audio file to the Gemini API
    audio_file = genai.upload_file(path=audio_file_path)
    print(f"Completed upload: {audio_file.name}")

    # Initialize the Gemini 1.5 Flash model
    model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")

    # The prompt is simple: just ask for the transcription.
    prompt = "Transcribe this audio."

    try:
        # Generate the content
        response = model.generate_content(
            [prompt, audio_file],
            request_options={"timeout": 600} # Set a 10-minute timeout
        )
        
        # Clean up the uploaded file
        genai.delete_file(audio_file.name)
        
        return response.text
    except Exception as e:
        # Clean up the file in case of an error
        genai.delete_file(audio_file.name)
        print(f"An error occurred: {e}")
        return f"Error during transcription: {e}"