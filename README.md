# Voice Gemini - Real-Time Speech-to-Text

A Streamlit application for real-time speech-to-text transcription using Google Cloud Speech-to-Text API.

## Features

- Real-time microphone streaming transcription
- Audio file upload support (WAV, MP3, OGG, WebM)
- Google Cloud Speech-to-Text API integration with Chirp 3 model
- WebRTC-based audio capture

## Environment Setup

### Environment Variables

The application requires the following environment variable:

- `GOOGLE_CLOUD_PROJECT`: Your Google Cloud Project ID (default: `general-ak`)

### For Local Development

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and update the values if needed:
   ```
   GOOGLE_CLOUD_PROJECT=your-project-id
   ```

3. The application will automatically load environment variables from the `.env` file.

### For Cloud Run Deployment

Use the `env.yaml` file to set environment variables during deployment:

```bash
gcloud run deploy voice-gemini \
  --source . \
  --env-vars-file env.yaml \
  --region asia-southeast1 \
  --allow-unauthenticated
```

Or set environment variables directly:

```bash
gcloud run deploy voice-gemini \
  --source . \
  --set-env-vars GOOGLE_CLOUD_PROJECT=general-ak \
  --region asia-southeast1 \
  --allow-unauthenticated
```

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running Locally

```bash
streamlit run app.py
```

Or with Docker:

```bash
docker build -t voice-gemini .
docker run -p 8080:8080 --env-file .env voice-gemini
```

## Google Cloud Authentication

Ensure you have Google Cloud credentials configured:

```bash
gcloud auth application-default login
```

Or set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to your service account key file.

## License

MIT
