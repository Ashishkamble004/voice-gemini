import React, { useState, useRef } from 'react';

function App() {
  const [file, setFile] = useState(null);
  const [transcription, setTranscription] = useState('');
  const [error, setError] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [liveTranscript, setLiveTranscript] = useState('');
  const wsRef = useRef(null);
  const mediaRecorderRef = useRef(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      const apiUrl = process.env.REACT_APP_API_URL;
      const response = await fetch(`${apiUrl}/transcribe`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      setTranscription(data.transcription);
      setError('');
    } catch (error) {
      console.error('Error uploading file:', error);
      setError('Error uploading file. Please try again.');
    }
  };

  // --- Real-time streaming logic ---
  const startStreaming = async () => {
    setIsStreaming(true);
    setLiveTranscript('');
    const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8080/ws/transcribe';
    wsRef.current = new WebSocket(wsUrl);
    wsRef.current.onmessage = (event) => {
      setLiveTranscript((prev) => prev + event.data);
    };
    wsRef.current.onerror = () => {
      setError('WebSocket error');
      setIsStreaming(false);
    };
    wsRef.current.onclose = () => {
      setIsStreaming(false);
    };
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorderRef.current = new window.MediaRecorder(stream, { mimeType: 'audio/webm' });
    mediaRecorderRef.current.ondataavailable = (e) => {
      if (wsRef.current && wsRef.current.readyState === 1) {
        wsRef.current.send(e.data);
      }
    };
    mediaRecorderRef.current.start(250); // send every 250ms
  };

  const stopStreaming = () => {
    setIsStreaming(false);
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
    }
    if (wsRef.current) {
      wsRef.current.close();
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Speech-to-Text</h1>
        <input type="file" onChange={handleFileChange} />
        <button onClick={handleUpload}>Transcribe</button>
        <hr />
        <h2>Real-time Streaming</h2>
        {!isStreaming ? (
          <button onClick={startStreaming}>Start Microphone</button>
        ) : (
          <button onClick={stopStreaming}>Stop</button>
        )}
        <div>
          <h3>Live Transcript:</h3>
          <p>{liveTranscript}</p>
        </div>
        {transcription && (
          <div>
            <h2>Transcription:</h2>
            <p>{transcription}</p>
          </div>
        )}
        {error && <p style={{ color: 'red' }}>{error}</p>}
      </header>
    </div>
  );
}

export default App;