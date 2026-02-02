# Voice Requirement AI

A clean, AI-powered tool to extract project requirements from voice recordings.

## Features
- **Local AI**: Uses OpenAI Whisper for transcription and Llama 3.2 (via Ollama) for reasoning.
- **Privacy**: All processing happens locally on your machine.
- **Formats**: Supports microphone recording and file uploads (MP3, WAV, MP4, etc.).

## Troubleshooting
- **FFmpeg Error**: If you see "Transcription failed" or "WinError 2", it means FFmpeg is missing.
    - **Fix**: Run the included setup script in PowerShell:
      ```powershell
      powershell -ExecutionPolicy Bypass -File setup_ffmpeg.ps1
      ```
    - This will automatically download and configure FFmpeg in the project folder.
- **Ollama connection**: Make sure `ollama serve` is running in a terminal window.

## Setup

1.  **Prerequisites**:
    - Python 3.10+
    - [Ollama](https://ollama.com) installed.
    - `ffmpeg` installed (for Whisper).

2.  **Prepare AI Model**:
    ```bash
    ollama pull llama3.2
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Run

```bash
streamlit run app/main.py
```

## Usage
1.  Open the app in your browser (default `http://localhost:8501`).
2.  **Record** your voice or **Upload** a file.
3.  Click **Analyze Requirements**.
4.  View the extracted requirements list.
