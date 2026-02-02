import streamlit as st
import tempfile
import os
from logic import transcribe_audio, extract_requirements

# Page Config
st.set_page_config(
    page_title="Voice Requirement AI",
    page_icon="🤖",
    layout="centered"
)

# Styling
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
        color: #f1f5f9;
    }
    h1 {
        background: linear-gradient(to right, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #6366f1;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("Voice Requirement AI")
st.write("Turn your voice into structured project requirements instantly.")

# --- Helper ---
@st.cache_data(show_spinner=False)
def process_audio_data(audio_bytes, file_ext):
    """
    Processes audio bytes directly. Cached by Streamlit to avoid re-running 
    unless the audio content changes.
    """
    # Write bytes to a temp file for Whisper/FFmpeg
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    
    try:
        # 1. Transcribe
        transcript = transcribe_audio(tmp_path)
        
        # 2. Extract
        reqs = extract_requirements(transcript)
        
        return transcript, reqs, None # None = No error
        
    except Exception as e:
        return None, None, str(e)
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- UI ---
st.info("💡 Usage: Simply record or upload. The AI analyzes it automatically.")

tab_record, tab_upload = st.tabs(["🎙️ Record", "📂 Upload"])

audio_bytes = None
file_ext = ".wav" # Default

with tab_record:
    audio_value = st.audio_input("Record your requirements")
    if audio_value:
        audio_value.seek(0)
        audio_bytes = audio_value.read()
        file_ext = ".wav"

with tab_upload:
    uploaded_file = st.file_uploader("Upload Audio/Video", type=["mp3", "wav", "m4a", "mp4", "webm"])
    if uploaded_file:
        uploaded_file.seek(0)
        audio_bytes = uploaded_file.read()
        file_ext = os.path.splitext(uploaded_file.name)[1] or ".tmp"

# --- Execution ---
if audio_bytes:
    # Processing status container
    with st.status("Processing Audio...", expanded=True) as status:
        
        status.write("🎧 Transcribing...")
        # Call cached function
        transcript, requirements, error = process_audio_data(audio_bytes, file_ext)
        
        if error:
            status.update(label="Error Occurred", state="error", expanded=True)
            st.error(f"Processing failed: {error}")
        else:
            status.write("✅ Transcription complete.")
            status.write("🧠 Requirements extracted.")
            status.update(label="Analysis Complete", state="complete", expanded=False)
            
            # Display Results
            st.divider()
            
            st.subheader("📝 Transcript")
            with st.expander("Show full transcript"):
                st.write(transcript)
            
            st.subheader("📋 Requirements")
            if requirements:
                for req in requirements:
                    st.success(req, icon="✅")
            else:
                st.warning("No specific requirements found in the transcript.")
