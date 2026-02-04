import streamlit as st
import shutil
import tempfile
import os

from logic import transcribe_audio, extract_requirements, get_whisper_model

# 1. Page Config
st.set_page_config(
    page_title="Voice Requirement AI",
    page_icon="🤖",
    layout="centered"
)

# --- Session State Management ---
if "transcript" not in st.session_state: st.session_state.transcript = None
if "requirements" not in st.session_state: st.session_state.requirements = None
if "error" not in st.session_state: st.session_state.error = None
if "logs" not in st.session_state: st.session_state.logs = []

def add_log(msg):
    st.session_state.logs.append(msg)
    if len(st.session_state.logs) > 5: st.session_state.logs.pop(0)

# Helper functions
def check_ffmpeg():
    # 1. Check local bin
    # local_bin_ffmpeg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bin", "ffmpeg.exe")
    # if os.path.exists(local_bin_ffmpeg):
        # return True
    # 2. Check system PATH
    return shutil.which("ffmpeg") is not None

# Styling
st.markdown("""
<style>
    .main { background-color: #0f172a; color: #f1f5f9; }
    h1 { background: linear-gradient(to right, #6366f1, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.title("Voice Requirement AI")
st.write("Professional requirement extraction from voice, audio, or text.")

# --- Load Models ---
# @st.cache_resource(show_spinner="Loading AI engine... (this takes a moment)")
# def load_models():
#     add_log("System: Initializing Whisper...")
#     return get_whisper_model()

# whisper_model = load_models()

# --- Analysis Functions ---
def run_audio_analysis(audio_bytes, ext):
    """Explicitly triggers the full audio analysis pipe."""
    add_log("User: Started audio analysis")
    st.session_state.requirements = None
    st.session_state.transcript = None
    st.session_state.error = None
    
    with st.status("🎧 Processing Audio...", expanded=True) as status:
        if not check_ffmpeg():
            st.session_state.error = "FFmpeg not found. Please install FFmpeg to enable audio processing."
            status.update(label="❌ Error: FFmpeg missing", state="error")
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        try:
            status.write("⌛ Transcribing recording...")
            whisper_model = get_whisper_model()             
            if not transcript or not transcript.strip():
                st.session_state.error = "No speech detected. Please check your mic and try again."
                status.update(label="❌ Silence Detected", state="error")
                return
            
            st.session_state.transcript = transcript
            status.write("🧠 Extracting structured requirements...")
            reqs = extract_requirements(transcript)
            st.session_state.requirements = reqs
            status.update(label="✅ Analysis Success", state="complete")
            add_log("System: Analysis complete")
        except Exception as e:
            st.session_state.error = f"Processing Failed: {str(e)}"
            status.update(label="❌ Analysis Failed", state="error")
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)

def run_text_analysis(text):
    """Explicitly triggers text analysis."""
    add_log("User: Started text analysis")
    st.session_state.requirements = None
    st.session_state.transcript = None
    st.session_state.error = None

    with st.status("🧠 Analyzing Text...", expanded=True) as status:
        try:
            reqs = extract_requirements(text)
            st.session_state.transcript = text
            st.session_state.requirements = reqs
            status.update(label="✅ Analysis Complete", state="complete")
        except Exception as e:
            st.session_state.error = str(e)
            status.update(label="❌ Analysis Failed", state="error")

# --- UI Layout ---
tab_record, tab_upload, tab_text = st.tabs(["🎙️ Record", "📂 Upload", "📧 Text"])

with tab_record:
    audio_val = st.audio_input("Record your voice")
    if audio_val:
        st.info("🎙️ Audio detected! Click the button below to start analysis.")
        if st.button("🚀 Analyze Recording", type="primary"):
            run_audio_analysis(audio_val.getvalue(), ".wav")

with tab_upload:
    up_file = st.file_uploader("Upload Audio", type=["mp3", "wav", "m4a", "webm"])
    if up_file:
        if st.button("🚀 Analyze Upload", type="primary"):
            run_audio_analysis(up_file.getvalue(), os.path.splitext(up_file.name)[1])

with tab_text:
    txt_input = st.text_area("Paste Content", height=200, placeholder="Paste emails or notes...")
    if st.button("🚀 Analyze Text", type="primary"):
        if txt_input.strip():
            run_text_analysis(txt_input)
        else:
            st.warning("Please paste some text first.")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ System Status")

    f_ok = check_ffmpeg()
    llm_ok = "GROQ_API_KEY" in st.secrets

    st.metric("Audio Engine (FFmpeg)", "✅ OK" if f_ok else "❌ Missing")
    st.metric("LLM Engine (Groq)", "✅ Ready" if llm_ok else "❌ API Key Missing")

    if not llm_ok:
        st.error("Missing GROQ_API_KEY in Streamlit Secrets")

    st.divider()
    with st.expander("🤔 Microphone Troubleshooting"):
        st.markdown("""
        **Network Mic Issues?** 
        Browsers block mics on HTTP network IPs. We have enabled **Built-in HTTPS**:
        
        1. **Built-in HTTPS (Current)**
           - Your app is now running on `https://`.
           - **Note:** You will see a "Not Secure" warning (Self-signed). 
           - Click **Advanced** -> **Proceed** to allow mic access.
        
        2. **Alternative: Ngrok**
           - `ngrok http 8501`
           - Use the provided `https://` link for a "Clean" certificate.
        
        3. **Chrome Workaround**
           - Go to: `chrome://flags/#unsafely-treat-insecure-origin-as-secure`
           - **Enable** & add `http://YOUR_IP:8501`
           - Relaunch browser.
        """)

    st.subheader("📜 Activity Log")
    for log in reversed(st.session_state.logs):
        st.caption(f"- {log}")
    
    if st.button("🗑️ Reset All"):
        st.session_state.clear()
        st.cache_data.clear()
        st.rerun()

# --- Display Results ---
if st.session_state.error:
    st.error(f"⚠️ **Error:** {st.session_state.error}")

if st.session_state.requirements:
    reqs = st.session_state.requirements
    st.divider()
    
    with st.expander("📝 View Input Transcript"):
        st.write(st.session_state.transcript)

    st.subheader("🔍 Analysis Approach")
    st.warning(reqs.justification)
    
    st.subheader("📊 Information Gathering")
    st.info("\n".join([f"- {i}" for i in reqs.information_gathering]))

    st.subheader("📋 Extracted Requirements")
    req_list = "\n".join([f"{idx+1}. {r}" for idx, r in enumerate(reqs.requirements)])
    st.success(f"**Functional Requirements:**\n\n{req_list}")

    st.divider()
    st.subheader("📤 Export Options")
    col1, col2 = st.columns(2)
    with col1:
        doc = f"REPORT\n\nApproach:\n{reqs.justification}\n\nRequirements:\n{req_list}"
        st.download_button("⬇️ Download (.txt)", doc, "requirements.txt", use_container_width=True)
    with col2:
        with st.expander("📧 Email Report"):
            email = st.text_input("Recipient")
            if st.button("Send Now", use_container_width=True):
                from logic import send_requirements_email
                try:
                    send_requirements_email(email, "AI Requirement Report", reqs, os.getenv("EMAIL_SENDER"), os.getenv("EMAIL_PASSWORD"))
                    st.success("✅ Sent!")
                except Exception as e:
                    st.error(f"❌ Failed: {e}")
