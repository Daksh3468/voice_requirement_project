import streamlit as st
import shutil
import tempfile
import os

from logic import transcribe_audio, extract_requirements, get_whisper_model, analyze_and_improve_requirements

# st.set_page_config(page_title="Health Check")
# st.title("✅ Health Check Passed")
# st.write("If you see this, Streamlit is working.")

# 1. Page Config
st.set_page_config(
    page_title="Voice Requirement AI",
    page_icon="🤖",
    layout="centered"
)

# --- Session State Management ---
if "transcript" not in st.session_state: st.session_state.transcript = None
if "requirements" not in st.session_state: st.session_state.requirements = None
if "improved_requirements" not in st.session_state: st.session_state.improved_requirements = None
if "analysis_questions" not in st.session_state: st.session_state.analysis_questions = None
if "question_answers" not in st.session_state: st.session_state.question_answers = {}
if "general_feedback" not in st.session_state: st.session_state.general_feedback = ""
if "extraction_model" not in st.session_state: st.session_state.extraction_model = "llama-3.3-70b-versatile"
if "error" not in st.session_state: st.session_state.error = None
if "logs" not in st.session_state: st.session_state.logs = []
if "show_improved" not in st.session_state: st.session_state.show_improved = False

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
    st.session_state.improved_requirements = None
    st.session_state.analysis_questions = None
    st.session_state.user_feedback = ""
    st.session_state.show_improved = False
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
            from logic import transcribe_audio
            transcript = transcribe_audio(tmp_path, whisper_model)
            if not transcript or not transcript.strip():
                st.session_state.error = "No speech detected. Please check your mic and try again."
                status.update(label="❌ Silence Detected", state="error")
                return
            
            st.session_state.transcript = transcript
            status.write(f"🧠 Extracting requirements using {st.session_state.extraction_model}...")
            reqs = extract_requirements(transcript, st.session_state.extraction_model)
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
    st.session_state.improved_requirements = None
    st.session_state.analysis_questions = None
    st.session_state.user_feedback = ""
    st.session_state.show_improved = False
    st.session_state.transcript = None
    st.session_state.error = None

    with st.status("🧠 Analyzing Text...", expanded=True) as status:
        try:
            reqs = extract_requirements(text, st.session_state.extraction_model)
            st.session_state.transcript = text
            st.session_state.requirements = reqs
            status.update(label="✅ Analysis Complete", state="complete")
        except Exception as e:
            st.session_state.error = str(e)
            status.update(label="❌ Analysis Failed", state="error")

# --- Sidebar Settings ---
with st.sidebar:
    st.header("⚙️ LLM Configuration")
    
    available_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ]
    
    st.session_state.extraction_model = st.selectbox(
        "Requirement Extraction Model",
        available_models,
        index=0,
        help="Model used to extract initial requirements from transcript/text."
    )
    
    st.session_state.analysis_model = st.selectbox(
        "Deep Analysis Model",
        available_models,
        index=0,
        help="Higher tier model recommended for gap analysis."
    )
    
    st.divider()
    st.subheader("📡 Connection Status")
    llm_ok = "GROQ_API_KEY" in st.secrets
    st.metric("Groq API", "✅ Ready" if llm_ok else "❌ Missing")
    
    if st.button("🗑️ Reset All"):
        st.session_state.clear()
        st.rerun()

# --- UI Layout ---
tab_record, tab_upload, tab_text, tab_email = st.tabs(["🎙️ Record", "📂 Upload", "📧 Text","📥 Email Inbox"])

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

with tab_email:
    st.subheader("📥 Scan Email for Requirements")

    max_emails = st.slider("Emails to scan", 1, 10, 3)
    unread_only = st.checkbox("Unread emails only", value=True)

    if st.button("📡 Fetch & Analyze Emails", type="primary"):
        from logic import fetch_email_requirements

        try:
            with st.status("📨 Reading Inbox...", expanded=True) as status:
                emails = fetch_email_requirements(
                    sender_email=os.getenv("EMAIL_SENDER"),
                    password=os.getenv("EMAIL_PASSWORD"),
                    max_emails=max_emails,
                    unread_only=unread_only
                )

                if not emails:
                    st.warning("No suitable emails found.")
                    status.update(state="complete")
                    st.stop()

                combined_text = "\n\n---\n\n".join(emails)

                status.write(f"🧠 Extracting requirements using {st.session_state.extraction_model}...")
                reqs = extract_requirements(combined_text, st.session_state.extraction_model)

                st.session_state.transcript = combined_text
                st.session_state.requirements = reqs
                status.update(label="✅ Email Analysis Complete", state="complete")

        except Exception as e:
            st.error(f"❌ Email scan failed: {e}")

# --- Sidebar ---
# with st.sidebar:
#     st.header("⚙️ System Status")

#     f_ok = check_ffmpeg()
#     llm_ok = "GROQ_API_KEY" in st.secrets

#     st.metric("Audio Engine (FFmpeg)", "✅ OK" if f_ok else "❌ Missing")
#     st.metric("LLM Engine (Groq)", "✅ Ready" if llm_ok else "❌ API Key Missing")

#     if not llm_ok:
#         st.error("Missing GROQ_API_KEY in Streamlit Secrets")

#     st.divider()
#     with st.expander("🤔 Microphone Troubleshooting"):
#         st.markdown("""
#         **Network Mic Issues?** 
#         Browsers block mics on HTTP network IPs. We have enabled **Built-in HTTPS**:
        
#         1. **Built-in HTTPS (Current)**
#            - Your app is now running on `https://`.
#            - **Note:** You will see a "Not Secure" warning (Self-signed). 
#            - Click **Advanced** -> **Proceed** to allow mic access.
        
#         2. **Alternative: Ngrok**
#            - `ngrok http 8501`
#            - Use the provided `https://` link for a "Clean" certificate.
        
#         3. **Chrome Workaround**
#            - Go to: `chrome://flags/#unsafely-treat-insecure-origin-as-secure`
#            - **Enable** & add `http://YOUR_IP:8501`
#            - Relaunch browser.
#         """)

#     st.subheader("📜 Activity Log")
#     for log in reversed(st.session_state.logs):
#         st.caption(f"- {log}")
    
#     if st.button("🗑️ Reset All"):
#         st.session_state.clear()
#         st.cache_data.clear()
#         st.rerun()

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
    st.success(f"**Functional & Non-Functional Requirements:**\n\n{req_list}")

    # --- NEW: Interactive Analyze & Improve Button ---
    st.divider()
    
    if not st.session_state.analysis_questions and not st.session_state.improved_requirements:
        if st.button("🔍 Analyze & Find Gaps", type="primary"):
            from logic import generate_clarification_questions
            with st.status(f"🧠 Deep Gap Analysis via {st.session_state.analysis_model}...", expanded=True) as status:
                try:
                    questions = generate_clarification_questions(reqs.requirements, st.session_state.analysis_model)
                    st.session_state.analysis_questions = questions
                    status.update(label="✅ Analysis Complete - Questions Generated", state="complete")
                    st.rerun()
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    status.update(label="❌ Analysis Failed", state="error")

    # --- Display Questions & Collect Feedback ---
    if st.session_state.analysis_questions and not st.session_state.improved_requirements:
        st.subheader("🤔 Clarifying Questions")
        st.info("The AI identified some potential gaps. Please answer each question below.")
        
        # Individual answer fields for each question
        for idx, q in enumerate(st.session_state.analysis_questions.questions):
            st.markdown(f"**Q{idx+1}: {q.question}**")
            st.caption(f"*Context: {q.context}*")
            
            # Use a unique key for each question's answer
            answer_key = f"answer_{idx}"
            if answer_key not in st.session_state.question_answers:
                st.session_state.question_answers[answer_key] = ""
            
            st.session_state.question_answers[answer_key] = st.text_area(
                f"Your answer to Q{idx+1}:",
                value=st.session_state.question_answers[answer_key],
                height=100,
                key=f"q_{idx}_input",
                placeholder="Provide your answer or clarification here..."
            )
            st.divider()
        
        # General feedback field
        st.markdown("**Additional Feedback (Optional)**")
        st.session_state.general_feedback = st.text_area(
            "Any additional comments or requirements:",
            value=st.session_state.general_feedback,
            height=150,
            placeholder="e.g., Additional features, constraints, or clarifications not covered above..."
        )
        
        col_gen, col_cancel = st.columns(2)
        with col_gen:
            if st.button("🚀 Generate Final Requirements", type="primary"):
                from logic import finalize_improved_requirements
                with st.status("🧠 Incorporating feedback & generating final set...", expanded=True) as status:
                    try:
                        # Combine all answers and general feedback
                        combined_feedback = ""
                        for idx, q in enumerate(st.session_state.analysis_questions.questions):
                            answer = st.session_state.question_answers.get(f"answer_{idx}", "")
                            combined_feedback += f"Q: {q.question}\nA: {answer}\n\n"
                        
                        if st.session_state.general_feedback.strip():
                            combined_feedback += f"Additional Feedback:\n{st.session_state.general_feedback}"
                        
                        improved = finalize_improved_requirements(
                            reqs.requirements, 
                            combined_feedback, 
                            st.session_state.analysis_model
                        )
                        st.session_state.improved_requirements = improved
                        st.session_state.show_improved = True
                        status.update(label="✅ Final Requirements Generated", state="complete")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Generation failed: {e}")
                        status.update(label="❌ Generation Failed", state="error")
        with col_cancel:
            if st.button("Cancel"):
                st.session_state.analysis_questions = None
                st.session_state.question_answers = {}
                st.session_state.general_feedback = ""
                st.rerun()

    # --- Display Improved Results ---
    if st.session_state.improved_requirements:
        imp = st.session_state.improved_requirements
        st.subheader("🚀 Improved & Merged Requirements")
        
        with st.expander("🚩 Identified Gaps & Missing Requirements", expanded=True):
            for gap in imp.gaps:
                st.markdown(f"- {gap}")
        
        improved_list = "\n".join([f"{idx+1}. {r}" for idx, r in enumerate(imp.improved_requirements)])
        st.success(f"**Merged Final Requirements:**\n\n{improved_list}")

        if st.button("🔙 Back to Original / Re-Analyze"):
            st.session_state.improved_requirements = None
            st.session_state.analysis_questions = None
            st.session_state.question_answers = {}
            st.session_state.general_feedback = ""
            st.session_state.show_improved = False
            st.rerun()

    current_reqs = st.session_state.improved_requirements.improved_requirements if st.session_state.improved_requirements else reqs.requirements
    current_req_list = "\n".join([f"{idx+1}. {r}" for idx, r in enumerate(current_reqs)])
    
    st.divider()
    st.subheader("📤 Export Options")
    col1, col2 = st.columns(2)
    with col1:
        doc = f"REPORT\n\nApproach:\n{reqs.justification}\n\nRequirements:\n{current_req_list}"
        st.download_button("⬇️ Download (.txt)", doc, "requirements.txt", use_container_width=True)
    with col2:
        with st.expander("📧 Email Report"):
            email = st.text_input("Recipient")
            if st.button("Send Now", use_container_width=True):
                from logic import send_requirements_email, RequirementExtraction
                try:
                    # Create a dummy RequirementExtraction object for the email function if using improved ones
                    # or update send_requirements_email to handle both.
                    # For simplicity, let's just send the text report if improved version is used, 
                    # or adapt the pydantic object.
                    
                    # Ideally, we'd update logic.py to handle ImprovedRequirements in email too.
                    # But if we want to reuse the existing HTML template, we can wrap improved into a Extraction object.
                    
                    report_to_send = reqs
                    if st.session_state.improved_requirements:
                        report_to_send = RequirementExtraction(
                            justification="Merged & Improved Requirements Analysis",
                            information_gathering=reqs.information_gathering,
                            requirements=st.session_state.improved_requirements.improved_requirements
                        )
                    
                    send_requirements_email(email, "AI Requirement Report", report_to_send, os.getenv("EMAIL_SENDER"), os.getenv("EMAIL_PASSWORD"))
                    st.success("✅ Sent!")
                except Exception as e:
                    st.error(f"❌ Failed: {e}")
