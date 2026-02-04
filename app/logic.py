import os
import smtplib
import streamlit as st
from email.message import EmailMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

# --- Configuration ---
MODEL_WHISPER = "base"
MODEL_GROQ = "llama-3.1-8b-instant"

# --- Setup Environment ---
# Add local 'bin' folder to PATH for FFmpeg if it exists
local_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bin")
if os.path.exists(local_bin):
    if local_bin not in os.environ["PATH"]:
        os.environ["PATH"] = local_bin + os.pathsep + os.environ["PATH"]

# --- Models ---
class RequirementExtraction(BaseModel):
    justification: str = Field(description="Why this technique was chosen based on the context of the transcript")
    information_gathering: List[str] = Field(description="List of key information points gathered during the elicitation process")
    requirements: List[str] = Field(description="List of extracted functional and non-functional requirements")

# --- Model Loading (Cached) ---
def get_whisper_model():
    """
    Loads the Whisper model and caches it to avoid redundant loads.
    """
    import whisper
    return whisper.load_model(MODEL_WHISPER)

def transcribe_audio(file_path: str, model=None) -> str:
    """
    Transcribes the audio file using OpenAI Whisper.
    """
    try:
        # If no model is passed, handle internally (though get_whisper_model is preferred)
        if model is None:
            model = get_whisper_model()
        result = model.transcribe(file_path)
        return result["text"]
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}")

def get_llm():
    """
    Lazily initializes the Groq LLM.
    Safe for Streamlit Cloud startup.
    """
    try:
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            api_key=st.secrets["GROQ_API_KEY"]
        )
    except KeyError:
        raise RuntimeError("GROQ_API_KEY not found in Streamlit Secrets")

def extract_requirements(transcript: str) -> RequirementExtraction:
    """
    Extracts requirements using Ollama (Llama 3.2).
    """
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_template(
        """You are a senior business analyst with expertise in analyzing stakeholder interviews and voice transcripts.

        Analyze the following transcript and extract structured project requirements. 
        The transcript may contain informal language, partial sentences, or ambiguous statements. 
        Infer intent carefully and clearly state assumptions where needed.

        Your analysis must include the following sections:

        1. Justification of Analysis Approach
        - Explain the methodology used to analyze the transcript
        - Describe how ambiguity, implicit needs, or incomplete statements were interpreted
        - Justify why certain requirements were inferred from the conversation

        2. Information Gathering Insights
        Extract key insights such as:
        - Business goals or objectives
        - User or stakeholder pain points
        - Identified stakeholders or user roles
        - Current process gaps or limitations
        - High-level expectations or success criteria

        3. Actionable Requirements
        Provide a single, unified list of requirements derived from the transcript.

        - Include functional behaviors, non-functional expectations, constraints, and   assumptions in ONE list.
        - Prefix each requirement with a clear type label:
            [Functional], [Non-Functional], [Constraint], or [Assumption]
        - Write each requirement as a clear, testable statement.
        - Avoid combining multiple ideas into one requirement.
        - Clearly indicate inferred requirements using precise language.

        Example format:
        - [Functional] The system shall allow users to submit requirements via voice input.
        - [Non-Functional] The system shall process voice input with low latency.
        - [Constraint] The system must integrate with existing email infrastructure.
        - [Assumption] It is assumed that users will have access to a microphone-enabled device.

        Use clear bullet points and concise language.
        Avoid speculation unless explicitly stated as an assumption.
        Ensure the output is precise, structured, and implementation-ready.

        Transcript: "{transcript}"
        """
    )
    
    # Structured output for reliability
    structured_llm = llm.with_structured_output(RequirementExtraction)
    chain = prompt | structured_llm
    
    try:
        result = chain.invoke({"transcript": transcript})
        return result # Returns the full Pydantic object now
    except Exception as e:
        # Fallback handling could go here, but raising for UI to catch is fine
        raise RuntimeError(f"LLM extraction failed: {e}")

def format_requirements_html(reqs: RequirementExtraction) -> str:
    """
    Generates a professional, responsive HTML template that matches the Streamlit app's UI.
    """
    # Create the Information Gathering list
    info_list = "\n".join([f"<li>{item}</li>" for item in reqs.information_gathering])
    
    # Create the Requirements list (numbered)
    reqs_list = "\n".join([f"<li>{r}</li>" for r in reqs.requirements])
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta name="color-scheme" content="light dark">
        <style>
            :root {{
                color-scheme: light dark;
            }}
            body {{ 
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6; 
                color: #1f2937; 
                background-color: #f8fafc;
                margin: 0;
                padding: 20px;
            }}
            .email-wrapper {{
                max-width: 600px;
                margin: 0 auto;
            }}
            .container {{ 
                background: #ffffff;
                border-radius: 12px; 
                overflow: hidden;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
                border: 1px solid #e2e8f0;
            }}
            .header {{ 
                background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
                color: white; 
                padding: 30px 40px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 24px;
                font-weight: 700;
            }}
            .content {{ 
                padding: 30px 40px;
            }}
            
            /* Streamlit-like Alert Boxes */
            .st-alert {{
                padding: 16px 20px;
                border-radius: 8px;
                margin-bottom: 24px;
                border: 1px solid transparent;
            }}
            
            /* Warning Style (Analysis Approach) */
            .st-warning {{
                background-color: #fffbeb;
                border-color: #fef3c7;
                color: #92400e;
            }}
            
            /* Info Style (Information Gathering) */
            .st-info {{
                background-color: #eff6ff;
                border-color: #dbeafe;
                color: #1e40af;
            }}
            
            /* Success Style (Requirements) */
            .st-success {{
                background-color: #f0fdf4;
                border-color: #dcfce7;
                color: #166534;
            }}
            
            .subheader {{
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 12px;
                color: #334155;
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            
            ul, ol {{
                margin: 10px 0;
                padding-left: 20px;
            }}
            
            li {{
                margin-bottom: 8px;
            }}

            .regards {{
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #e2e8f0;
                font-size: 14px;
                color: #64748b;
            }}
            
            .signature {{
                margin-top: 12px;
                font-weight: 600;
                color: #6366f1;
            }}

            .footer {{ 
                background: #f1f5f9;
                padding: 20px; 
                text-align: center; 
                font-size: 12px;
                color: #94a3b8;
            }}
            
            /* Dark Mode Adjustments */
            @media (prefers-color-scheme: dark) {{
                body {{ background-color: #0f172a; color: #f1f5f9; }}
                .container {{ background: #1e293b; border-color: #334155; }}
                .subheader {{ color: #e2e8f0; }}
                .st-warning {{ background-color: #451a03; border-color: #78350f; color: #fef3c7; }}
                .st-info {{ background-color: #172554; border-color: #1e3a8a; color: #dbeafe; }}
                .st-success {{ background-color: #052e16; border-color: #064e3b; color: #dcfce7; }}
                .regards {{ border-top-color: #334155; color: #94a3b8; }}
                .footer {{ background: #0f172a; border-top: 1px solid #334155; }}
            }}
        </style>
    </head>
    <body>
        <div class="email-wrapper">
            <div class="container">
                <div class="header">
                    <h1>Voice Requirement AI Report</h1>
                </div>
                
                <div class="content">
                    <div class="subheader">🔍 Analysis Approach</div>
                    <div class="st-alert st-warning">
                        {reqs.justification}
                    </div>
                    
                    <div class="subheader">📊 Information Gathering</div>
                    <div class="st-alert st-info">
                        <ul style="margin: 0;">
                            {info_list}
                        </ul>
                    </div>
                    
                    <div class="subheader">📋 Requirements</div>
                    <div class="st-alert st-success">
                        <strong>Functional Requirements:</strong>
                        <ol style="margin-top: 12px; margin-bottom: 0;">
                            {reqs_list}
                        </ol>
                    </div>
                    
                    <div class="regards">
                        Thank you for using our AI-powered requirements analysis service. We hope this report provides valuable insights for your project.
                        <div class="signature">
                            Best Regards,<br>
                            Voice Requirement AI Team
                        </div>
                    </div>
                </div>
                
                <div class="footer">
                    Generated by Voice Requirement AI<br>
                    Transforming conversations into actionable requirements
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html



def send_requirements_email(recipient: str, subject: str, reqs: RequirementExtraction, sender: str, password: str):
    """
    Sends the requirements via email using SMTP_SSL with HTML formatting and Plain Text fallback.
    """
    # Create the email message
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    
    if not sender or not password:
        raise RuntimeError("Email configuration missing. Please check your .env file for EMAIL_SENDER and EMAIL_PASSWORD.")
    
    # Sanitize password (remove spaces often found in Google App Passwords)
    password = password.replace(" ", "")

    # 1. Plain Text Version (Primary content)
    text_content = f"Project Requirements Report\n\n"
    text_content += f"Analysis Approach:\n{reqs.justification}\n\n"
    
    text_content += "Information Gathering:\n"
    for item in reqs.information_gathering:
        text_content += f"- {item}\n"
    text_content += "\n"
    
    text_content += "Requirements:\n"
    for r in reqs.requirements:
        text_content += f"- {r}\n"
    
    msg.set_content(text_content)
    
    # 2. HTML Version (Alternative)
    html_content = format_requirements_html(reqs)
    msg.add_alternative(html_content, subtype='html')

    try:
        # Connecting to Gmail's SMTP server using SSL
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.send_message(msg)
    except Exception as e:
        raise RuntimeError(f"Email failed: {e}")
