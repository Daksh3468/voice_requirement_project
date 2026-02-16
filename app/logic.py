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

class ImprovedRequirements(BaseModel):
    gaps: List[str] = Field(description="List of identified gaps, missing features, or ambiguities in the original requirements")
    improved_requirements: List[str] = Field(description="The complete high-quality list of requirements (Original + New improvements merged)")

class AnalysisQuestion(BaseModel):
    question: str = Field(description="Clarifying question for the user")
    context: str = Field(description="Why this question is being asked (e.g., specific gap found)")

class AnalysisQuestions(BaseModel):
    questions: List[AnalysisQuestion] = Field(description="List of clarifying questions to improve requirements")

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

def get_llm(model_name: str = "llama-3.3-70b-versatile"):
    """
    Lazily initializes the Groq LLM with a specific model.
    """
    try:
        return ChatGroq(
            model=model_name,
            temperature=0.7,
            api_key=st.secrets["GROQ_API_KEY"]
        )
    except KeyError:
        raise RuntimeError("GROQ_API_KEY not found in Streamlit Secrets")

def extract_requirements(transcript: str, model_name: str = "llama-3.3-70b-versatile") -> RequirementExtraction:
    """
    Extracts requirements using the specified Groq model.
    """
    llm = get_llm(model_name)
    
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
        Provide a single, unified list of important requirements derived from the transcript.

        - Include functional behaviors, non-functional expectations in list.
        - Prefix each requirement with a clear type label:
            [Functional] or [Non-Functional]
        - Write each requirement as a clear, testable statement.
        - Avoid combining multiple ideas into one requirement.
        - Clearly indicate inferred requirements using precise language.

        Example format:
        - [Functional] The system shall allow users to submit requirements via voice input.
        - [Non-Functional] The system shall process voice input with low latency.

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

def analyze_and_improve_requirements(original_reqs: List[str], model_name: str = "llama-3.1-8b-instant") -> ImprovedRequirements:
    """
    Analyzes the original requirements, finds gaps, and generates an improved merged list.
    """
    llm = get_llm(model_name)
    
    # Format the requirements for the prompt
    reqs_text = "\n".join(original_reqs)
    
    prompt = ChatPromptTemplate.from_template(
        """You are a senior requirements engineer and quality assurance expert.
        
        Your task is to review the following set of requirements and perform a deep gap analysis.
        
        Original Requirements:
        {requirements}
        
        Perform the following:
        1. Identify missing functional requirements that are logically expected based on the existing ones (e.g., if there's a 'login', is there a 'logout' or 'password reset'?).
        2. Identify missing non-functional requirements (security, performance, scalability, usability).
        3. Find ambiguities or vague statements in the original set.
        4. Generate a 'Gaps' list summarizing these findings.
        5. Provide a final 'Improved Requirements' list which is a MERGED and REFINED version of the original requirements plus the new ones you identified.
        
        Rules:
        - Keep the [Functional] and [Non-Functional] labels.
        - Ensure every original requirement is preserved but can be rephrased for better clarity.
        - The final list must be comprehensive and implementation-ready.
        
        Your output must follow the structured format.
        """
    )
    
    structured_llm = llm.with_structured_output(ImprovedRequirements)
    chain = prompt | structured_llm
    
    try:
        result = chain.invoke({"requirements": reqs_text})
        return result
    except Exception as e:
        raise RuntimeError(f"Requirement improvement failed: {e}")

def generate_clarification_questions(original_reqs: List[str], model_name: str = "llama-3.3-70b-versatile") -> AnalysisQuestions:
    """
    Analyzes original requirements and generates clarifying questions to fill gaps.
    """
    llm = get_llm(model_name)
    reqs_text = "\n".join(original_reqs)
    
    prompt = ChatPromptTemplate.from_template(
        """You are a senior business analyst. Review these requirements and find gaps, missing edge cases, or ambiguities.
        
        Original Requirements:
        {requirements}
        
        Generate between 5-10 focused, professional clarifying questions for the stakeholder to help make these requirements complete and implementation-ready.
        Each question should be tied to a specific gap or ambiguity you've identified.
        """
    )
    
    structured_llm = llm.with_structured_output(AnalysisQuestions)
    chain = prompt | structured_llm
    
    try:
        return chain.invoke({"requirements": reqs_text})
    except Exception as e:
        raise RuntimeError(f"Question generation failed: {e}")

def finalize_improved_requirements(original_reqs: List[str], user_feedback: str, model_name: str = "llama-3.3-70b-versatile") -> ImprovedRequirements:
    """
    Generates improved requirements by merging original ones with user feedback.
    """
    llm = get_llm(model_name)
    reqs_text = "\n".join(original_reqs)
    
    prompt = ChatPromptTemplate.from_template(
        """You are a senior requirements engineer.
        
        Original Requirements:
        {requirements}
        
        User Feedback/Clarifications:
        {feedback}
        
        Generate a final, comprehensive, and high-quality list of requirements.
        - Merge the original requirements with the new information provided in the feedback.
        - Resolve any ambiguities mentioned in the feedback.
        - Use [Functional] and [Non-Functional] labels.
        - Ensure the output is implementation-ready.
        
        Produce a list of 'gaps' you solved and the final 'improved_requirements' list.
        """
    )
    
    structured_llm = llm.with_structured_output(ImprovedRequirements)
    chain = prompt | structured_llm
    
    try:
        return chain.invoke({"requirements": reqs_text, "feedback": user_feedback})
    except Exception as e:
        raise RuntimeError(f"Final requirement generation failed: {e}")

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

# -------------------------Scan Email Code-----------------------------

import imaplib
import email
from email.header import decode_header
from bs4 import BeautifulSoup

def fetch_email_requirements(
    sender_email: str,
    password: str,
    max_emails: int = 5,
    unread_only: bool = True
) -> list[str]:
    """
    Fetches recent emails and extracts text content for requirement analysis.
    Returns a list of email bodies.
    """

    if not sender_email or not password:
        raise RuntimeError("Email credentials missing")

    password = password.replace(" ", "")

    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(sender_email, password)
    mail.select("inbox")

    search_criteria = "(UNSEEN)" if unread_only else "ALL"
    status, messages = mail.search(None, search_criteria)

    if status != "OK":
        raise RuntimeError("Failed to search inbox")

    email_ids = messages[0].split()[-max_emails:]
    extracted_texts = []

    for eid in email_ids:
        _, msg_data = mail.fetch(eid, "(RFC822)")
        raw_email = msg_data[0][1]
        msg = email.message_from_bytes(raw_email)

        body_text = ""

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    body_text = part.get_payload(decode=True).decode(errors="ignore")
                    break
                elif content_type == "text/html":
                    html = part.get_payload(decode=True).decode(errors="ignore")
                    soup = BeautifulSoup(html, "html.parser")
                    body_text = soup.get_text()
        else:
            body_text = msg.get_payload(decode=True).decode(errors="ignore")

        if body_text.strip():
            extracted_texts.append(body_text.strip())

    mail.logout()
    return extracted_texts
