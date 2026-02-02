import whisper
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List
import os

# --- Configuration ---
MODEL_WHISPER = "base"
MODEL_OLLAMA = "llama3.2"

# --- Setup Environment ---
# Add local 'bin' folder to PATH for FFmpeg if it exists
local_bin = os.path.join(os.getcwd(), "bin")
if os.path.exists(local_bin):
    os.environ["PATH"] += os.pathsep + local_bin

# --- Models ---
class RequirementExtraction(BaseModel):
    requirements: List[str] = Field(description="List of extracted functional and non-functional requirements")

# --- Logic ---

# Load Whisper model once (global cache in memory)
print(f"Loading Whisper model: {MODEL_WHISPER}...")
whisper_model = whisper.load_model(MODEL_WHISPER)
print("Whisper model loaded.")

def transcribe_audio(file_path: str) -> str:
    """
    Transcribes the audio file using OpenAI Whisper.
    """
    try:
        result = whisper_model.transcribe(file_path)
        return result["text"]
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}")

def extract_requirements(transcript: str) -> List[str]:
    """
    Extracts requirements using Ollama (Llama 3.2).
    """
    llm = ChatOllama(model=MODEL_OLLAMA, temperature=0)
    
    prompt = ChatPromptTemplate.from_template(
        """You are an expert business analyst.
        Extract clear, actionable project requirements from the transcript below.
        Return ONLY a JSON object with a 'requirements' list of strings.
        
        Transcript: "{transcript}"
        """
    )
    
    # Structured output for reliability
    structured_llm = llm.with_structured_output(RequirementExtraction)
    chain = prompt | structured_llm
    
    try:
        result = chain.invoke({"transcript": transcript})
        return result.requirements
    except Exception as e:
        # Fallback handling could go here, but raising for UI to catch is fine
        raise RuntimeError(f"LLM extraction failed: {e}")
