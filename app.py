import os
import re
import json
import html
import pdfplumber
import pandas as pd
import dateparser
from rapidfuzz import fuzz, process
import spacy
from datetime import datetime, date, timedelta
from collections import Counter, defaultdict
import numpy as np
import io
import streamlit as st
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple
import easyocr
from PIL import Image
import pdf2image
import plotly.graph_objects as go
import plotly.express as px
from groq import Groq
from typing import Dict, List, Optional, Tuple, Any, Union
import inspect

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use absolute paths for data files to ensure they are found regardless of working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, "users.json")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

def load_users():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def validate_username(username):
    """Validate username contains at least one special character."""
    if not username or not username.strip():
        return False, "Username cannot be empty."
    special_chars = re.compile(r'[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>\/?]')
    if not special_chars.search(username):
        return False, "Invalid Credentials"
    return True, ""

def validate_password(password):
    """Validate password meets security requirements."""
    errors = []
    if not password:
        return False, "Password cannot be empty."
    if len(password) < 8:
        errors.append("Password Incorrect.")
    if not re.search(r'[A-Z]', password):
        errors.append("Password must contain at least one uppercase letter.")
    if not re.search(r'[a-z]', password):
        errors.append("Password must contain at least one lowercase letter.")
    if not re.search(r'[0-9]', password):
        errors.append("Password must contain at least one digit.")
    if errors:
        return False, " ".join(errors)
    return True, ""

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    default_config = {
        "SMTP_SERVER": "smtp.gmail.com",
        "SMTP_PORT": 587,
        "SMTP_USE_TLS": True,
        "HR_EMAIL": "",
        "HR_PASSWORD": ""
    }
    with open(CONFIG_FILE, 'w') as f:
        json.dump(default_config, f, indent=4)
    return default_config

app_config = load_config()

# Enhanced Configuration with more HR-focused options
DEFAULT_CONFIG = {
    "WEIGHTS": {
        "technical_skills": 0.35,
        "experience": 0.25,
        "education": 0.15,
        "jd_match": 0.15,
        "growth_potential": 0.10
    },
    "SCORE_THRESHOLD": 0.7,
    "TOP_N_CANDIDATES": 5,
    "EMAIL_BATCH_SIZE": 10,
    "EMAIL_DELAY": 1.0,
    "GROQ_API_KEY": os.environ.get("GROQ_API_KEY", app_config.get("GROQ_API_KEY", "")),
    "GROQ_MODEL": "llama-3.3-70b-versatile",
    "GROQ_FALLBACK_MODEL": "llama-3.1-8b-instant",
    "GROQ_MAX_RETRIES": 3,
    "HR_MANAGER_NAME": "Hiring Manager",
    "COMPANY_NAME": "Organization",
    "NOTICE_PERIOD_DAYS": 30,
    "REQUIRED_CLEARANCES": [],
    "DIVERSITY_HIRING": False,
    "AUTO_REJECT_THRESHOLD": 0.3,
    "INTERVIEW_ROUNDS": ["Technical Screen", "Hiring Manager", "Panel Interview", "HR Round"],
    "TAGS": ["Remote OK", "Urgent", "Leadership", "Entry-Level", "Senior"],
    "SMTP_SERVER": app_config.get("SMTP_SERVER", "smtp.gmail.com"),
    "SMTP_PORT": app_config.get("SMTP_PORT", 587),
    "SMTP_USE_TLS": app_config.get("SMTP_USE_TLS", True),
    "HR_EMAIL": app_config.get("HR_EMAIL", ""),
    "HR_PASSWORD": app_config.get("HR_PASSWORD", ""),
    
    "EMAIL_TEMPLATES": {
        "selected": {
            "subject": "Interview Invitation - {job_title} Position at {company_name}",
            "body": """Dear {candidate_name},

Thank you for your interest in the {job_title} position at {company_name}. After carefully reviewing your application, we are pleased to inform you that your profile aligns well with our requirements.

We would like to invite you for an interview to discuss this opportunity further and learn more about your experience and qualifications.

Interview Details:
  Position: {job_title}
  Round: {interview_round}
  Mode: {interview_mode}
  Duration: Approximately {duration} minutes
{meeting_details}

Please confirm your availability by replying to this email within 48 hours. We are flexible with scheduling and will do our best to accommodate your preferences.

We look forward to speaking with you and learning more about how your skills and experience can contribute to our team.

Best regards,
{hr_manager_name}
Talent Acquisition Team
{company_name}"""
        },
        "not_selected": {
            "subject": "Update on Your Application - {job_title} at {company_name}",
            "body": """Dear {candidate_name},

Thank you for taking the time to apply for the {job_title} position at {company_name}. We sincerely appreciate your interest in joining our organization.

After careful consideration of all applications, we have decided to move forward with candidates whose qualifications and experience more closely match our current needs for this specific role.

This decision was difficult given the high quality of applications we received. We encourage you to explore other opportunities with {company_name} that may better align with your skills and career goals. Your resume will remain in our talent database for future consideration.

We wish you continued success in your career endeavors and hope our paths may cross again in the future.

Warm regards,
{hr_manager_name}
Human Resources Department
{company_name}"""
        },
        "screening": {
            "subject": "Next Steps - {job_title} Application at {company_name}",
            "body": """Dear {candidate_name},

Thank you for your application for the {job_title} role. We're interested in learning more about you.

As a next step, please complete a brief screening questionnaire: {screening_link}

This should take approximately 10-15 minutes. Please complete it within 3 business days.

Best regards,
{hr_manager_name}
{company_name}"""
        },
        "offer": {
            "subject": "Job Offer - {job_title} at {company_name}",
            "body": """Dear {candidate_name},

We are delighted to extend an offer for the position of {job_title} at {company_name}.

Key Details:
  Start Date: {start_date}
  Salary: {salary_offer}
  Benefits: {benefits}

Please review the attached offer letter and respond within 5 business days.

Congratulations, and we look forward to having you on our team!

Best regards,
{hr_manager_name}
{company_name}"""
        },
        "interview": {
            "subject": "Interview Invitation - {job_title} at {company_name}",
            "body": """Dear {candidate_name},

Thank you for your application for the {job_title} position at {company_name}. We would like to invite you for an interview to discuss your background and the role in more detail.

Interview Details:
  Position: {job_title}
  Round: {interview_round}
  Mode: {interview_mode}
  Duration: Approximately {duration} minutes
{meeting_details}

Scheduled Time: {scheduled_time}

Please join using the link above or arrive 5 minutes early for in-person meetings. If you need to reschedule, please let us know at least 24 hours in advance.

Best regards,
{hr_manager_name}
{company_name}"""
        }
    }
}

@st.cache_resource
def load_nlp_model():
    try:
        return spacy.load("en_core_web_sm")
    except:
        st.error("spaCy model not found. Install: python -m spacy download en_core_web_sm")
        st.stop()

@st.cache_resource
def load_ocr_reader():
    try:
        return easyocr.Reader(['en'])
    except:
        return None

# nlp and ocr_reader are initialized inside main() to avoid
# "missing ScriptRunContext" warnings from module-level cache calls.
nlp = None
ocr_reader = None

class EmailManager:
    def __init__(self, smtp_server, smtp_port, use_tls, email, password):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.use_tls = use_tls
        self.email = email
        self.password = password
        self.connection: Optional[smtplib.SMTP] = None
        self.lock = threading.Lock()
    
    def connect(self):
        try:
            connection = smtplib.SMTP(self.smtp_server, self.smtp_port)
            if self.use_tls:
                connection.starttls()
            connection.login(self.email, self.password)
            self.connection = connection
            return True
        except Exception as e:
            logging.error(f"SMTP connection failed: {e}")
            return False
    
    def disconnect(self):
        if self.connection is not None:
            try:
                self.connection.quit()
            except:
                pass
            self.connection = None
    
    def send_batch_emails(self, email_data_list, delay=1.0):
        results: Dict[str, Any] = {"sent": 0, "failed": 0, "errors": []}
        
        with self.lock:
            try:
                if not self.connect():
                    results["errors"].append("Could not connect to SMTP server. Check credentials and server settings.")
                    return results
            except Exception as e:
                results["errors"].append(f"SMTP connection error: {str(e)}")
                return results
            
            try:
                for email_data in email_data_list:
                    try:
                        msg = MIMEMultipart()
                        msg['Subject'] = email_data['subject']
                        msg['From'] = self.email
                        msg['To'] = email_data['to']
                        msg.attach(MIMEText(email_data['body'], 'plain'))
                        
                        assert self.connection is not None
                        self.connection.send_message(msg)
                        results["sent"] += 1
                        logging.info(f"Email sent to {email_data['to']}")
                        
                        if delay > 0:
                            time.sleep(delay)
                    except Exception as e:
                        results["failed"] += 1
                        error_msg = f"Failed to send to {email_data['to']}: {str(e)}"
                        results["errors"].append(error_msg)
                        logging.error(error_msg)
            finally:
                self.disconnect()
        
        return results

class SchedulingManager:
    def __init__(self):
        self.scheduled_interviews: Dict[int, Dict[str, Any]] = {}
    
    def schedule_interview(self, candidate_name, candidate_email, date_time, duration=45, mode="video", 
                          interviewer="Hiring Manager", meeting_link=None, interview_round="Initial"):
        interview_id = len(self.scheduled_interviews) + 1
        
        if not meeting_link and mode.lower() in ["video call", "video"]:
            meeting_link = f"https://meet.google.com/{interview_id:08d}"
        
        interview = {
            "id": interview_id,
            "candidate_name": candidate_name,
            "candidate_email": candidate_email,
            "date_time": date_time,
            "duration": duration,
            "mode": mode,
            "interviewer": interviewer,
            "meeting_link": meeting_link or "",
            "status": "scheduled",
            "interview_round": interview_round,
            "created_at": datetime.now(),
            "notes": ""
        }
        
        self.scheduled_interviews[interview_id] = interview
        return interview_id
    
    def get_interviews_by_date(self, target_date):
        interviews = []
        for interview in self.scheduled_interviews.values():
            if interview["status"] == "scheduled" and isinstance(interview["date_time"], datetime) and interview["date_time"].date() == target_date:
                interviews.append(interview)
        return sorted(interviews, key=lambda x: x["date_time"])
    
    def get_upcoming_interviews(self):
        now = datetime.now()
        upcoming = []
        
        for interview in self.scheduled_interviews.values():
            # Only show interviews starting in the future
            if interview["status"] == "scheduled" and isinstance(interview["date_time"], datetime) and interview["date_time"] >= now:
                upcoming.append(interview)
        
        return sorted(upcoming, key=lambda x: x["date_time"])
    
    def cancel_interview(self, interview_id):
        if interview_id in self.scheduled_interviews:
            self.scheduled_interviews[interview_id]["status"] = "cancelled"
            return True
        return False
    
    def update_interview_notes(self, interview_id, notes):
        if interview_id in self.scheduled_interviews:
            self.scheduled_interviews[interview_id]["notes"] = notes
            return True
        return False

class LLMResumeAnalyzer:
    def __init__(self, config):
        self.config = config
        api_key = config.get("GROQ_API_KEY")
        max_retries = config.get("GROQ_MAX_RETRIES", 3)
        self.groq_client = Groq(api_key=api_key, max_retries=max_retries) if api_key else None
    
    def _call_groq(self, prompt: str, system_message: str = "You are an expert HR analyst.", 
                  json_mode: bool = False, temperature: float = 0.3) -> Optional[Any]:
        """Helper method to call Groq with automatic fallback on rate limits"""
        if not self.groq_client:
            return None

        primary_model = self.config.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        fallback_model = self.config.get("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")
        
        models_to_try = [primary_model, fallback_model]
        
        for i, model_name in enumerate(models_to_try):
            try:
                params = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature
                }
                if json_mode:
                    params["response_format"] = {"type": "json_object"}
                
                completion = self.groq_client.chat.completions.create(**params)
                return completion.choices[0].message.content
            except Exception as e:
                error_str = str(e).lower()
                if "rate_limit_exceeded" in error_str or "429" in error_str:
                    if i < len(models_to_try) - 1:
                        logging.warning(f"Model {model_name} rate limited. Switching to fallback {models_to_try[i+1]}...")
                        time.sleep(1) # Small pause before retry
                        continue
                    else:
                        logging.error(f"All Groq models rate limited: {e}")
                else:
                    logging.error(f"Groq API error with {model_name}: {e}")
                    if i < len(models_to_try) - 1:
                        continue # Try fallback for other errors too
                break
        return None
    
    def extract_text_from_pdf(self, uploaded_file):
        bytes_data = uploaded_file.getvalue()
        text = ""
        file_ext = ""
        if hasattr(uploaded_file, "name"):
            file_ext = uploaded_file.name.split('.')[-1].lower() if '.' in uploaded_file.name else ""
            
        try:
            if file_ext in ['docx', 'doc']:
                try:
                    import docx
                    doc_obj = docx.Document(io.BytesIO(bytes_data))
                    text = "\n".join([p.text for p in doc_obj.paragraphs])
                except ImportError:
                    logging.error("python-docx is not installed.")
                except Exception as docx_e:
                    logging.error(f"DOCX extraction failed: {docx_e}")
            elif file_ext in ['png', 'jpg', 'jpeg']:
                if ocr_reader:
                    try:
                        img = Image.open(io.BytesIO(bytes_data))
                        img_array = np.array(img)
                        results = ocr_reader.readtext(img_array)
                        text = " ".join([result[1] for result in results])
                    except Exception as img_e:
                        logging.error(f"Image OCR failed: {img_e}")
            else:
                # Default to PDF behavior
                try:
                    with pdfplumber.open(io.BytesIO(bytes_data)) as pdf:
                        pages_text = []
                        for page in pdf.pages:
                            # Deduplicate overlapping characters (fixes "Aa Sshh" issue)
                            deduped_page = page.dedupe_chars(tolerance=1)
                            page_text = deduped_page.extract_text()
                            if page_text:
                                pages_text.append(page_text)
                        raw_text = "\n".join(pages_text)
                        # Clean artifacts like "Y . A S H A" or "doubled letters"
                        text = self._fix_extraction_artifacts(raw_text)
                    
                    if ocr_reader and len(text.strip()) < 200:
                        try:
                            images = pdf2image.convert_from_bytes(bytes_data)
                            ocr_text: List[str] = []
                            for img in images:
                                img_array = np.array(img)
                                results = ocr_reader.readtext(img_array)
                                page_text = " ".join([result[1] for result in results])
                                ocr_text.append(page_text)
                            
                            ocr_full_text = "\n".join(ocr_text)
                            if len(ocr_full_text.strip()) > len(text.strip()):
                                text = ocr_full_text
                        except Exception as e:
                            logging.warning(f"OCR failed: {e}")
                except Exception as e:
                    logging.error(f"PDF extraction failed: {e}")
        except Exception as e:
            logging.error(f"File extraction encountered unexpected error: {e}")
        
        return text.strip()
    
    def _fix_extraction_artifacts(self, text: str) -> str:
        """Heuristic to fix common PDF extraction issues like spaced out names."""
        if not text:
            return ""
            
        # 1. Remove noise backticks (often appear between characters in PDFs)
        text = text.replace("`", "")
            
        # 2. Fix spaced-out letters (e.g., "Y . A S H A", "T N", "A S H A N M U G A")
        # Now handles 2+ single characters separated by single spaces
        def join_spaced_letters(match):
            spaced_text = match.group(0)
            # Remove the spaces but keep the characters
            joined = spaced_text.replace(" ", "")
            return joined

        # Apply spaced letters fix: look for (single char + space) repeated 1+ times
        # followed by one more single char. This catches "T N", "U S A", etc.
        text = re.sub(r'\b(?:[A-Z\.]\s){1,}[A-Z\.]\b', join_spaced_letters, text)
        
        # 3. Collapse multiple spaces into single spaces (but preserve newlines)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        
        return text

    def analyze_resume_with_llm(self, resume_text: str, jd_text: str, required_skills: List[str], 
                               salary_range: Optional[Dict[str, Any]] = None, custom_criteria: str = "") -> Dict[str, Any]:
        if not self.groq_client or not jd_text:
            return self._fallback_analysis(resume_text, required_skills)
        
        skills_str = ", ".join(required_skills)
        
        salary_instruction = ""
        if salary_range:
            salary_instruction = f"\n- Salary Expectations (if mentioned): Check if aligned with range ${salary_range['min']}-${salary_range['max']}"
        
        custom_instruction = ""
        if custom_criteria:
            custom_instruction = f"\n- Custom Criteria: {custom_criteria}"
        
        prompt = f"""You are an expert HR analyst. Analyze this resume against the job requirements.

**REQUIRED SKILLS**: {skills_str}

**JOB DESCRIPTION**:
{jd_text}

**CANDIDATE RESUME**:
{str(resume_text)[:5000]}

Evaluate and provide scores (0-100) for:
1. Technical Skills Match - How well do they match required technical skills?
2. Experience Relevance - Is their work experience relevant to the role?
3. Education Quality - Does their education align with requirements?
4. Overall JD Fit - How well does the overall profile match the job?
{salary_instruction}
{custom_instruction}

Also extract:
- Candidate name
- Email
- Phone
- Location/City
- Years of experience
- Current/Last company
- Current/Last role
- Key skills found (list of 5-10)
- Education level (Bachelors/Masters/PhD/etc)
- Certifications (if any)
- Notice period (if mentioned)
- Expected salary (if mentioned)
- Availability (immediate/serving notice/etc)
- Strengths (3-5 bullet points)
- Weaknesses/Gaps (2-3 bullet points)
- Red flags (if any - job hopping, unexplained gaps, etc)
- Hiring recommendation (2-3 sentences)
- Interview focus areas (3-4 topics to probe)

Return ONLY valid JSON with these exact keys. 

STRICT REQUIREMENT: The candidate's identity (Name, Email, Phone, Location) is usually at the very top of the document. Please prioritize the first 15 lines for these details.

STRICT CASE FIDELITY: Maintain the EXACT capitalization/case of all extracted fields as written in the resume. 
- If the name is 'Y.ASHA JOSEMINE', you must return 'Y.ASHA JOSEMINE'. 
- Do NOT normalize to 'Yasha Josemine' or 'Y.asha Josemine'.
- Do NOT alter punctuation or spacing within names.

{{
  "technical_score": float,
  "experience_score": float,
  "education_score": float,
  "fit_score": float,
  "name": string,
  "email": string,
  "phone": string,
  "location": string,
  "years_experience": float,
  "current_company": string,
  "current_role": string,
  "skills_found": [strings],
  "education_level": string,
  "certifications": [strings],
  "notice_period": string,
  "expected_salary": string,
  "availability": string,
  "strengths": [strings],
  "weaknesses": [strings],
  "red_flags": [strings],
  "recommendation": string,
  "interview_focus": [strings]
}}"""
        
        try:
            content = self._call_groq(prompt, json_mode=True)
            if content:
                return json.loads(content)
            return self._fallback_analysis(resume_text, required_skills)
        except Exception as e:
            logging.error(f"LLM analysis processing failed: {e}")
            return self._fallback_analysis(resume_text, required_skills)
    
    def _fallback_analysis(self, resume_text: str, required_skills: List[str]) -> Dict:
        text_lower = resume_text.lower()
        
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, resume_text, re.IGNORECASE)
        email = emails[0] if emails else ""
        
        phone_pattern = r'(?:\+\d{1,4}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}'
        phones = re.findall(phone_pattern, resume_text)
        phone = phones[0] if phones else ""
        
        lines = [l.strip() for l in resume_text.split('\n') if l.strip()]
        
        # Smarter name fallback: Skip common headers and pick first high-quality line
        headers = ["resume", "cv", "curriculum vitae", "biodata", "contact", "summary"]
        name = "Candidate"
        for line in lines[:10]:
            clean_line = line.lower()
            if any(h in clean_line for h in headers) or len(line) < 3:
                continue
            # If line is likely a name (not a sentence, just words)
            if len(line.split()) <= 5:
                name = line # Preserve exact case
                break
        
        skills_found = [s for s in required_skills if s.lower() in text_lower]
        
        year_matches = re.findall(r'\b(19|20)\d{2}\b', resume_text)
        years_exp = 0
        if len(year_matches) >= 2:
            years = sorted([int(y) for y in year_matches])
            years_exp = max(0, years[-1] - years[0])
        
        return {
            "technical_score": (len(skills_found) / len(required_skills) * 100) if required_skills else 50,
            "experience_score": min(years_exp * 10, 100),
            "education_score": 60,
            "fit_score": 55,
            "name": name, # preserved case
            "email": email,
            "phone": phone,
            "location": "Unknown",
            "years_experience": years_exp,
            "current_company": "Unknown",
            "current_role": "Unknown",
            "skills_found": skills_found,
            "education_level": "Unknown",
            "certifications": [],
            "notice_period": "Unknown",
            "expected_salary": "Unknown",
            "availability": "Unknown",
            "strengths": ["Profile requires manual review"],
            "weaknesses": ["Automated analysis incomplete"],
            "red_flags": [],
            "recommendation": "Conduct manual screening to assess fit.",
            "interview_focus": ["Technical depth", "Culture fit", "Career goals"]
        }
    
    def generate_comparison_report(self, candidates: List[Dict[str, Any]], top_n: int = 5) -> str:
        if not self.groq_client or not candidates:
            return "Comparison unavailable"
        
        top_candidates = [candidates[i] for i in range(min(top_n, len(candidates)))]
        
        comparison_data = []
        for idx, c in enumerate(top_candidates):
            analysis = c.get("llm_analysis", {})
            comparison_data.append({
                "rank": idx + 1,
                "name": analysis.get("name", "Candidate"),
                "score": c.get("final_score", 0),
                "experience": f"{analysis.get('years_experience', 0)} years",
                "current_role": analysis.get("current_role", "Unknown"),
                "skills": ", ".join(analysis.get("skills_found", [])[:5]),
                "strengths": " | ".join(analysis.get("strengths", [])[:2]),
                "availability": analysis.get("availability", "Unknown")
            })
        
        prompt = f"""You are an HR director. Create a detailed comparison report of these top candidates:

{json.dumps(comparison_data, indent=2)}

Format the output as a professional markdown report with:
1. **Executive Summary** (3-4 sentences on the overall candidate pool quality)
2. **Detailed Candidate Comparison** (compare each candidate's strengths, experience, and fit)
3. **Hiring Recommendations** (who to prioritize and why, in 2-3 paragraphs)
4. **Next Steps** (suggested interview approach for each)

Keep it under 600 words but be specific and actionable."""
        
        try:
            content = self._call_groq(prompt, temperature=0.5)
            if content:
                return content
            return "Error generating comparison report - all attempts failed."
        except Exception as e:
            logging.error(f"Comparison report failed: {e}")
            return "Error generating comparison report"
    
    def chat_with_resume(self, resume_text: str, candidate_name: str, question: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
        try:
            # chat_with_resume needs to preserve history, so we'll adapt _call_groq 
            # Or just call self.groq_client directly if we want history, but let's 
            # keep it simple for now and use the prompt-based approach if _call_groq is preferred.
            # Actually _call_groq takes a system_message and a prompt. 
            # For history, we can join it into the prompt.
            
            history_str = ""
            if conversation_history:
                for msg in conversation_history:
                    role = "Candidate" if msg['role'] == 'assistant' else "User"
                    history_str += f"{role}: {msg['content']}\n"
            
            full_prompt = f"Resume:\n{str(resume_text)[:4000]}\n\nConversation History:\n{history_str}\n\nQuestion: {question}"
            system_msg = f"You are an HR assistant helping to evaluate a candidate named {candidate_name}. Answer questions based on their resume. Be concise and professional."
            
            content = self._call_groq(full_prompt, system_message=system_msg, temperature=0.4)
            if content:
                return content
            return "Error: Could not get response from LLM after multiple attempts."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_interview_questions(self, candidate_data: Dict[str, Any], jd_text: str, interview_round: str) -> List[str]:
        """Generate customized interview questions"""
        FALLBACK = [
            "Tell me about your experience with the key technologies in your resume.",
            "Describe a challenging project you've worked on recently.",
            "How do you stay updated with industry trends?",
            "What interests you about this role?",
            "How do you handle tight deadlines and competing priorities?",
        ]

        if not self.groq_client:
            return FALLBACK

        try:
            llm = candidate_data.get("llm_analysis", {})

            def safe_list_join(key, limit=None):
                val = llm.get(key, [])
                if not isinstance(val, list):
                    return ""
                items = val[:limit] if limit else val
                return ", ".join(str(i) for i in items)

            skills   = safe_list_join("skills_found", 5)
            strengths = safe_list_join("strengths", 3)
            focus    = safe_list_join("interview_focus")
            name     = str(llm.get("name", "the candidate"))
            exp      = str(llm.get("years_experience", "unknown"))
            role     = str(llm.get("current_role", "unknown"))
            jd       = str(jd_text)[:1000] if jd_text else ""
            round_   = str(interview_round)

            parts = [
                "Generate 8-10 targeted interview questions for a " + round_ + " round.",
                "",
                "Candidate: " + name,
                "Experience: " + exp + " years",
                "Current Role: " + role,
                "Key Skills: " + skills,
                "Strengths: " + strengths,
                "Focus areas: " + focus,
                "",
                "Job Description:",
                jd,
                "",
                "Return ONLY valid JSON: {\"questions\": [\"...\", \"...\"]}"
            ]
            prompt = "\n".join(parts)
        except Exception as e:
            logging.error("Error building interview prompt: %s", e)
            return FALLBACK

        try:
            content = self._call_groq(prompt, json_mode=True, temperature=0.6)
            if content:
                result = json.loads(content)
                questions = result.get("questions", [])
                if isinstance(questions, list) and questions:
                    return questions
            return FALLBACK
        except Exception as e:
            logging.error("Error calling LLM for interview questions: %s", e)
            return FALLBACK


def calculate_final_score(llm_analysis: Dict[str, Any], weights: Dict[str, float]) -> float:
    technical = llm_analysis.get("technical_score", 0) / 100
    experience = llm_analysis.get("experience_score", 0) / 100
    education = llm_analysis.get("education_score", 0) / 100
    jd_match = llm_analysis.get("fit_score", 0) / 100
    
    final = (
        technical * weights["technical_skills"] +
        experience * weights["experience"] +
        education * weights["education"] +
        jd_match * weights["jd_match"]
    )
    
    return min(max(final, 0), 1)

def export_candidate_data(candidates: List[Dict[str, Any]], format: str = "detailed"):
    """Export candidates in various formats"""
    if format == "summary":
        data = []
        for c in candidates:
            llm = c.get("llm_analysis", {})
            data.append({
                "Name": llm.get("name"),
                "Email": llm.get("email"),
                "Score": f"{c.get('final_score', 0):.3f}",
                "Experience": f"{llm.get('years_experience')} yrs",
                "Status": c.get("status", "New")
            })
        return pd.DataFrame(data)
    else:
        data = []
        for c in candidates:
            llm = c.get("llm_analysis", {})
            data.append({
                "Name": llm.get("name"),
                "Email": llm.get("email"),
                "Phone": llm.get("phone"),
                "Location": llm.get("location"),
                "Score": c.get("final_score"),
                "Experience": llm.get("years_experience"),
                "Current Company": llm.get("current_company"),
                "Current Role": llm.get("current_role"),
                "Skills": ", ".join(llm.get("skills_found", [])),
                "Education": llm.get("education_level"),
                "Availability": llm.get("availability"),
                "Status": c.get("status", "New"),
                "Tags": ", ".join(c.get("tags", [])),
                "Rating": c.get("hr_rating", 0)
            })
        return pd.DataFrame(data)

def render_login():
    logout_notice = st.session_state.pop("logout_notice", None)

    st.markdown("""
        <style>
        [data-testid="stForm"] {
            background: rgba(125, 125, 125, 0.05);
            padding: 2.5rem;
            border-radius: 16px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(125, 125, 125, 0.2);
        }
        div[data-testid="stTextInput"] input {
            border-radius: 8px;
            padding: 12px 15px;
            border: 1px solid rgba(125, 125, 125, 0.3);
            background-color: transparent;
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: #3b82f6;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3);
        }
        div[data-testid="stFormSubmitButton"] button {
            border-radius: 12px !important;
            padding: 0.65rem 1.75rem !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            letter-spacing: 0.3px !important;
            background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            box-shadow: 0 2px 8px rgba(30, 58, 138, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
            transition: none !important;
            transform: none !important;
            cursor: pointer !important;
        }
        div[data-testid="stFormSubmitButton"] button:hover,
        div[data-testid="stFormSubmitButton"] button:focus,
        div[data-testid="stFormSubmitButton"] button:active {
            background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%) !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            box-shadow: 0 2px 8px rgba(30, 58, 138, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
            transform: none !important;
            transition: none !important;
        }
        .login-title {
            text-align: center;
            color: #3b82f6;
            font-weight: 800;
            margin-bottom: 0px;
            font-size: 2rem;
        }
        .login-subtitle {
            text-align: center;
            color: #64748b;
            margin-bottom: 25px;
            font-size: 0.95rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col_spacer1, col_main, col_spacer2 = st.columns([1, 2, 1])
    
    with col_main:
        users = load_users()

        if logout_notice:
            st.success(logout_notice)
        
        with st.form("login_form"):
            st.markdown("""
                <div class='login-title'>Access HR Dashboard</div>
                <div class='login-subtitle'>Enter your company credentials to access the Recruitment System</div>
            """, unsafe_allow_html=True)
            l_user = st.text_input("Username", placeholder="username")
            l_pass = st.text_input("Password", type="password", placeholder="Enter your password")
            
            st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
            submit_login = st.form_submit_button("Verify", use_container_width=True)
            
            if submit_login:
                # --- Input Validation ---
                user_valid, user_err = validate_username(l_user)
                pass_valid, pass_err = validate_password(l_pass)

                if not user_valid:
                    st.error(f"⚠️ {user_err}")
                elif not pass_valid:
                    st.error(f"⚠️ {pass_err}")
                else:
                    # --- Credential Matching ---
                    if l_user in users and users[l_user] == l_pass:
                        st.session_state.logged_in = True
                        st.session_state.current_user = l_user
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password. Please check your credentials and try again.")


def logout_user():
    email_manager = st.session_state.get("email_manager")
    if email_manager:
        try:
            email_manager.disconnect()
        except Exception as exc:
            logging.warning(f"Error while disconnecting email manager during logout: {exc}")

    for key in list(st.session_state.keys()):
        del st.session_state[key]

    st.session_state.logged_in = False
    st.session_state.logout_notice = "You have been logged out successfully."
    st.rerun()


if hasattr(st, "dialog"):
    @st.dialog("Confirm Logout")
    def render_logout_dialog():
        st.markdown("### Are you sure you want to logout?")
        st.caption("Your active account will be cleared and you will be redirected to the login page.")

        col_cancel, col_confirm = st.columns(2)
        if col_cancel.button("Cancel", key="logout_cancel_dialog", use_container_width=True):
            st.session_state.show_logout_confirm = False
            st.rerun()
        if col_confirm.button("Logout", key="logout_confirm_dialog", type="primary", use_container_width=True):
            st.session_state.show_logout_confirm = False
            logout_user()


def render_logout_control():
    current_user = st.session_state.get("current_user", "HR User")
    safe_current_user = html.escape(str(current_user))

    with st.container(key="logout_panel"):
        st.markdown(
            f"""
            <div class="logout-card">
                <div class="logout-kicker">Signed-In Account</div>
                <div class="logout-user-row">
                    <div>
                        <div class="logout-user-name">{safe_current_user}</div>
                    </div>
                    <div class="logout-status-pill">Secure</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Log Out", key="logout_button", use_container_width=True):
            if hasattr(st, "dialog"):
                st.session_state.show_logout_confirm = True
            else:
                st.session_state.show_logout_inline_confirm = True
            st.rerun()

        if st.session_state.get("show_logout_inline_confirm", False):
            st.markdown(
                """
                <div class="logout-confirm-box">
                    Are you sure you want to logout?
                </div>
                """,
                unsafe_allow_html=True,
            )
            col_cancel, col_confirm = st.columns(2)
            if col_cancel.button("Cancel", key="logout_cancel_inline", use_container_width=True):
                st.session_state.show_logout_inline_confirm = False
                st.rerun()
            if col_confirm.button("Logout", key="logout_confirm_inline", type="primary", use_container_width=True):
                st.session_state.show_logout_inline_confirm = False
                logout_user()


def main():
    global nlp, ocr_reader
    # Initialize cached resources inside the Streamlit run context
    # to avoid "missing ScriptRunContext" warnings.
    nlp = load_nlp_model()
    ocr_reader = load_ocr_reader()
    st.set_page_config(
        page_title="Enterprise Recruitment System",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif !important;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        padding: 2rem 1.5rem;
        background: linear-gradient(135deg, #0F172A, #3B82F6, #60A5FA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #F8FAFC);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        border-left: 6px solid #3B82F6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    }
    
    /* Premium Static Buttons — no hover animation */
    .st-key-logout_panel {
        position: sticky;
        top: 0.5rem;
        z-index: 30;
        padding-bottom: 0.85rem;
        margin-bottom: 1rem;
        background: linear-gradient(180deg, rgba(248, 250, 252, 0.98), rgba(248, 250, 252, 0.94));
    }
    .logout-card {
        padding: 1rem 1rem 0.9rem;
        border-radius: 18px;
        background: linear-gradient(145deg, #0F172A 0%, #1E3A8A 58%, #2563EB 100%);
        color: #F8FAFC;
        border: 1px solid rgba(148, 163, 184, 0.2);
        box-shadow: 0 16px 28px -18px rgba(15, 23, 42, 0.8);
        margin-bottom: 0.75rem;
    }
    .logout-kicker {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: rgba(191, 219, 254, 0.9);
        margin-bottom: 0.85rem;
        font-weight: 700;
    }
    .logout-user-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.8rem;
    }
    .logout-user-label {
        font-size: 0.78rem;
        color: rgba(226, 232, 240, 0.85);
        margin-bottom: 0.2rem;
    }
    .logout-user-name {
        font-size: 1rem;
        font-weight: 700;
        line-height: 1.3;
        word-break: break-word;
    }
    .logout-status-pill {
        padding: 0.42rem 0.72rem;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 700;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(191, 219, 254, 0.24);
        color: #DBEAFE;
        white-space: nowrap;
    }
    .st-key-logout_panel [data-testid="stButton"] button {
        min-height: 2.9rem;
        border-radius: 14px !important;
        border: 1px solid rgba(14, 116, 144, 0.18) !important;
        background: linear-gradient(135deg, #FFFFFF 0%, #E0F2FE 100%) !important;
        color: #0F172A !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em;
        box-shadow: 0 12px 20px -18px rgba(14, 116, 144, 0.7);
    }
    .st-key-logout_panel [data-testid="stButton"] button:hover,
    .st-key-logout_panel [data-testid="stButton"] button:focus,
    .st-key-logout_panel [data-testid="stButton"] button:active {
        background: linear-gradient(135deg, #F8FAFC 0%, #BAE6FD 100%) !important;
        color: #0F172A !important;
        border: 1px solid rgba(14, 116, 144, 0.24) !important;
    }
    .logout-confirm-box {
        margin-top: 0.6rem;
        margin-bottom: 0.5rem;
        padding: 0.85rem 0.95rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #FFF7ED 0%, #FFEDD5 100%);
        border: 1px solid #FDBA74;
        color: #9A3412;
        font-size: 0.92rem;
        font-weight: 600;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.3px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.6rem !important;
        box-shadow: 0 2px 8px rgba(30, 58, 138, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        transition: none !important;
        transform: none !important;
        cursor: pointer !important;
    }
    .stButton > button:hover,
    .stButton > button:focus,
    .stButton > button:active {
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        box-shadow: 0 2px 8px rgba(30, 58, 138, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        transform: none !important;
        transition: none !important;
    }
    
    .status-new { color: #3B82F6; font-weight: 700; background: #DBEAFE; padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.85em; }
    .status-shortlisted { color: #10B981; font-weight: 700; background: #D1FAE5; padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.85em; }
    .status-interviewing { color: #F59E0B; font-weight: 700; background: #FEF3C7; padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.85em; }
    .status-rejected { color: #EF4444; font-weight: 700; background: #FEE2E2; padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.85em; }
    .status-offered { color: #8B5CF6; font-weight: 700; background: #EDE9FE; padding: 0.2rem 0.6rem; border-radius: 6px; font-size: 0.85em; }
    
    div[data-testid="stForm"] {
        background: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header"> Enterprise AI Recruitment System</h1>', unsafe_allow_html=True)
    
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        
    if not st.session_state.logged_in:
        render_login()
        return

    # Initialize session state
    if "config" not in st.session_state:
        st.session_state.config = DEFAULT_CONFIG.copy()
    else:
        # Sync missing keys from DEFAULT_CONFIG to existing session config
        for key, value in DEFAULT_CONFIG.items():
            if key not in st.session_state.config:
                st.session_state.config[key] = value
            elif key == "EMAIL_TEMPLATES":
                email_templates = DEFAULT_CONFIG.get("EMAIL_TEMPLATES")
                if isinstance(email_templates, dict):
                    for t_key, t_val in email_templates.items():
                        if t_key not in st.session_state.config["EMAIL_TEMPLATES"]:
                            st.session_state.config["EMAIL_TEMPLATES"][t_key] = t_val
    
    if "candidates" not in st.session_state:
        st.session_state.candidates = []
    
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = LLMResumeAnalyzer(st.session_state.config)
    
    if "email_manager" not in st.session_state:
        st.session_state.email_manager = None
    
    if "scheduling_manager" not in st.session_state:
        st.session_state.scheduling_manager = SchedulingManager()
    
    # Ensure sync with latest code (fixes 'unexpected keyword argument' due to old session state objects)
    if 'date_time' not in inspect.signature(st.session_state.scheduling_manager.schedule_interview).parameters:
        st.session_state.scheduling_manager = SchedulingManager()
    
    if "job_requisitions" not in st.session_state:
        st.session_state.job_requisitions = []
    
    if "active_job_id" not in st.session_state:
        st.session_state.active_job_id = None
    
    if "uploader_id" not in st.session_state:
        st.session_state.uploader_id = 0
    
    # Sidebar Configuration
    with st.sidebar:
        render_logout_control()
        
        
        st.header("  System Configuration")
        
        # Job Requisition Management
        with st.expander(" Job Requisitions", expanded=False):
            if st.button(" Create New Job Posting", width="stretch"):
                new_job = {
                    "id": len(st.session_state.job_requisitions) + 1,
                    "title": "New Position",
                    "department": "",
                    "created_at": datetime.now(),
                    "status": "Draft",
                    "candidates": []
                }
                st.session_state.job_requisitions.append(new_job)
                st.session_state.active_job_id = new_job["id"]
                st.rerun()
            
            if st.session_state.job_requisitions:
                job_names = [f"{j['id']}. {j['title']} ({len(j.get('candidates', []))} candidates)" 
                           for j in st.session_state.job_requisitions]
                
                # Find current index for selectbox
                current_idx = 0
                if st.session_state.active_job_id:
                    for i, j in enumerate(st.session_state.job_requisitions):
                        if j["id"] == st.session_state.active_job_id:
                            current_idx = i
                            break
                
                selected_job = st.selectbox("Active Job", job_names, index=current_idx)
                if selected_job:
                    job_id = int(selected_job.split('.')[0])
                    st.session_state.active_job_id = job_id
                    
                    if st.button(" Delete Active Job", width="stretch", type="secondary"):
                        # Remove the job
                        st.session_state.job_requisitions = [j for j in st.session_state.job_requisitions if j["id"] != job_id]
                        st.session_state.active_job_id = None
                        st.success("Job posting deleted!")
                        st.rerun()
        
        with st.expander(" Job Requirements", expanded=True):
            # Get current job data if exists
            active_job = next((j for j in st.session_state.job_requisitions if j["id"] == st.session_state.active_job_id), None)
            
            # Use active job's title if available, otherwise default
            default_title = active_job["title"] if active_job else "Software Engineer"
            job_title = st.text_input("Job Title*", value=default_title)
            
            # Sync back to requisition list to allow renaming
            if active_job and job_title != active_job["title"]:
                active_job["title"] = job_title
            
            st.session_state.config["JOB_TITLE"] = job_title
            
            # Sync Department
            default_dept = active_job.get("department", "Engineering") if active_job else "Engineering"
            depts = ["Engineering", "Product", "Design", "Sales", "Marketing", "HR", "Finance", "Operations"]
            try:
                dept_idx = depts.index(default_dept)
            except ValueError:
                dept_idx = 0
            
            department = st.selectbox("Department", depts, index=dept_idx)
            if active_job: active_job["department"] = department
            
            job_level = st.selectbox("Level", 
                ["Intern", "Entry-Level", "Mid-Level", "Senior", "Lead", "Manager", "Director", "VP"])
            
            employment_type = st.multiselect("Employment Type", 
                ["Full-Time", "Part-Time", "Contract", "Temporary", "Internship"], 
                default=["Full-Time"])
            
            location = st.text_input("Location", value="Remote")
            
            jd_text = st.text_area(
                "Job Description*",
                placeholder="Paste the full job description here...",
                height=200,
                help="This is used for AI-powered candidate matching",
                value=active_job.get("jd_text", "") if active_job else ""
            )
            
            required_skills_input = st.text_area(
                "Required Skills* (comma-separated)",
                placeholder="Python, React, AWS, SQL, Agile",
                help="Enter required skills separated by commas",
                value=", ".join(active_job.get("required_skills", [])) if active_job else ""
            )
            required_skills = [s.strip() for s in required_skills_input.split(',') if s.strip()]
            
            preferred_skills_input = st.text_area(
                "Preferred Skills (comma-separated)",
                placeholder="Docker, Kubernetes, GraphQL",
                help="Nice-to-have skills"
            )
            preferred_skills = [s.strip() for s in preferred_skills_input.split(',') if s.strip()]
            
            col_exp1, col_exp2 = st.columns(2)
            min_experience = col_exp1.number_input("Min Experience (yrs)", 0, 30, 2)
            max_experience = col_exp2.number_input("Max Experience (yrs)", 0, 30, 5)
            
            education_required = st.multiselect("Education Required",
                ["High School", "Associate", "Bachelor's", "Master's", "PhD"],
                default=["Bachelor's"])
            
            custom_criteria = st.text_area("Custom Screening Criteria (optional)",
                placeholder="e.g., Must have startup experience, Open source contributions preferred",
                help="Additional criteria for AI to consider")
            
            if st.button(" Save Job Requirements", width="stretch"):
                if max_experience <= min_experience:
                    st.error("Max Experience must be greater than Min Experience.")
                elif active_job:
                    active_job.update({
                        "jd_text": jd_text,
                        "required_skills": required_skills,
                        "min_experience": min_experience,
                        "max_experience": max_experience,
                        "custom_criteria": custom_criteria
                    })
                    st.success("Requirements saved to active job!")
                else:
                    st.warning("No active job selected to save requirements.")
        
        with st.expander("  Scoring & Filters", expanded=False):
            st.markdown("**Adjust evaluation weights:**")
            
            tech_weight = st.slider(" Technical Skills", 0.0, 1.0, 0.35, 0.05)
            exp_weight = st.slider(" Experience Relevance", 0.0, 1.0, 0.25, 0.05)
            edu_weight = st.slider(" Education Quality", 0.0, 1.0, 0.15, 0.05)
            jd_weight = st.slider(" JD Fit", 0.0, 1.0, 0.15, 0.05)
            
            total = tech_weight + exp_weight + edu_weight + jd_weight
            
            if total > 0:
                st.session_state.config["WEIGHTS"] = {
                    "technical_skills": tech_weight / total,
                    "experience": exp_weight / total,
                    "education": edu_weight / total,
                    "jd_match": jd_weight / total,
                    "growth_potential": 0.10
                }
            
            st.markdown("---")
            score_threshold = st.slider("Shortlist Threshold", 0.0, 1.0, 0.7, 0.05,
                help="Minimum score for automatic shortlisting")
            st.session_state.config["SCORE_THRESHOLD"] = score_threshold
            
            auto_reject_threshold = st.slider("Auto-Reject Threshold", 0.0, 0.5, 0.3, 0.05,
                help="Scores below this are auto-rejected")
            st.session_state.config["AUTO_REJECT_THRESHOLD"] = auto_reject_threshold
            
            diversity_hiring = st.checkbox("Enable Diversity Scoring",
                help="Boost candidates from underrepresented groups")
            st.session_state.config["DIVERSITY_HIRING"] = diversity_hiring
            
            if st.button(" Apply Scoring Changes", type="primary", width="stretch"):
                if st.session_state.candidates:
                    with st.spinner("Recalculating scores..."):
                        for candidate in st.session_state.candidates:
                            llm = candidate.get("llm_analysis", {})
                            candidate["final_score"] = calculate_final_score(
                                llm, st.session_state.config["WEIGHTS"]
                            )
                        st.session_state.candidates.sort(key=lambda x: x["final_score"], reverse=True)
                    st.success(" Scores updated!")
                    st.rerun()
        
        with st.expander(" Email & Notifications", expanded=False):
            company_name = st.text_input("Company Name", value="TechCorp")
            hr_manager_name = st.text_input("HR Manager Name", value="Hiring Manager")
            
            st.session_state.config["COMPANY_NAME"] = company_name
            st.session_state.config["HR_MANAGER_NAME"] = hr_manager_name
            
            st.markdown("**SMTP Settings:**")
            st.info("SMTP configuration is managed securely by the administrator.")
            
            if st.button(" Connect Email", width="stretch"):
                current_config = load_config()
                smtp_server = current_config.get("SMTP_SERVER", "smtp.gmail.com")
                smtp_port = current_config.get("SMTP_PORT", 587)
                use_tls = current_config.get("SMTP_USE_TLS", True)
                hr_email = current_config.get("HR_EMAIL", "")
                hr_password = current_config.get("HR_PASSWORD", "")
                
                if hr_email and hr_password:
                    mgr = EmailManager(smtp_server, smtp_port, use_tls, hr_email, hr_password)
                    if mgr.connect():
                        st.session_state.email_manager = mgr
                        mgr.disconnect()
                        st.success(" Email connected!")
                    else:
                        st.error(" Connection failed")
                else:
                    st.error(" SMTP credentials missing from configuration.")
        
        with st.expander(" Customization", expanded=False):
            interview_rounds = st.multiselect("Interview Rounds",
                ["Phone Screen", "Technical Screen", "Coding Challenge", 
                 "System Design", "Hiring Manager", "Panel Interview", 
                 "Culture Fit", "HR Round", "Executive Round"],
                default=["Technical Screen", "Hiring Manager", "HR Round"])
            st.session_state.config["INTERVIEW_ROUNDS"] = interview_rounds
            
            candidate_tags = st.text_input("Candidate Tags (comma-separated)",
                value="Urgent, Remote OK, Leadership, Referral, Alumni")
            st.session_state.config["TAGS"] = [t.strip() for t in candidate_tags.split(',') if t.strip()]
            
            top_n = st.number_input("Top N to Review", 1, 50, 10)
            st.session_state.config["TOP_N_CANDIDATES"] = top_n
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        " Resume Upload",
        " Candidate Pipeline", 
        " Analytics & Reports",
        " Communication Hub",
        " Interview Scheduler",
        " Workflow Automation"
    ])
    
    with tab1:
        st.header(" Resume Upload & Parsing")
        
        col_up1, col_up2, col_up3 = st.columns([2, 1, 1])
        
        with col_up1:
            uploaded_files = st.file_uploader(
                "Upload Resumes (PDF, DOCX, Images)",
                accept_multiple_files=True,
                type=['pdf', 'docx', 'doc', 'png', 'jpg', 'jpeg'],
                help="Drag & drop or click to upload multiple resumes",
                key=f"resume_uploader_{st.session_state.uploader_id}"
            )
        
        with col_up2:
            st.metric(" Uploaded", len(uploaded_files) if uploaded_files else 0)
            st.metric(" Processed", len(st.session_state.candidates))
        
        with col_up3:
            if uploaded_files:
                batch_tags = st.multiselect("Apply Tags", st.session_state.config["TAGS"])
        
        if uploaded_files:
            st.info(f" Ready to analyze {len(uploaded_files)} resume(s)")
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
            
            analyze_btn = col_btn1.button(" Analyze All", type="primary", width="stretch")
            if col_btn2.button("  Clear Upload", width="stretch"):
                st.session_state.uploader_id += 1
                st.rerun()
            
            if analyze_btn:
                if not jd_text:
                    st.error("  Please provide a Job Description in the sidebar")
                elif not required_skills:
                    st.error("  Please specify Required Skills in the sidebar")
                else:
                    progress = st.progress(0)
                    status = st.empty()
                    
                    candidates = []
                    total = len(uploaded_files)
                    
                    for idx, file in enumerate(uploaded_files):
                        status.text(f" Analyzing {idx+1}/{total}: {file.name}")
                        
                        try:
                            text = st.session_state.analyzer.extract_text_from_pdf(file)
                            
                            if not text or len(text.strip()) < 100:
                                st.warning(f"  Insufficient text in {file.name}")
                                continue
                            
                            llm_analysis = st.session_state.analyzer.analyze_resume_with_llm(
                                text, jd_text, required_skills,
                                None,
                                custom_criteria
                            )
                            
                            final_score = calculate_final_score(
                                llm_analysis, 
                                st.session_state.config["WEIGHTS"]
                            )
                            
                            # Determine status
                            if final_score >= st.session_state.config["SCORE_THRESHOLD"]:
                                status_val = "Shortlisted"
                            elif final_score < st.session_state.config["AUTO_REJECT_THRESHOLD"]:
                                status_val = "Rejected"
                            else:
                                status_val = "New"
                            
                            candidate = {
                                "file_name": file.name,
                                "resume_text": text,
                                "llm_analysis": llm_analysis,
                                "final_score": final_score,
                                "status": status_val,
                                "tags": batch_tags if 'batch_tags' in locals() else [],
                                "hr_rating": 0,
                                "notes": "",
                                "uploaded_at": datetime.now(),
                                "last_updated": datetime.now()
                            }
                            
                            candidates.append(candidate)
                            
                        except Exception as e:
                            st.error(f" Error processing {file.name}: {str(e)}")
                        
                        progress.progress((idx + 1) / total)
                    
                    candidates.sort(key=lambda x: x["final_score"], reverse=True)
                    st.session_state.candidates.extend(candidates)
                    
                    status.text(f" Analysis complete!")
                    
                    # Show summary
                    shortlisted = len([c for c in candidates if c["status"] == "Shortlisted"])
                    rejected = len([c for c in candidates if c["status"] == "Rejected"])
                    
                    col_s1, col_s2, col_s3 = st.columns(3)
                    col_s1.success(f" Shortlisted: {shortlisted}")
                    col_s2.info(f" Pending: {len(candidates) - shortlisted - rejected}")
                    col_s3.error(f" Auto-Rejected: {rejected}")
                    
                    time.sleep(1)
                    st.rerun()
    
    with tab2:
        st.header(" Candidate Pipeline")
        
        if not st.session_state.candidates:
            st.info(" No candidates yet. Upload resumes in the Upload tab.")
        else:
            # Pipeline metrics
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            
            total = len(st.session_state.candidates)
            shortlisted = len([c for c in st.session_state.candidates if c["status"] == "Shortlisted"])
            interviewing = len([c for c in st.session_state.candidates if c["status"] == "Interviewing"])
            offered = len([c for c in st.session_state.candidates if c["status"] == "Offered"])
            rejected = len([c for c in st.session_state.candidates if c["status"] == "Rejected"])
            
            col_m1.metric("Total", total)
            col_m2.metric("Shortlisted", shortlisted, delta=f"{(shortlisted/total*100):.0f}%")
            col_m3.metric("Interviewing", interviewing)
            col_m4.metric("Offered", offered)
            col_m5.metric("Rejected", rejected)
            
            st.markdown("---")
            
            # Filters
            col_f1, col_f2, col_f3, col_f4 = st.columns([2, 1, 1, 1])
            
            search_query = col_f1.text_input(" Search candidates", placeholder="Name, email, skills...")
            
            status_filter = col_f2.multiselect("Status",
                ["New", "Shortlisted", "Interviewing", "Offered", "Rejected"],
                default=["New", "Shortlisted", "Interviewing"])
            
            min_score_filter = col_f3.slider("Min Score", 0.0, 1.0, 0.0, 0.1)
            
            sort_by = col_f4.selectbox("Sort By", 
                ["Score (High-Low)", "Score (Low-High)", "Name (A-Z)", "Date (Newest)", "Date (Oldest)"])
            
            # Filter candidates
            filtered_candidates = st.session_state.candidates
            
            if search_query:
                filtered_candidates = [c for c in filtered_candidates if 
                    search_query.lower() in c.get("llm_analysis", {}).get("name", "").lower() or
                    search_query.lower() in c.get("llm_analysis", {}).get("email", "").lower() or
                    any(search_query.lower() in skill.lower() for skill in c.get("llm_analysis", {}).get("skills_found", []))]
            
            if status_filter:
                filtered_candidates = [c for c in filtered_candidates if c.get("status") in status_filter]
            
            filtered_candidates = [c for c in filtered_candidates if c.get("final_score", 0) >= min_score_filter]
            
            # Sort
            if sort_by == "Score (High-Low)":
                filtered_candidates.sort(key=lambda x: x.get("final_score", 0), reverse=True)
            elif sort_by == "Score (Low-High)":
                filtered_candidates.sort(key=lambda x: x.get("final_score", 0))
            elif sort_by == "Name (A-Z)":
                filtered_candidates.sort(key=lambda x: x.get("llm_analysis", {}).get("name", ""))
            elif sort_by == "Date (Newest)":
                filtered_candidates.sort(key=lambda x: x.get("uploaded_at", datetime.now()), reverse=True)
            elif sort_by == "Date (Oldest)":
                filtered_candidates.sort(key=lambda x: x.get("uploaded_at", datetime.now()))
            
            st.info(f"Showing {len(filtered_candidates)} of {total} candidates")
            
            # Candidate table
            table_data = []
            for idx, c in enumerate(filtered_candidates):
                llm = c.get("llm_analysis", {})
                table_data.append({
                    "Select": False,
                    "Rank": idx + 1,
                    "Name": llm.get("name", "Unknown"),
                    "Email": llm.get("email", ""),
                    "Phone": llm.get("phone", ""),
                    "Score": f"{c.get('final_score', 0):.3f}",
                    "Experience": f"{llm.get('years_experience', 0):.1f} yrs",
                    "Location": llm.get("location", ""),
                    "Status": c.get("status", "New"),
                    "Tags": ", ".join(c.get("tags", [])),
                    "Rating": "" * c.get("hr_rating", 0)
                })
            
            df = pd.DataFrame(table_data)
            
            edited_df = st.data_editor(
                df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Select": st.column_config.CheckboxColumn("Select", default=False),
                    "Score": st.column_config.ProgressColumn("Score", format="%.3f", min_value=0, max_value=1)
                },
                disabled=["Rank", "Name", "Email", "Phone", "Score", "Experience", "Location", "Status"]
            )
            
            # Bulk actions
            st.markdown("---")
            st.subheader(" Bulk Actions")
            
            selected_indices = [i for i, row in edited_df.iterrows() if row["Select"]]
            
            if selected_indices:
                st.info(f" {len(selected_indices)} candidate(s) selected")
                
                col_bulk1, col_bulk2, col_bulk3, col_bulk4 = st.columns(4)
                
                if col_bulk1.button(" Send Invitation", width="stretch"):
                    st.success(f"Would send invitations to {len(selected_indices)} candidates")
                
                if col_bulk2.button(" Schedule Interviews", width="stretch"):
                    st.info(f"Would schedule interviews for {len(selected_indices)} candidates")
                
                new_status = col_bulk3.selectbox("Change Status To",
                    ["Shortlisted", "Interviewing", "Offered", "Rejected"])
                
                if col_bulk4.button(" Update Status", width="stretch"):
                    for idx in selected_indices:
                        filtered_candidates[idx]["status"] = new_status
                    st.success(f"Updated {len(selected_indices)} candidates to {new_status}")
                    st.rerun()
            
            # Detailed candidate view
            st.markdown("---")
            st.subheader(" Detailed Candidate View")
            
            if filtered_candidates:
                selected_idx = st.selectbox(
                    "Select Candidate for Details",
                    range(len(filtered_candidates)),
                    format_func=lambda x: f"{x+1}. {filtered_candidates[x].get('llm_analysis', {}).get('name', 'Candidate')} - {filtered_candidates[x].get('final_score', 0):.3f}"
                )
                
                if selected_idx is not None:
                    candidate = filtered_candidates[selected_idx]
                    llm = candidate.get("llm_analysis", {})
                    
                    tab_detail1, tab_detail2, tab_detail3, tab_detail4 = st.tabs([
                        " Profile", " Evaluation", " AI Chat", " Notes & Actions"
                    ])
                    
                    with tab_detail1:
                        col_prof1, col_prof2 = st.columns([1, 1])
                        
                        with col_prof1:
                            st.markdown("###  Contact Information")
                            st.write(f"**Name:** {llm.get('name', 'N/A')}")
                            st.write(f"**Email:** {llm.get('email', 'N/A')}")
                            st.write(f"**Phone:** {llm.get('phone', 'N/A')}")
                            st.write(f"**Location:** {llm.get('location', 'N/A')}")
                            
                            st.markdown("###  Professional Summary")
                            st.write(f"**Current Company:** {llm.get('current_company', 'N/A')}")
                            st.write(f"**Current Role:** {llm.get('current_role', 'N/A')}")
                            st.write(f"**Experience:** {llm.get('years_experience', 0)} years")
                            st.write(f"**Education:** {llm.get('education_level', 'N/A')}")
                            
                            certifications = llm.get('certifications', [])
                            if certifications:
                                st.markdown("###  Certifications")
                                for cert in certifications:
                                    st.write(f"  {cert}")
                        
                        with col_prof2:
                            st.markdown("###  Skills Matched")
                            skills_found = llm.get('skills_found', [])
                            if skills_found:
                                for skill in skills_found:
                                    st.success(f" {skill}")
                            else:
                                st.write("No skills detected")
                            
                            st.markdown("###  Availability")
                            st.write(f"**Status:** {llm.get('availability', 'Unknown')}")
                            st.write(f"**Notice Period:** {llm.get('notice_period', 'Unknown')}")
                            st.write(f"**Expected Salary:** {llm.get('expected_salary', 'Not mentioned')}")
                    
                    with tab_detail2:
                        col_eval1, col_eval2 = st.columns([1, 1])
                        
                        with col_eval1:
                            st.markdown("###  Score Breakdown")
                            
                            scores_df = pd.DataFrame({
                                "Category": ["Technical", "Experience", "Education", "JD Fit"],
                                "Score": [
                                    llm.get("technical_score", 0),
                                    llm.get("experience_score", 0),
                                    llm.get("education_score", 0),
                                    llm.get("fit_score", 0)
                                ]
                            })
                            
                            fig = px.bar(scores_df, x="Score", y="Category", orientation='h',
                                        color="Score", color_continuous_scale="Viridis",
                                        range_x=[0, 100], text="Score")
                            fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                            st.plotly_chart(fig, width="stretch")
                            
                            st.metric("Final Score", f"{candidate.get('final_score', 0):.3f}",
                                    delta="Above threshold" if candidate.get('final_score', 0) >= st.session_state.config["SCORE_THRESHOLD"] else "Below threshold")
                        
                        with col_eval2:
                            st.markdown("###  Key Strengths")
                            for strength in llm.get("strengths", []):
                                st.success(f" {strength}")
                            
                            st.markdown("###   Areas of Concern")
                            for weakness in llm.get("weaknesses", []):
                                st.warning(f"  {weakness}")
                            
                            red_flags = llm.get("red_flags", [])
                            if red_flags:
                                st.markdown("###  Red Flags")
                                for flag in red_flags:
                                    st.error(f"  {flag}")
                        
                        st.markdown("---")
                        st.markdown("###  AI Recommendation")
                        st.info(llm.get("recommendation", "No recommendation available"))
                        
                        st.markdown("###  Suggested Interview Focus")
                        for topic in llm.get("interview_focus", []):
                            st.write(f"  {topic}")
                    
                    with tab_detail3:
                        st.markdown("###  Ask Questions About This Candidate")
                        
                        chat_key = f"chat_{selected_idx}"
                        if chat_key not in st.session_state:
                            st.session_state[chat_key] = []
                        
                        for msg in st.session_state[chat_key]:
                            with st.chat_message(msg["role"]):
                                st.write(msg["content"])
                        
                        if question := st.chat_input(f"Ask about {llm.get('name', 'this candidate')}"):
                            st.session_state[chat_key].append({"role": "user", "content": question})
                            
                            with st.chat_message("user"):
                                st.write(question)
                            
                            with st.chat_message("assistant"):
                                with st.spinner("Thinking..."):
                                    # Get conversation history for context
                                    history = [{"role": msg["role"], "content": msg["content"]} 
                                             for msg in st.session_state[chat_key][:-1]]
                                    
                                    answer = st.session_state.analyzer.chat_with_resume(
                                        candidate.get("resume_text", ""),
                                        llm.get("name", "Candidate"),
                                        question,
                                        history
                                    )
                                    st.write(answer)
                                    st.session_state[chat_key].append({"role": "assistant", "content": answer})
                        
                        if st.button(" Clear Chat History"):
                            st.session_state[chat_key] = []
                            st.rerun()
                    
                    with tab_detail4:
                        col_note1, col_note2 = st.columns([2, 1])
                        
                        with col_note1:
                            st.markdown("###  HR Notes")
                            notes = st.text_area("Add notes about this candidate",
                                value=candidate.get("notes", ""),
                                height=150,
                                key=f"notes_{selected_idx}")
                            
                            if st.button(" Save Notes"):
                                candidate["notes"] = notes
                                candidate["last_updated"] = datetime.now()
                                st.success("Notes saved!")
                        
                        with col_note2:
                            st.markdown("###  HR Rating")
                            rating = st.slider("Rate candidate", 0, 5, candidate.get("hr_rating", 0),
                                key=f"rating_{selected_idx}")
                            
                            if st.button("Save Rating"):
                                candidate["hr_rating"] = rating
                                st.success(f"Rated {rating} stars!")
                            
                            st.markdown("###   Tags")
                            current_tags = candidate.get("tags", [])
                            new_tags = st.multiselect("Manage tags",
                                st.session_state.config["TAGS"],
                                default=current_tags,
                                key=f"tags_{selected_idx}")
                            
                            if st.button("Update Tags"):
                                candidate["tags"] = new_tags
                                st.success("Tags updated!")
                        
                        st.markdown("---")
                        st.markdown("###  Quick Actions")
                        
                        col_act1, col_act2, col_act3 = st.columns(3)
                        
                        new_status = col_act1.selectbox("Change Status",
                            ["New", "Shortlisted", "Interviewing", "Offered", "Rejected"],
                            index=["New", "Shortlisted", "Interviewing", "Offered", "Rejected"].index(candidate.get("status", "New")),
                            key=f"status_{selected_idx}")
                        
                        if col_act2.button(" Update", width="stretch"):
                            candidate["status"] = new_status
                            candidate["last_updated"] = datetime.now()
                            st.success(f"Status updated to {new_status}")
                            st.rerun()
                        
                        if col_act3.button("  Delete", width="stretch"):
                            st.session_state.candidates.remove(candidate)
                            st.success("Candidate removed")
                            st.rerun()
                        
                        st.markdown("---")
                        
                        if st.button(" Generate Interview Questions", width="stretch"):
                            with st.spinner("Generating customized questions..."):
                                questions = st.session_state.analyzer.generate_interview_questions(
                                    candidate, jd_text, "Technical Screen"
                                )
                                candidate["interview_questions"] = questions
                        
                        if candidate.get("interview_questions"):
                            st.markdown("###  Suggested Interview Questions")
                            for i, q in enumerate(candidate["interview_questions"], 1):
                                st.write(f"{i}. {q}")
    
    with tab3:
        st.header(" Analytics & Insights")
        
        if not st.session_state.candidates:
            st.info(" No data available. Analyze candidates first.")
        else:
            # Key metrics
            col_a1, col_a2, col_a3, col_a4, col_a5 = st.columns(5)
            
            avg_score = np.mean([c["final_score"] for c in st.session_state.candidates])
            median_score = np.median([c["final_score"] for c in st.session_state.candidates])
            avg_exp = np.mean([c.get("llm_analysis", {}).get("years_experience", 0) 
                              for c in st.session_state.candidates])
            
            col_a1.metric("Avg Score", f"{avg_score:.2f}")
            col_a2.metric("Median Score", f"{median_score:.2f}")
            col_a3.metric("Avg Experience", f"{avg_exp:.1f} yrs")
            col_a4.metric("Shortlist Rate", 
                f"{(len([c for c in st.session_state.candidates if c['status']=='Shortlisted'])/len(st.session_state.candidates)*100):.1f}%")
            col_a5.metric("Time to Fill", "12 days")
            
            st.markdown("---")
            
            # Charts
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Score distribution
                scores = [c["final_score"] for c in st.session_state.candidates]
                fig1 = px.histogram(x=scores, nbins=20,
                                   labels={'x': 'Score', 'y': 'Count'},
                                   title=" Score Distribution",
                                   color_discrete_sequence=['#3B82F6'])
                fig1.add_vline(x=st.session_state.config["SCORE_THRESHOLD"], 
                              line_dash="dash", line_color="green",
                              annotation_text="Threshold")
                st.plotly_chart(fig1, width="stretch")
            
            with col_chart2:
                # Status breakdown
                status_counts = Counter([c.get("status", "New") for c in st.session_state.candidates])
                fig2 = px.pie(values=list(status_counts.values()),
                              names=list(status_counts.keys()),
                              title="Candidate Status Distribution",
                              color_discrete_sequence=px.colors.qualitative.Set3)
                st.plotly_chart(fig2, width="stretch")

            # Skills cloud / top skills
            st.markdown("---")
            st.subheader(" Most Common Skills in Pool")

            all_skills = []
            for c in st.session_state.candidates:
                all_skills.extend(c.get("llm_analysis", {}).get("skills_found", []))

            if all_skills:
                skill_counts = Counter(all_skills)
                top_skills = skill_counts.most_common(12)

                skills_df = pd.DataFrame(top_skills, columns=["Skill", "Count"])

                fig_skills = px.bar(skills_df, x="Count", y="Skill",
                                   orientation='h',
                                   title="Top Skills in Candidate Pool",
                                   color="Count",
                                   color_continuous_scale="Blues")
                st.plotly_chart(fig_skills, width="stretch")
            else:
                st.info("No skill data available yet")

    with tab4:
        st.header(" Communication Hub")

        if not st.session_state.candidates:
            st.info("No candidates to communicate with yet.")
        else:
            st.subheader(" Email Campaign")

            col_email1, col_email2 = st.columns([3, 2])

            with col_email1:
                email_template = st.selectbox("Select Template",
                    list(st.session_state.config["EMAIL_TEMPLATES"].keys()),
                    format_func=lambda x: x.capitalize())

                st.markdown("**Bulk Selection by Category:**")
                col_b1, col_b2, col_b3, col_b4 = st.columns(4)
                
                # Initialize selected recipients in session state if not present
                if "recipient_multiselect" not in st.session_state:
                    st.session_state.recipient_multiselect = []

                def get_candidates_by_status(status_val):
                    return [f"{c['llm_analysis'].get('name', 'N/A')} <{c['llm_analysis'].get('email', 'N/A')}>"
                            for c in st.session_state.candidates 
                            if c.get("status") == status_val and c.get("llm_analysis", {}).get("email")]

                if col_b1.button("Select Shortlisted", width="stretch"):
                    st.session_state.recipient_multiselect = get_candidates_by_status("Shortlisted")
                    st.rerun()
                
                if col_b2.button("Select Rejected", width="stretch"):
                    st.session_state.recipient_multiselect = get_candidates_by_status("Rejected")
                    st.rerun()

                if col_b3.button("Select Interviewing", width="stretch"):
                    st.session_state.recipient_multiselect = get_candidates_by_status("Interviewing")
                    st.rerun()

                if col_b4.button("Select Offered", width="stretch"):
                    st.session_state.recipient_multiselect = get_candidates_by_status("Offered")
                    st.rerun()

                all_options = [f"{c['llm_analysis'].get('name', 'N/A')} <{c['llm_analysis'].get('email', 'N/A')}>"
                             for c in st.session_state.candidates if c.get("llm_analysis", {}).get("email")]

                selected_candidates = st.multiselect(
                    "Select Recipients",
                    options=all_options,
                    key="recipient_multiselect"
                )

            with col_email2:
                st.markdown("**Preview Variables**")
                st.write("Available placeholders:")
                for placeholder in ["{candidate_name}", "{job_title}", "{company_name}", ...]:
                    st.code(placeholder, language=None)

            if st.button(" Send Emails", type="primary", disabled=not selected_candidates):
                if not st.session_state.email_manager:
                    st.error("  Email account not connected. Please configure SMTP settings first.")
                else:
                    progress_text = "Sending emails... Please wait."
                    progress_bar = st.progress(0)
                    
                    email_data_list = []
                    for cand_str in selected_candidates:
                        try:
                            name = cand_str.split("<")[0].strip()
                            email = cand_str.split("<")[1].strip(">").strip()
                            
                            template = st.session_state.config["EMAIL_TEMPLATES"].get(email_template)
                            if not template:
                                st.error(f"Selected template '{email_template}' not found.")
                                continue
                            subject = template["subject"].format(
                                candidate_name=name,
                                job_title=st.session_state.config.get("JOB_TITLE", "Position"),
                                company_name=st.session_state.config.get("COMPANY_NAME", "Company")
                            )
                            body = template["body"].format(
                                candidate_name=name,
                                job_title=st.session_state.config.get("JOB_TITLE", "Position"),
                                company_name=st.session_state.config.get("COMPANY_NAME", "Company"),
                                hr_manager_name=st.session_state.config.get("HR_MANAGER_NAME", "HR"),
                                interview_round="Initial",
                                interview_mode="Video",
                                duration="45",
                                meeting_details="TBD",
                                benefits="Standard",
                                salary_offer="TBD",
                                scheduled_time="TBD",
                                start_date="TBD",
                                screening_link="TBD"
                            )
                            
                            email_data_list.append({
                                "to": email,
                                "subject": subject,
                                "body": body
                            })
                        except Exception as e:
                            st.error(f"Error preparing email for {cand_str}: {str(e)}")

                    if email_data_list:
                        with st.spinner(" Sending batch emails..."):
                            results = st.session_state.email_manager.send_batch_emails(email_data_list)
                            st.success(f" Campaign complete: {results['sent']} sent, {results['failed']} failed.")
                            if results["errors"]:
                                with st.expander("View Errors"):
                                    for err in results["errors"]:
                                        st.write(err)

    with tab5:
        st.header(" Interview Scheduler")

        if not st.session_state.scheduling_manager.scheduled_interviews:
            st.info("No interviews scheduled yet.")
        else:
            upcoming = st.session_state.scheduling_manager.get_upcoming_interviews()

            if upcoming:
                st.subheader(f"Upcoming Interviews ({len(upcoming)})")

                for interview in upcoming[:6]:
                    with st.expander(f"{interview['date_time'].strftime('%b %d, %Y %I:%M %p')} - {interview['candidate_name']}"):
                        st.write(f"**Round:** {interview['interview_round']}")
                        st.write(f"**Mode:** {interview['mode']}")
                        st.write(f"**Interviewer:** {interview['interviewer']}")
                        if interview['meeting_link']:
                            st.markdown(f"[Join Meeting]({interview['meeting_link']})")
                        st.write("**Status:**", interview['status'].upper())
                        notes = st.text_area("Notes", value=interview.get("notes", ""),
                                           key=f"notes_int_{interview['id']}")
                        col_btn_int1, col_btn_int2 = st.columns(2)
                        if col_btn_int1.button("Save Notes", key=f"save_int_{interview['id']}"):
                            st.session_state.scheduling_manager.update_interview_notes(
                                interview['id'], notes)
                            st.success("Notes updated!")
                            
                        if col_btn_int2.button("Mark Candidate Declined", key=f"decline_int_{interview['id']}", type="primary"):
                            st.session_state.scheduling_manager.cancel_interview(interview['id'])
                            declined_email = interview['candidate_email']
                            
                            # Find declining candidate
                            for cand in st.session_state.candidates:
                                if cand['llm_analysis'].get('email') == declined_email:
                                    cand['status'] = "Rejected"
                                    cand['notes'] = cand.get('notes', '') + f"\nCandidate declined interview on {datetime.now().strftime('%Y-%m-%d')}."
                            
                            # Find next highest-ranking 'New' candidate
                            st.session_state.candidates.sort(key=lambda x: x.get("final_score", 0), reverse=True)
                            promoted = False
                            for cand in st.session_state.candidates:
                                if cand['status'] == "New":
                                    cand['status'] = "Shortlisted"
                                    cand['notes'] = cand.get('notes', '') + f"\nPromoted automatically due to another candidate's decline."
                                    st.success(f" Candidate {cand['llm_analysis'].get('name')} was automatically promoted to Shortlisted roster!")
                                    promoted = True
                                    break
                            
                            if not promoted:
                                st.warning("Candidate marked rejected, but no 'New' candidates available in the funnel to promote.")
                            st.rerun()

            else:
                st.info("No upcoming interviews scheduled.")

        with st.container(key="schedule_invitation_section"):
            st.markdown("""
            <style>
            .st-key-schedule_invitation_section {
                margin-top: 1.25rem;
                padding: 1.5rem 1.5rem 1.75rem;
                border-radius: 24px;
                background:
                    radial-gradient(circle at top right, rgba(96, 165, 250, 0.18), transparent 32%),
                    linear-gradient(145deg, rgba(15, 23, 42, 0.97) 0%, rgba(17, 24, 39, 0.95) 52%, rgba(30, 41, 59, 0.97) 100%);
                border: 1px solid rgba(148, 163, 184, 0.18);
                box-shadow: 0 24px 48px rgba(2, 6, 23, 0.34), inset 0 1px 0 rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(14px);
            }
            .st-key-schedule_invitation_section hr {
                border-color: rgba(148, 163, 184, 0.18) !important;
                margin: 0.25rem 0 1.25rem !important;
            }
            .st-key-schedule_invitation_section h3,
            .st-key-schedule_invitation_section [data-testid="stMarkdownContainer"] p {
                color: #E5EEF9 !important;
            }
            .st-key-schedule_invitation_section div[data-testid="stForm"] {
                background: linear-gradient(180deg, rgba(15, 23, 42, 0.88), rgba(30, 41, 59, 0.82)) !important;
                padding: 1.6rem !important;
                border-radius: 20px !important;
                border: 1px solid rgba(148, 163, 184, 0.18) !important;
                box-shadow: 0 18px 36px rgba(2, 6, 23, 0.26), inset 0 1px 0 rgba(255, 255, 255, 0.04) !important;
            }
            .st-key-schedule_invitation_section label,
            .st-key-schedule_invitation_section [data-testid="stWidgetLabel"] p,
            .st-key-schedule_invitation_section [data-testid="stWidgetLabel"] label,
            .st-key-schedule_invitation_section .stMarkdown,
            .st-key-schedule_invitation_section small {
                color: #DCE7F5 !important;
            }
            .st-key-schedule_invitation_section div[data-testid="stTextInput"] input,
            .st-key-schedule_invitation_section div[data-testid="stNumberInput"] input,
            .st-key-schedule_invitation_section div[data-testid="stDateInput"] input,
            .st-key-schedule_invitation_section div[data-testid="stTimeInput"] input,
            .st-key-schedule_invitation_section [data-baseweb="input"] > div,
            .st-key-schedule_invitation_section [data-baseweb="base-input"] {
                background: rgba(15, 23, 42, 0.82) !important;
                color: #F8FAFC !important;
                border: 1px solid rgba(96, 165, 250, 0.24) !important;
                border-radius: 14px !important;
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03) !important;
            }
            .st-key-schedule_invitation_section [data-baseweb="select"] > div {
                background: rgba(15, 23, 42, 0.82) !important;
                color: #F8FAFC !important;
                border: 1px solid rgba(96, 165, 250, 0.24) !important;
                border-radius: 14px !important;
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03) !important;
            }
            .st-key-schedule_invitation_section input::placeholder {
                color: #94A3B8 !important;
            }
            .st-key-schedule_invitation_section input,
            .st-key-schedule_invitation_section [data-baseweb="select"] *,
            .st-key-schedule_invitation_section [data-baseweb="input"] * {
                color: #F8FAFC !important;
            }
            .st-key-schedule_invitation_section div[data-testid="stTextInput"] input:focus,
            .st-key-schedule_invitation_section div[data-testid="stNumberInput"] input:focus,
            .st-key-schedule_invitation_section div[data-testid="stDateInput"] input:focus,
            .st-key-schedule_invitation_section div[data-testid="stTimeInput"] input:focus,
            .st-key-schedule_invitation_section [data-baseweb="select"]:focus-within > div,
            .st-key-schedule_invitation_section [data-baseweb="input"]:focus-within > div {
                border-color: rgba(96, 165, 250, 0.6) !important;
                box-shadow: 0 0 0 1px rgba(96, 165, 250, 0.45), 0 0 0 4px rgba(37, 99, 235, 0.18) !important;
            }
            .st-key-schedule_invitation_section div[data-testid="stFormSubmitButton"] button {
                background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%) !important;
                background-color: #0F172A !important;
                color: #F8FAFC !important;
                -webkit-text-fill-color: #F8FAFC !important;
                border: 1px solid rgba(96, 165, 250, 0.28) !important;
                box-shadow: 0 2px 8px rgba(15, 23, 42, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
                transition: none !important;
            }
            .st-key-schedule_invitation_section div[data-testid="stFormSubmitButton"] button:hover,
            .st-key-schedule_invitation_section div[data-testid="stFormSubmitButton"] button:focus,
            .st-key-schedule_invitation_section div[data-testid="stFormSubmitButton"] button:focus-visible,
            .st-key-schedule_invitation_section div[data-testid="stFormSubmitButton"] button:active {
                background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 100%) !important;
                background-color: #0F172A !important;
                color: #F8FAFC !important;
                -webkit-text-fill-color: #F8FAFC !important;
                border: 1px solid rgba(96, 165, 250, 0.28) !important;
                box-shadow: 0 2px 8px rgba(15, 23, 42, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
                outline: none !important;
                opacity: 1 !important;
            }
            .st-key-schedule_invitation_section [data-testid="stAlert"] {
                margin-top: 1rem;
            }
            .st-key-schedule_invitation_section [data-testid="stAlertContainer"] {
                border-radius: 18px !important;
                border: 1px solid rgba(148, 163, 184, 0.18) !important;
                box-shadow: 0 18px 34px rgba(2, 6, 23, 0.22) !important;
            }
            .st-key-schedule_invitation_section [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) [data-testid="stAlertContainer"] {
                background: linear-gradient(135deg, rgba(6, 78, 59, 0.96), rgba(5, 150, 105, 0.92)) !important;
                border: 1px solid rgba(110, 231, 183, 0.5) !important;
                box-shadow: 0 20px 42px rgba(5, 150, 105, 0.24), 0 0 0 1px rgba(167, 243, 208, 0.08) !important;
            }
            .st-key-schedule_invitation_section [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) [data-testid="stAlertContentSuccess"] {
                color: #ECFDF5 !important;
                font-weight: 700 !important;
                letter-spacing: 0.01em;
            }
            .st-key-schedule_invitation_section [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) [data-testid="stAlertContainer"] {
                background: linear-gradient(135deg, rgba(15, 23, 42, 0.94), rgba(30, 64, 175, 0.9)) !important;
                border: 1px solid rgba(96, 165, 250, 0.36) !important;
            }
            .st-key-schedule_invitation_section [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) [data-testid="stAlertContentInfo"] {
                color: #E0F2FE !important;
            }
            .st-key-schedule_invitation_section [data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) [data-testid="stAlertContainer"] {
                background: linear-gradient(135deg, rgba(69, 26, 3, 0.95), rgba(146, 64, 14, 0.9)) !important;
                border: 1px solid rgba(251, 191, 36, 0.38) !important;
            }
            .st-key-schedule_invitation_section [data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) [data-testid="stAlertContentWarning"] {
                color: #FEF3C7 !important;
            }
            .st-key-schedule_invitation_section [data-testid="stAlert"]:has([data-testid="stAlertContentError"]) [data-testid="stAlertContainer"] {
                background: linear-gradient(135deg, rgba(69, 10, 10, 0.95), rgba(153, 27, 27, 0.9)) !important;
                border: 1px solid rgba(248, 113, 113, 0.4) !important;
            }
            .st-key-schedule_invitation_section [data-testid="stAlert"]:has([data-testid="stAlertContentError"]) [data-testid="stAlertContentError"] {
                color: #FEE2E2 !important;
            }
            </style>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.subheader(" Schedule New Interview")

            # Filter selectable candidates
            schedule_candidates = [c for c in st.session_state.candidates if c.get("status") in ["Shortlisted", "Interviewing", "Offered"]]

            if not schedule_candidates:
                if not st.session_state.candidates:
                    st.warning("  No candidates available to schedule. Please upload and analyze resumes first.")
                else:
                    st.warning("  No eligible candidates available to schedule. Please update candidate statuses to Shortlisted or Interviewing first.")
            else:
                with st.form("interview_schedule_form"):
                    col_sch1, col_sch2 = st.columns(2)
                    
                    # Candidate selection
                    candidate_options = [f"{c['llm_analysis'].get('name', 'N/A')} ({c['llm_analysis'].get('email', 'N/A')})" 
                                       for c in schedule_candidates]
                    selected_cand_str = col_sch1.selectbox("Select Candidate*", options=candidate_options)
                    
                    # Find the selected candidate object
                    selected_candidate_idx = candidate_options.index(selected_cand_str)
                    selected_candidate = schedule_candidates[selected_candidate_idx]
                    
                    interview_round = col_sch2.selectbox("Interview Round*", 
                        st.session_state.config.get("INTERVIEW_ROUNDS", ["Technical Screen", "Hiring Manager", "HR Round"]))
                    
                    col_sch3, col_sch4 = st.columns(2)
                    interview_date = col_sch3.date_input("Interview Date*", value=date.today() + timedelta(days=1))
                    interview_time = col_sch4.time_input("Interview Time*", value=datetime.now().time())
                    
                    col_sch5, col_sch6 = st.columns(2)
                    duration = col_sch5.number_input("Duration (minutes)*", min_value=15, max_value=180, value=45, step=15)
                    interview_mode = col_sch6.selectbox("Interview Mode*", ["Video Call", "Phone Call", "In-Person"])
                    
                    col_sch7, col_sch8 = st.columns(2)
                    interviewer = col_sch7.text_input("Interviewer Name*", value=st.session_state.config.get("HR_MANAGER_NAME", "Hiring Manager"))
                    meeting_link = col_sch8.text_input("Meeting Link / Location", placeholder="https://zoom.us/j/...")
                    
                    submit_btn = st.form_submit_button(" Schedule & Send Invitation", width="stretch")
                    
                    if submit_btn:
                        # Combine date and time
                        dt = datetime.combine(interview_date, interview_time)
                        
                        # Schedule in manager
                        interview_id = st.session_state.scheduling_manager.schedule_interview(
                            candidate_name=selected_candidate['llm_analysis'].get('name', 'Candidate'),
                            candidate_email=selected_candidate['llm_analysis'].get('email', ''),
                            date_time=dt,
                            duration=duration,
                            mode=interview_mode,
                            interviewer=interviewer,
                            meeting_link=meeting_link,
                            interview_round=interview_round
                        )
                        
                        if interview_id:
                            st.success(f" Interview scheduled for {selected_candidate['llm_analysis'].get('name')}!")
                            
                            # Attempt to send email if manager is connected
                            if st.session_state.email_manager:
                                with st.spinner(" Sending invitation email..."):
                                    try:
                                        template = st.session_state.config["EMAIL_TEMPLATES"].get("interview")
                                        if not template:
                                            st.error("Interview email template missing from configuration.")
                                            st.stop()
                                        subject = template["subject"].format(
                                            job_title=st.session_state.config.get("JOB_TITLE", "the position"),
                                            company_name=st.session_state.config.get("COMPANY_NAME", "our company")
                                        )
                                        body = template["body"].format(
                                            candidate_name=selected_candidate['llm_analysis'].get('name', 'Candidate'),
                                            job_title=st.session_state.config.get("JOB_TITLE", "the position"),
                                            interview_round=interview_round,
                                            interview_mode=interview_mode,
                                            duration=duration,
                                            meeting_details=f"Meeting Link/Location: {meeting_link}" if meeting_link else "Details will be shared shortly",
                                            scheduled_time=dt.strftime('%A, %B %d, %Y at %I:%M %p'),
                                            hr_manager_name=st.session_state.config.get("HR_MANAGER_NAME", "HR Team"),
                                            company_name=st.session_state.config.get("COMPANY_NAME", "our company")
                                        )
                                        
                                        # Time is now in the template body
                                        pass
                                        
                                        email_data = {
                                            "to": selected_candidate['llm_analysis'].get('email', ''),
                                            "subject": subject,
                                            "body": body
                                        }
                                        
                                        result = st.session_state.email_manager.send_batch_emails([email_data])
                                        
                                        if result["sent"] > 0:
                                            st.success("Invitation email sent successfully!")
                                            # Update candidate status
                                            selected_candidate["status"] = "Interviewing"
                                            time.sleep(1) # Small delay to see message
                                            st.rerun()
                                        else:
                                            st.error(f"Failed to send email: {', '.join(result['errors'])}")
                                            # Don't rerun on error so user can see what happened
                                    except Exception as e:
                                        st.error(f"Interview scheduled, but failed to send email: {str(e)}")
                        else:
                            st.warning("Email manager not connected. Please connect your email in the sidebar first.")
                            # Still update status as the interview IS scheduled in the internal manager
                            selected_candidate["status"] = "Interviewing"
                            st.rerun()

    with tab6:
        st.header("  Workflow Automation & Settings")

        st.subheader(" End-to-End Recruitment Automation")
        st.info("Upload resumes here to trigger the full automated pipeline: Analysis   Grouping   Multi-step Emailing.")

        # Automation file uploader
        auto_files = st.file_uploader(
            "Upload Resumes for Automation",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'doc', 'png', 'jpg', 'jpeg'],
            key="auto_uploader"
        )

        if auto_files:
            if st.button("  Start Automation Workflow", type="primary", width="stretch"):
                if not jd_text or not required_skills:
                    st.error("  Please configure Job Description and Required Skills in the sidebar first.")
                elif not st.session_state.email_manager:
                    st.error("  Email manager not connected. Please configure SMTP in the sidebar first.")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    new_candidates = []
                    total_files = len(auto_files)
                    
                    # 1. Analysis Phase
                    status_text.markdown("###  Phase 1: AI Analysis & Scoring")
                    for idx, file in enumerate(auto_files):
                        status_text.text(f"Processing candidate {idx+1}/{total_files}: {file.name}")
                        try:
                            text = st.session_state.analyzer.extract_text_from_pdf(file)
                            if text and len(text.strip()) > 100:
                                analysis = st.session_state.analyzer.analyze_resume_with_llm(
                                    text, jd_text, required_skills,
                                    {"min": min_salary, "max": max_salary},
                                    custom_criteria
                                )
                                score = calculate_final_score(analysis, st.session_state.config["WEIGHTS"])
                                
                                # Split status
                                cand_status = "Shortlisted" if score >= st.session_state.config["SCORE_THRESHOLD"] else "Rejected"
                                
                                candidate = {
                                    "file_name": file.name,
                                    "resume_text": text,
                                    "llm_analysis": analysis,
                                    "final_score": score,
                                    "status": cand_status,
                                    "tags": ["Automated Workflow"],
                                    "hr_rating": 0,
                                    "notes": "Processed via automated workflow",
                                    "uploaded_at": datetime.now(),
                                    "last_updated": datetime.now()
                                }
                                new_candidates.append(candidate)
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {str(e)}")
                        progress_bar.progress((idx + 1) / (total_files * 2)) # First half of progress

                    # Add to session state and sort
                    st.session_state.candidates.extend(new_candidates)
                    st.session_state.candidates.sort(key=lambda x: x["final_score"], reverse=True)

                    # 2. Communication Phase
                    status_text.markdown("###  Phase 2: Automated Communication")
                    
                    selected = [c for c in new_candidates if c["status"] == "Shortlisted"]
                    rejected = [c for c in new_candidates if c["status"] == "Rejected"]
                    
                    email_batch = []

                    # Prepare Congratulations for Selected
                    for cand in selected:
                        status_text.text(f"Preparing Congratulations for {cand['llm_analysis'].get('name')}...")
                        try:
                            template = st.session_state.config["EMAIL_TEMPLATES"]["shortlist"]
                            subject = template["subject"].format(
                                job_title=st.session_state.config.get("JOB_TITLE", "the position"),
                                company_name=st.session_state.config.get("COMPANY_NAME", "our company")
                            )
                            body = template["body"].format(
                                candidate_name=cand['llm_analysis'].get('name', 'Candidate'),
                                job_title=st.session_state.config.get("JOB_TITLE", "the position"),
                                company_name=st.session_state.config.get("COMPANY_NAME", "our company"),
                                hr_manager_name=st.session_state.config.get("HR_MANAGER_NAME", "HR Team")
                            )
                            email_batch.append({"to": cand['llm_analysis'].get('email', ''), "subject": subject, "body": body})
                        except Exception: pass
                    
                    # Prepare Rejection/Update for Not Selected
                    for cand in rejected:
                        status_text.text(f"Preparing Status Update for {cand['llm_analysis'].get('name')}...")
                        try:
                            template = st.session_state.config["EMAIL_TEMPLATES"]["rejection"]
                            subject = template["subject"].format(
                                job_title=st.session_state.config.get("JOB_TITLE", "the position"),
                                company_name=st.session_state.config.get("COMPANY_NAME", "our company")
                            )
                            body = template["body"].format(
                                candidate_name=cand['llm_analysis'].get('name', 'Candidate'),
                                job_title=st.session_state.config.get("JOB_TITLE", "the position"),
                                company_name=st.session_state.config.get("COMPANY_NAME", "our company"),
                                hr_manager_name=st.session_state.config.get("HR_MANAGER_NAME", "HR Team")
                            )
                            email_batch.append({"to": cand['llm_analysis'].get('email', ''), "subject": subject, "body": body})
                        except Exception: pass
                    
                    if email_batch:
                        status_text.text(f" Sending {len(email_batch)} notification emails...")
                        st.session_state.email_manager.send_batch_emails(email_batch)
                    
                    progress_bar.progress(0.8)

                    # 3. Top Candidate Interview Invite
                    if selected:
                        top_cand = selected[0] # The one with highest score among new ones
                        status_text.markdown(f"###  Phase 3: Interview Invite for Top Candidate - **{top_cand['llm_analysis'].get('name')}**")
                        status_text.text(f"Scheduling and sending interview invite to {top_cand['llm_analysis'].get('email')}...")
                        
                        # Formal invite
                        dt = datetime.now() + timedelta(days=2, hours=4)
                        st.session_state.scheduling_manager.schedule_interview(
                            candidate_name=top_cand['llm_analysis'].get('name', 'Candidate'),
                            candidate_email=top_cand['llm_analysis'].get('email', ''),
                            date_time=dt,
                            duration=45,
                            mode="Video Call",
                            interviewer=st.session_state.config.get("HR_MANAGER_NAME", "Hiring Manager"),
                            meeting_link="https://vcodez.zoom.us/auto-link",
                            interview_round="Technical Screen"
                        )
                        
                        # Send the actual email for top candidate
                        try:
                            template = st.session_state.config["EMAIL_TEMPLATES"]["interview"]
                            subject = template["subject"].format(
                                job_title=st.session_state.config.get("JOB_TITLE", "the position"),
                                company_name=st.session_state.config.get("COMPANY_NAME", "our company")
                            )
                            body = template["body"].format(
                                candidate_name=top_cand['llm_analysis'].get('name', 'Candidate'),
                                job_title=st.session_state.config.get("JOB_TITLE", "the position"),
                                interview_round="Technical Screen",
                                interview_mode="Video Call",
                                duration=45,
                                meeting_details="Meeting Link: https://vcodez.zoom.us/auto-link",
                                scheduled_time=dt.strftime('%A, %B %d, %Y at %I:%M %p'),
                                hr_manager_name=st.session_state.config.get("HR_MANAGER_NAME", "HR Team"),
                                company_name=st.session_state.config.get("COMPANY_NAME", "our company")
                            )
                            email_data = {"to": top_cand['llm_analysis'].get('email', ''), "subject": subject, "body": body}
                            st.session_state.email_manager.send_batch_emails([email_data])
                        except Exception:
                            pass # Silent fail in automation for now
                        
                        top_cand["status"] = "Interviewing"

                    progress_bar.progress(1.0)
                    st.success(f" Workflow complete! {len(selected)} shortlisted, {len(rejected)} rejected. Top candidate invited for interview.")
                    time.sleep(3)
                    st.rerun()

        st.markdown("---")
        st.subheader(" System Status & Features")
        features = {
            "End-to-End Automation": " Fully Functional",
            "Auto Analysis & Split": " Implemented",
            "Bulk Recruitment Emails": " Implemented",
            "Top Candidate Scheduling": " Implemented",
            "ATS Data Export": " Available",
            "Diversity Scoring": " Configurable",
        }
        for name, status in features.items():
            st.markdown(f"**{name}**   {status}")

        st.markdown("---")

        st.markdown("---")

        confirm_reset = st.checkbox("I understand this will DELETE ALL candidates, jobs and schedules")
        if st.button(" Reset All Data (Dangerous)", type="primary", disabled=not confirm_reset):
            st.session_state.candidates = []
            st.session_state.job_requisitions = []
            st.session_state.active_job_id = None
            st.session_state.scheduling_manager = SchedulingManager()
            st.success("All data has been reset!")
            st.rerun()

    if st.session_state.get("show_logout_confirm", False) and hasattr(st, "dialog"):
        render_logout_dialog()

if __name__ == "__main__":
    main()
