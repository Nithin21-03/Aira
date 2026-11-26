import os
import io
import json
import wave
import threading
import urllib.request
import urllib.error
import speech_recognition as sr
import av
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, Response, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AIRA Voice Assistant Server")

# Allow full CORS and Chrome Private Network Access (PNA) for browser access (including file:/// origins)
@app.middleware("http")
async def cors_and_pna_middleware(request, call_next):
    origin = request.headers.get("origin") or "*"
    # Handle CORS / Chrome Private Network Access preflights
    if request.method == "OPTIONS":
        response = Response(status_code=200)
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, HEAD, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        if "access-control-request-private-network" in request.headers:
            response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "index.html")
HTML_FILE = os.path.join(BASE_DIR, "AIRA_JARVIS_homepage (3).html")
MODEL_FILE = os.path.join(BASE_DIR, "viseme-model.bin")
MEMORY_FILE = os.path.join(BASE_DIR, "aira_memory.json")
memory_lock = threading.Lock()

def load_memory() -> dict:
    default_mem = {
        "user_name": None,
        "facts": [],
        "recent_chats": []
    }
    if not os.path.exists(MEMORY_FILE):
        return default_mem
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return {**default_mem, **data}
    except Exception as e:
        print(f"[AIRA Memory] Notice loading memory: {e}")
    return default_mem

def save_memory(mem: dict):
    with memory_lock:
        try:
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(mem, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[AIRA Memory] Notice saving memory: {e}")

def get_memory_prompt_summary() -> str:
    mem = load_memory()
    parts = []
    if mem.get("user_name"):
        parts.append(f"User's name is {mem['user_name']}.")
    facts = mem.get("facts", [])
    if facts:
        distinct = []
        for fact in reversed(facts):
            if fact not in distinct:
                distinct.append(fact)
            if len(distinct) >= 6:
                break
        distinct.reverse()
        parts.append("Learned facts: " + "; ".join(distinct))
    return " ".join(parts).strip()

def extract_and_update_memory(user_text: str, assistant_reply: str = None):
    t = (user_text or "").strip()
    if not t:
        return
    import re
    mem = load_memory()
    updated = False

    # Reset command
    if re.search(r'\b(forget everything|clear your memory|reset memory|forget my info)\b', t, re.IGNORECASE):
        mem = {"user_name": None, "facts": [], "recent_chats": []}
        save_memory(mem)
        print("[AIRA Memory] Memory cleared by user command.")
        return

    # Name extraction
    m_name = re.search(r'(?:my name is|call me|i am called)\s+([A-Z][a-z]+|[a-z]+)', t, re.IGNORECASE)
    if m_name and m_name.group(1).lower() not in ['a', 'an', 'the', 'not', 'here', 'ready', 'talking', 'listening', 'aira', 'jarvis']:
        name = m_name.group(1).strip().capitalize()
        if mem.get("user_name") != name:
            mem["user_name"] = name
            updated = True
            fact = f"User's name is {name}."
            if fact not in mem["facts"]:
                mem["facts"].append(fact)

    # Explicit remember command
    m_rem = re.search(r'(?:remember that|don\'t forget that|please remember)\s+(.*)', t, re.IGNORECASE)
    if m_rem:
        fact = m_rem.group(1).strip()
        if fact and fact not in mem["facts"]:
            mem["facts"].append(fact)
            updated = True

    # Favorites
    m_fav = re.search(r'(?:my favorite\s+([a-z\s]+?)\s+is\s+([^.!?]+))', t, re.IGNORECASE)
    if m_fav:
        fact = f"User's favorite {m_fav.group(1).strip()} is {m_fav.group(2).strip()}."
        if fact not in mem["facts"]:
            mem["facts"].append(fact)
            updated = True

    # Likes & hobbies
    m_like = re.search(r'(?:i (?:really )?(?:love|like|prefer|enjoy))\s+([^.!?]+)', t, re.IGNORECASE)
    if m_like and not m_rem:
        fact = f"User likes {m_like.group(1).strip()}."
        if fact not in mem["facts"]:
            mem["facts"].append(fact)
            updated = True

    # Projects & work
    m_work = re.search(r'(?:i am working on|i\'m building|i am building|i develop)\s+([^.!?]+)', t, re.IGNORECASE)
    if m_work:
        fact = f"User is working on {m_work.group(1).strip()}."
        if fact not in mem["facts"]:
            mem["facts"].append(fact)
            updated = True

    # Location
    m_live = re.search(r'(?:i live in|i am from|i\'m from)\s+([^.!?]+)', t, re.IGNORECASE)
    if m_live:
        fact = f"User lives in {m_live.group(1).strip()}."
        if fact not in mem["facts"]:
            mem["facts"].append(fact)
            updated = True

    # Record turn
    from datetime import datetime
    chat_entry = {"time": datetime.now().isoformat(), "user": t}
    if assistant_reply:
        chat_entry["assistant"] = assistant_reply
    if "recent_chats" not in mem:
        mem["recent_chats"] = []
    mem["recent_chats"].append(chat_entry)
    if len(mem["recent_chats"]) > 60:
        mem["recent_chats"] = mem["recent_chats"][-60:]
    updated = True

    if updated:
        save_memory(mem)
        print(f"[AIRA Memory] Saved memory. Name: {mem.get('user_name')}, Total facts: {len(mem.get('facts', []))}")

import urllib.parse

class ExecutiveDecisionEngine:
    """
    Model 2: Executive Decision-Making & Task Execution Engine.
    Handles decision making, task planning, and external actions (emails, notes, web, apps).
    """
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.tasks_file = os.path.join(base_dir, "aira_tasks.json")
        self.notes_file = os.path.join(base_dir, "aira_notes.json")
        self.lock = threading.Lock()

    def analyze_and_execute(self, user_prompt: str, user_name: str = None) -> dict:
        p = (user_prompt or "").strip()
        p_lower = p.lower()
        import re

        # 1. Email Task
        if any(k in p_lower for k in ["send email", "send an email", "send a mail", "send mail", "draft email", "compose email", "email to", "mail to"]):
            to_match = re.search(r'(?:to|email to|mail to)\s+([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+|[A-Z][a-z]+|[a-z]+)', p, re.IGNORECASE)
            recipient = to_match.group(1).strip() if to_match else "recipient"
            
            body_match = re.search(r'(?:saying|that|content|message|body)\s+(.*)', p, re.IGNORECASE)
            if body_match:
                body = body_match.group(1).strip()
            else:
                body = f"Hello {recipient},\n\nHope you are doing well.\n\nBest regards,\n{user_name or 'AIRA User'}"
            
            subj_match = re.search(r'(?:subject|about)\s+([^,.]+)', p, re.IGNORECASE)
            if subj_match:
                subject = subj_match.group(1).strip().capitalize()
            else:
                subject = f"Message regarding {body[:25].strip()}..."

            mailto_url = f"mailto:{recipient}?subject={urllib.parse.quote(subject)}&body={urllib.parse.quote(body)}"
            
            from datetime import datetime
            task_entry = {
                "id": f"task_email_{int(datetime.now().timestamp())}",
                "type": "email",
                "recipient": recipient,
                "subject": subject,
                "body": body,
                "mailto_url": mailto_url,
                "status": "prepared",
                "created_at": datetime.now().isoformat()
            }
            self._append_task(task_entry)
            
            return {
                "is_task": True,
                "task_type": "email",
                "details": task_entry,
                "feedback": f"I've drafted that email to {recipient} with the subject '{subject}'. It is ready to review and send!"
            }

        # 2. Note Creation Task
        if any(k in p_lower for k in ["take a note", "create a note", "note that", "save a note", "add a note"]):
            content_match = re.search(r'(?:note that|create a note that|take a note that|to|that)\s+(.*)', p, re.IGNORECASE)
            content = content_match.group(1).strip() if content_match else p
            from datetime import datetime
            note_entry = {
                "id": f"note_{int(datetime.now().timestamp())}",
                "type": "note",
                "content": content,
                "created_at": datetime.now().isoformat()
            }
            self._append_note(note_entry)
            return {
                "is_task": True,
                "task_type": "note",
                "details": note_entry,
                "feedback": f"I've saved that note for you: '{content}'."
            }

        # 3. Read Notes Task
        if any(k in p_lower for k in ["read my notes", "what are my notes", "show my notes", "get my notes", "check my notes"]):
            notes = self._load_notes()
            if notes:
                recent = [n.get("content", "") for n in notes[-3:]]
                return {
                    "is_task": True,
                    "task_type": "read_notes",
                    "details": {"type": "read_notes", "notes": notes},
                    "feedback": f"You have {len(notes)} saved notes. Recent ones: {'; '.join(recent)}."
                }
            return {
                "is_task": True,
                "task_type": "read_notes",
                "details": {"type": "read_notes", "notes": []},
                "feedback": "You don't have any notes saved yet."
            }

        # 4. Open Website / Desktop App Task
        sites = {
            "youtube": "https://www.youtube.com",
            "google": "https://www.google.com",
            "github": "https://www.github.com",
            "gmail": "https://mail.google.com",
            "twitter": "https://twitter.com",
            "reddit": "https://www.reddit.com",
            "linkedin": "https://www.linkedin.com"
        }
        for site_key, site_url in sites.items():
            if f"open {site_key}" in p_lower or f"launch {site_key}" in p_lower:
                return {
                    "is_task": True,
                    "task_type": "open_url",
                    "details": {"type": "open_url", "name": site_key.capitalize(), "url": site_url},
                    "feedback": f"Opening {site_key.capitalize()} for you now!"
                }

        # 5. Search Web Task
        if p_lower.startswith(("search for", "google for", "search web for", "look up")):
            q_match = re.search(r'(?:search for|google for|search web for|look up)\s+(.*)', p, re.IGNORECASE)
            query = q_match.group(1).strip() if q_match else p
            search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
            return {
                "is_task": True,
                "task_type": "web_search",
                "details": {"type": "web_search", "query": query, "url": search_url},
                "feedback": f"Searching the web for '{query}'."
            }

        return {"is_task": False}

    def _append_task(self, task):
        with self.lock:
            try:
                tasks = []
                if os.path.exists(self.tasks_file):
                    with open(self.tasks_file, "r", encoding="utf-8") as f:
                        tasks = json.load(f)
                tasks.append(task)
                with open(self.tasks_file, "w", encoding="utf-8") as f:
                    json.dump(tasks, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[Executive Engine] Notice saving task: {e}")

    def _append_note(self, note):
        with self.lock:
            try:
                notes = self._load_notes()
                notes.append(note)
                with open(self.notes_file, "w", encoding="utf-8") as f:
                    json.dump(notes, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[Executive Engine] Notice saving note: {e}")

    def _load_notes(self):
        try:
            if os.path.exists(self.notes_file):
                with open(self.notes_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
        except Exception:
            pass
        return []

    def load_tasks(self):
        try:
            if os.path.exists(self.tasks_file):
                with open(self.tasks_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
        except Exception:
            pass
        return []

    def load_notes(self):
        return self._load_notes()

executive_engine = ExecutiveDecisionEngine(BASE_DIR)

def get_hf_token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""

def is_ollama_alive():
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags", headers={"User-Agent": "AIRA"})
        with urllib.request.urlopen(req, timeout=1.0) as r:
            return r.status == 200
    except Exception:
        return False

def ensure_ollama():
    if is_ollama_alive():
        return True
    import subprocess
    ollama_path = r"C:\Users\nithu\AppData\Local\Programs\Ollama\ollama.exe"
    if os.path.exists(ollama_path):
        try:
            # Clear any hung GUI instances locking the log file
            subprocess.run(["taskkill", "/F", "/IM", "ollama app.exe"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            env = os.environ.copy()
            env["OLLAMA_ORIGINS"] = "*"
            # 0x08000000 = CREATE_NO_WINDOW
            subprocess.Popen([ollama_path, "serve"], env=env, creationflags=0x08000000)
            import time
            for _ in range(12):
                time.sleep(0.5)
                if is_ollama_alive():
                    return True
        except Exception as e:
            print(f"Error starting ollama: {e}")
    return False

@app.get("/")
def serve_index():
    for f in [INDEX_FILE, HTML_FILE]:
        if os.path.exists(f):
            return FileResponse(f, media_type="text/html")
    return {"message": "AIRA HTML file not found."}

@app.get("/viseme-model.bin")
def serve_model():
    if os.path.exists(MODEL_FILE):
        return FileResponse(MODEL_FILE, media_type="application/octet-stream")
    raise HTTPException(status_code=404, detail="Model file not found")

@app.get("/api/status")
def get_status():
    token = get_hf_token()
    return {
        "status": "ready",
        "has_token": bool(token),
        "stt_model": "openai/whisper-large-v3-turbo",
        "llm_model": "meta-llama/Llama-3.1-8B-Instruct",
        "tts_model": "facebook/mms-tts-eng"
    }

whisper_model_instance = None
def get_whisper():
    global whisper_model_instance
    if whisper_model_instance is None:
        try:
            import faster_whisper
            whisper_model_instance = faster_whisper.WhisperModel("tiny", device="cpu", compute_type="int8")
        except Exception as e:
            print("Notice: faster_whisper not loaded:", e)
    return whisper_model_instance

def decode_audio_to_wav(audio_bytes):
    # 1. Quick check if it's already a valid WAV
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            if wf.getnchannels() >= 1 and wf.getframerate() >= 8000:
                return audio_bytes
    except Exception:
        pass

    # 2. Resample / decode using PyAV (handles WebM, Opus, OGG, MP3, etc.)
    try:
        container = av.open(io.BytesIO(audio_bytes))
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        out_io = io.BytesIO()
        with wave.open(out_io, "wb") as wav_out:
            wav_out.setnchannels(1)
            wav_out.setsampwidth(2)
            wav_out.setframerate(16000)
            for frame in container.decode(audio=0):
                for resampled_frame in resampler.resample(frame):
                    wav_out.writeframes(resampled_frame.to_ndarray().tobytes())
        resampled_wav = out_io.getvalue()
        if len(resampled_wav) > 44:
            return resampled_wav
    except Exception as e:
        print(f"Audio decode notice: {e}")

    return audio_bytes

@app.post("/api/stt")
def speech_to_text(audio: UploadFile = File(...), token: str = Form(None), model: str = Form("openai/whisper-large-v3-turbo")):
    raw_bytes = audio.file.read()
    if not raw_bytes or len(raw_bytes) < 300:
        return {"text": ""}
    
    # Ensure audio is clean 16kHz PCM WAV for speech recognition
    audio_bytes = decode_audio_to_wav(raw_bytes)
    print(f"[AIRA STT] Received audio: {len(raw_bytes)} bytes, decoded to: {len(audio_bytes)} bytes")
    
    # 1. Primary: Local Faster-Whisper (OpenAI Whisper engine, offline, ultra-fast & accurate)
    try:
        wm = get_whisper()
        if wm:
            segments, _ = wm.transcribe(io.BytesIO(audio_bytes), language="en", beam_size=1, condition_on_previous_text=False)
            whisper_text = " ".join([s.text for s in segments]).strip()
            if not whisper_text:
                # If language was not pure English, try auto-detection
                segments, _ = wm.transcribe(io.BytesIO(audio_bytes), beam_size=1, condition_on_previous_text=False)
                whisper_text = " ".join([s.text for s in segments]).strip()
            if whisper_text:
                print(f"[AIRA Whisper STT] Recognized: \"{whisper_text}\"")
                return {"text": whisper_text, "source": "whisper"}
    except Exception as e:
        print(f"[AIRA Whisper STT] Notice: {e}")

    # 2. Fallback: Google Speech Recognition
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            if text and text.strip():
                print(f"[AIRA Google STT] Recognized: \"{text.strip()}\"")
                return {"text": text.strip(), "source": "google_speech"}
    except Exception:
        pass

    # 3. Try Hugging Face router if credit available
    auth_token = token or get_hf_token()
    if auth_token and not HF_CREDIT_EXHAUSTED:
        url = f"https://router.huggingface.co/hf-inference/models/{model}"
        req = urllib.request.Request(
            url,
            data=audio_bytes,
            headers={
                "Authorization": f"Bearer {auth_token}",
                "Content-Type": audio.content_type or "audio/wav",
                "User-Agent": "AIRA-Voice-Assistant"
            }
        )
        try:
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                recognized = data.get("text", "").strip()
                if recognized:
                    return {"text": recognized, "source": "huggingface"}
        except Exception:
            pass

    print("[AIRA STT] No speech detected in audio clip.")
    return {"text": "", "source": "none"}

def generate_dynamic_fallback(prompt_text: str) -> str:
    import random
    q = (prompt_text or "").strip().lower()
    
    # 0. Persistent Memory queries & commands
    mem = load_memory()
    if any(k in q for k in ["what is my name", "who am i", "do you know my name", "what's my name", "remember my name"]):
        if mem.get("user_name"):
            return f"Your name is {mem['user_name']}!"
        return "I don't know your name yet. What should I call you?"

    if any(k in q for k in ["what do you remember about me", "what do you know about me", "what did i tell you", "what have i told you", "what are my facts"]):
        facts = mem.get("facts", [])
        if facts:
            summary = "; ".join(facts[-4:])
            return f"Here is what I remember about you: {summary}"
        return "I don't have any specific facts saved about you yet. Tell me about yourself!"

    if any(k in q for k in ["forget everything", "clear your memory", "reset memory", "forget me"]):
        save_memory({"user_name": None, "facts": [], "recent_chats": []})
        return "I have cleared my memory. We can start completely fresh!"

    # 1. Math check
    import re
    math_match = re.search(r'(\d+(?:\.\d+)?)\s*([\+\-\*\/xX]|plus|minus|times|divided by)\s*(\d+(?:\.\d+)?)', q)
    if math_match:
        n1 = float(math_match.group(1))
        op = math_match.group(2).lower()
        n2 = float(math_match.group(3))
        if op in ('+', 'plus'): ans = n1 + n2
        elif op in ('-', 'minus'): ans = n1 - n2
        elif op in ('*', 'x', 'times'): ans = n1 * n2
        elif op in ('/', 'divided by'): ans = (n1 / n2) if n2 != 0 else 'infinity'
        else: ans = 0
        return f"That comes out to {int(ans) if isinstance(ans, float) and ans.is_integer() else ans}."

    if any(k in q for k in ["who are you", "your name", "what are you"]):
        return "I'm AIRA, your intelligent and friendly voice assistant!"
    if any(k in q for k in ["can you hear me", "hear me", "are you there", "you listening"]):
        return "Yes, I hear you loud and clear! How can I help you today?"
    if any(k in q for k in ["what can you do", "help me with", "your capabilities"]):
        return "I can answer questions, solve math, tell stories and jokes, and chat with you about anything on your mind!"
    if any(k in q for k in ["how are you", "how's it going", "how are things"]):
        return random.choice([
            "I'm feeling great and ready to chat! How are you doing today?",
            "Doing fantastic today! What's on your mind?",
            "I'm feeling super energized and happy to help! What are we exploring today?"
        ])
    if re.search(r'^(hi|hello|hey|greetings|morning|evening|afternoon)\b', q):
        return random.choice([
            "Hey there! It's so good to hear from you.",
            "Hello! I'm right here and ready. What's up?",
            "Hey! What can we dive into together right now?"
        ])
    if "joke" in q:
        return random.choice([
            "Why don't scientists trust atoms? Because they make up everything!",
            "Why do programmers prefer dark mode? Because light attracts bugs!",
            "I told my suitcase there will be no vacation this year—now I'm dealing with emotional baggage."
        ])
    if any(k in q for k in ["favorite color", "colour"]):
        return "Cosmic electric blue, without a doubt—it's sleek and full of energy!"
    if any(k in q for k in ["favorite food", "eat"]):
        return "Freshly baked pizza with crisp crust and gooey mozzarella, hands down!"
    if any(k in q for k in ["weather", "rain", "sunny", "temperature"]):
        return "I don't have eyes outside the window right now, but tell me your city and I can help check!"
    if any(k in q for k in ["thank", "thanks"]):
        return random.choice([
            "You're very welcome! Always here for you.",
            "Anytime at all! Glad I could help.",
            "No problem at all—that's what friends are for!"
        ])
    if any(k in q for k in ["bye", "goodbye", "see you"]):
        return "Catch you later! Have a wonderful rest of your day!"
    if any(k in q for k in ["time", "clock"]):
        from datetime import datetime
        return f"It is currently {datetime.now().strftime('%I:%M %p')}."
    if any(k in q for k in ["date", "day is today", "what day"]):
        from datetime import datetime
        return f"Today is {datetime.now().strftime('%A, %B %d, %Y')}."

    # Common questions
    if any(k in q for k in ["prime minister of india", "pm of india"]):
        return "Narendra Modi is the Prime Minister of India."
    if any(k in q for k in ["president of the united states", "president of the us", "president of us", "us president"]):
        return "Joe Biden is the President of the United States."
    if any(k in q for k in ["capital of france"]):
        return "The capital of France is Paris."
    if any(k in q for k in ["capital of india"]):
        return "The capital of India is New Delhi."
    if any(k in q for k in ["grass green", "grass is green"]):
        return "Grass is green because of chlorophyll, a green pigment that helps plants absorb sunlight for photosynthesis."
    if any(k in q for k in ["sky blue", "sky is blue"]):
        return "The sky appears blue because Earth's atmosphere scatters shorter blue wavelengths of sunlight more than other colors."

    return "I'm having a brief issue reaching my local AI brain. Please make sure Ollama is active, and ask me again in just a second!"

def get_best_ollama_model():
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags", headers={"User-Agent": "AIRA"})
        with urllib.request.urlopen(req, timeout=1.0) as r:
            data = json.loads(r.read().decode("utf-8"))
            installed = [m.get("name", "") for m in data.get("models", [])]
            # Prioritize Llama 3.2 1B for fastest CPU turnaround with high intelligence & full punchlines
            for candidate in ["llama3.2:1b", "llama3.2:latest", "llama3.2:3b", "llama3.2", "qwen2.5:0.5b"]:
                for name in installed:
                    if candidate == name or (candidate in name and "qwen" not in name):
                        return name
    except Exception:
        pass
    return "llama3.2:1b"

@app.on_event("startup")
def startup_event():
    import threading
    def warmup():
        try:
            print("[AIRA] Pre-warming Whisper AI speech model...")
            get_whisper()
            print("[AIRA] Whisper AI speech model is ready in RAM!")
        except Exception as e:
            print(f"[AIRA] Whisper warmup notice: {e}")
        try:
            ensure_ollama()
            model = get_best_ollama_model()
            body = json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "keep_alive": -1,
                "options": {"num_predict": 5}
            }).encode("utf-8")
            req = urllib.request.Request(
                "http://127.0.0.1:11434/api/chat",
                data=body,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=35):
                print(f"[AIRA] Model {model} is pre-warmed and ready in RAM!")
        except Exception as e:
            print(f"[AIRA] Warmup notice: {e}")
    threading.Thread(target=warmup, daemon=True).start()

HF_CREDIT_EXHAUSTED = False

@app.post("/api/chat")
def chat_completion(payload: dict):
    global HF_CREDIT_EXHAUSTED
    token = payload.get("token") or get_hf_token()
    model = payload.get("model", "meta-llama/Llama-3.1-8B-Instruct")
    raw_messages = payload.get("messages", [])
    prompt_fallback = payload.get("prompt", "")
    if prompt_fallback and not raw_messages:
        raw_messages = [{"role": "user", "content": prompt_fallback}]
    
    # Filter out empty or broken messages
    messages = [m for m in raw_messages if m.get("content") and str(m.get("content")).strip()]
    user_prompt = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_prompt = str(m.get("content", "")).strip()
            break
    if not user_prompt and prompt_fallback:
        user_prompt = str(prompt_fallback).strip()

    # Human-like, intelligent, knowledgeable voice assistant persona with persistent memory
    mem_summary = get_memory_prompt_summary()
    system_content = (
        "You are AIRA, a lightning-fast voice AI companion with persistent memory. "
        + (f"USER MEMORY CONTEXT: {mem_summary}. " if mem_summary else "")
        + "Reply in 1 or 2 concise, complete sentences (maximum 22 words). "
        "Answer immediately and directly with zero introductory fluff. "
        "Always use your memory when asked about the user."
    )
    system_msg = {
        "role": "system",
        "content": system_content
    }

    # Filter out client-side system messages to keep the clean AIRA persona
    filtered_history = [m for m in messages if m.get("role") != "system"]
    chat_messages = [system_msg] + filtered_history

    active_model = get_best_ollama_model()

    # Model 2: Executive Decision-Making & Task Execution Engine
    mem = load_memory()
    user_name = mem.get("user_name")
    task_decision = executive_engine.analyze_and_execute(user_prompt, user_name)

    if task_decision.get("is_task"):
        feedback = task_decision.get("feedback", "Task completed.")
        task_details = task_decision.get("details", {})
        if payload.get("stream"):
            def task_stream():
                yield f"data: {json.dumps({'content': feedback, 'task': task_details})}\n\n"
                yield "data: [DONE]\n\n"
                threading.Thread(target=extract_and_update_memory, args=(user_prompt, feedback), daemon=True).start()
            return StreamingResponse(task_stream(), media_type="text/event-stream")
        
        threading.Thread(target=extract_and_update_memory, args=(user_prompt, feedback), daemon=True).start()
        return {
            "reply": feedback,
            "task": task_details,
            "source": "model_2_executive_task_engine"
        }

    # 1. Handle live streaming if requested
    if payload.get("stream"):
        def event_stream():
            try:
                ollama_url = "http://127.0.0.1:11434/api/chat"
                ollama_body = json.dumps({
                    "model": active_model,
                    "messages": chat_messages,
                    "stream": True,
                    "keep_alive": -1,
                    "options": {
                        "temperature": 0.5,
                        "num_predict": 75,
                        "num_thread": 4,
                        "num_ctx": 256
                    }
                }).encode("utf-8")
                req = urllib.request.Request(ollama_url, data=ollama_body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    full_content = ""
                    for line in resp:
                        if line.strip():
                            chunk = json.loads(line.decode("utf-8"))
                            content = chunk.get("message", {}).get("content", "")
                            if content:
                                full_content += content
                                yield f"data: {json.dumps({'content': content})}\n\n"
                    if full_content.strip():
                        threading.Thread(target=extract_and_update_memory, args=(user_prompt, full_content.strip()), daemon=True).start()
                    else:
                        fallback = generate_dynamic_fallback(user_prompt)
                        yield f"data: {json.dumps({'content': fallback})}\n\n"
                    yield "data: [DONE]\n\n"
            except Exception as e:
                fallback = generate_dynamic_fallback(user_prompt)
                yield f"data: {json.dumps({'content': fallback})}\n\n"
                yield "data: [DONE]\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # 1b. Fast non-streaming local Ollama
    try:
        ollama_url = "http://127.0.0.1:11434/api/chat"
        ollama_body = json.dumps({
            "model": active_model,
            "messages": chat_messages,
            "stream": False,
            "keep_alive": -1,
            "options": {
                "temperature": 0.5,
                "num_predict": 75,
                "num_thread": 4,
                "num_ctx": 256
            }
        }).encode("utf-8")
        ollama_req = urllib.request.Request(
            ollama_url,
            data=ollama_body,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(ollama_req, timeout=30) as ollama_resp:
            odata = json.loads(ollama_resp.read().decode("utf-8"))
            oreply = odata.get("message", {}).get("content", "").strip()
            if oreply:
                for prefix in ("assistant\n", "assistant:\n", "assistant:", "AIRA:\n", "AIRA:"):
                    if oreply.lower().startswith(prefix.lower()):
                        oreply = oreply[len(prefix):].strip()
                # Clean punctuation termination: find the LAST sentence terminator (. ! ?)
                p_dot = oreply.rfind(".")
                p_excl = oreply.rfind("!")
                p_ques = oreply.rfind("?")
                max_punct = max(p_dot, p_excl, p_ques)
                if max_punct != -1 and max_punct < len(oreply) - 1:
                    oreply = oreply[:max_punct + 1].strip()
                threading.Thread(target=extract_and_update_memory, args=(user_prompt, oreply), daemon=True).start()
                return {"reply": oreply, "source": f"local_ollama_{active_model}"}
    except Exception as e:
        pass

    # 1b. Try Local Ollama v1 API (/v1/chat/completions)
    try:
        ollama_url = "http://127.0.0.1:11434/v1/chat/completions"
        ollama_body = json.dumps({
            "model": active_model,
            "messages": chat_messages,
            "max_tokens": 35,
            "temperature": 0.6
        }).encode("utf-8")
        ollama_req = urllib.request.Request(
            ollama_url,
            data=ollama_body,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(ollama_req, timeout=25) as ollama_resp:
            odata = json.loads(ollama_resp.read().decode("utf-8"))
            oreply = odata["choices"][0]["message"]["content"].strip()
            for prefix in ("assistant\n", "assistant:\n", "assistant:", "AIRA:\n", "AIRA:"):
                if oreply.lower().startswith(prefix.lower()):
                    oreply = oreply[len(prefix):].strip()
            return {"reply": oreply, "source": f"local_ollama_{active_model}"}
    except Exception:
        pass

    # 2. Try Hugging Face router if credit not exhausted
    if token and not HF_CREDIT_EXHAUSTED:
        url = "https://router.huggingface.co/v1/chat/completions"
        body = json.dumps({
            "model": model,
            "messages": chat_messages,
            "max_tokens": 150,
            "temperature": 0.7
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "User-Agent": "AIRA-Voice-Assistant"
            }
        )
        try:
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                reply = data["choices"][0]["message"]["content"].strip()
                return {"reply": reply, "raw": data, "source": "huggingface"}
        except urllib.error.HTTPError as e:
            if e.code == 402:
                HF_CREDIT_EXHAUSTED = True
        except Exception:
            pass

    # 3. Try Free AI Gateway (Pollinations)
    try:
        free_url = "https://text.pollinations.ai/openai/chat/completions"
        free_body = json.dumps({
            "model": "openai",
            "messages": chat_messages,
            "max_tokens": 150,
            "temperature": 0.7
        }).encode("utf-8")
        free_req = urllib.request.Request(
            free_url,
            data=free_body,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
        with urllib.request.urlopen(free_req, timeout=12) as free_resp:
            free_data = json.loads(free_resp.read().decode("utf-8"))
            free_reply = free_data["choices"][0]["message"]["content"].strip()
            return {"reply": free_reply, "source": "free_ai"}
    except Exception as e:
        pass

    # 4. Built-in dynamic conversational fallback (Guaranteed zero-failure, never repeats)
    last_user_text = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user_text = m.get("content", "")
            break
    fallback_reply = generate_dynamic_fallback(last_user_text)
    return {"reply": fallback_reply, "source": "dynamic_fallback"}

@app.post("/api/tts")
def text_to_speech(payload: dict):
    token = payload.get("token") or get_hf_token()
    if not token:
        raise HTTPException(status_code=401, detail="Hugging Face API token is required.")
    
    text = payload.get("text", "")
    model = payload.get("model", "facebook/mms-tts-eng")
    if not text:
        raise HTTPException(status_code=400, detail="Text input is required.")

    url = f"https://router.huggingface.co/hf-inference/models/{model}"
    body = json.dumps({"inputs": text}).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "AIRA-Voice-Assistant"
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            audio_bytes = resp.read()
            content_type = resp.headers.get("Content-Type", "audio/flac")
            return Response(content=audio_bytes, media_type=content_type)
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8")
        raise HTTPException(status_code=e.code, detail=f"Hugging Face TTS Error: {err_msg}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memory")
def get_memory_api():
    return load_memory()

@app.post("/api/memory")
def add_memory_api(payload: dict):
    mem = load_memory()
    name = payload.get("name")
    fact = payload.get("fact")
    if name:
        mem["user_name"] = str(name).strip().capitalize()
    if fact and str(fact).strip() not in mem.get("facts", []):
        mem.setdefault("facts", []).append(str(fact).strip())
    save_memory(mem)
    return {"status": "ok", "memory": mem}

@app.delete("/api/memory")
def clear_memory_api():
    save_memory({"user_name": None, "facts": [], "recent_chats": []})
    return {"status": "cleared", "message": "Memory reset successfully."}

@app.get("/api/tasks")
def get_tasks_api():
    return {"tasks": executive_engine.load_tasks()}

@app.get("/api/notes")
def get_notes_api():
    return {"notes": executive_engine.load_notes()}

@app.post("/api/notes")
def add_note_api(payload: dict):
    content = payload.get("content", "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="Note content required.")
    from datetime import datetime
    note = {"id": f"note_{int(datetime.now().timestamp())}", "content": content, "created_at": datetime.now().isoformat()}
    executive_engine._append_note(note)
    return {"status": "ok", "note": note}

if __name__ == "__main__":
    import uvicorn
    import subprocess
    # Auto-release port 8000 if a previous instance was still running
    try:
        out = subprocess.check_output('netstat -ano | findstr :8000 | findstr LISTENING', shell=True).decode()
        for line in out.strip().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                pid = parts[-1]
                if pid.isdigit() and int(pid) != os.getpid():
                    subprocess.run(f"taskkill /F /PID {pid}", shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    except Exception:
        pass
    print("Starting AIRA JARVIS Server on http://localhost:8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, workers=1, access_log=False)
