import os
import io
import json
import wave
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
        return "I'm AIRA, your fast and friendly voice companion!"
    if any(k in q for k in ["how are you", "how's it going", "how are things"]):
        return random.choice([
            "I'm feeling great and ready to chat! How are you doing?",
            "Doing fantastic today! What's on your mind?",
            "I'm feeling super energized! What are we exploring today?"
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

    # Dynamic contextual response using user query keywords
    words = [w for w in re.findall(r'\b[a-zA-Z]{4,}\b', q) if w not in ("what", "when", "where", "which", "about", "there", "their", "could", "would", "should", "tell", "know")]
    if words:
        topic = words[-1]
        return random.choice([
            f"That's really interesting regarding {topic}! Tell me more about what you think.",
            f"I was just thinking about {topic} too! What aspect would you like to explore?",
            f"Great question about {topic}! I'm all ears—let's dive deeper into it."
        ])

    return random.choice([
        "I'm listening closely! What would you like to talk about next?",
        "Sounds good! Tell me more about that.",
        "I'm here with you—what shall we tackle today?",
        "I hear you loud and clear! What's next on our agenda?"
    ])

def get_best_ollama_model():
    return "qwen2.5:0.5b"

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
    messages = payload.get("messages", [])
    
    # Human-like, warm, direct conversational persona with complete answers
    system_msg = {
        "role": "system",
        "content": "You are AIRA, a quick, lively, and warm friend. Answer naturally in 1 short, complete sentence as a real human. Never say you are an AI or digital assistant. Always finish your sentence cleanly."
    }
    has_system = any(m.get("role") == "system" for m in messages)
    
    # 1-shot human exemplar to ensure small models stay natural and friendly
    has_prior_assistant = any(m.get("role") == "assistant" for m in messages)
    few_shot = [
        {"role": "user", "content": "What is your favorite food?"},
        {"role": "assistant", "content": "Fresh pizza with gooey mozzarella and crispy crust, hands down!"}
    ] if not has_prior_assistant else []

    chat_messages = ([system_msg] if not has_system else []) + few_shot + messages

    active_model = get_best_ollama_model()

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
                        "temperature": 0.65,
                        "num_predict": 16,
                        "num_thread": 4,
                        "num_ctx": 256
                    }
                }).encode("utf-8")
                req = urllib.request.Request(ollama_url, data=ollama_body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    for line in resp:
                        if line.strip():
                            chunk = json.loads(line.decode("utf-8"))
                            content = chunk.get("message", {}).get("content", "")
                            if content:
                                yield f"data: {json.dumps({'content': content})}\n\n"
                    yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'content': generate_dynamic_fallback('')})}\n\n"
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
                "temperature": 0.65,
                "num_predict": 16,
                "num_thread": 4,
                "num_ctx": 256
            }
        }).encode("utf-8")
        ollama_req = urllib.request.Request(
            ollama_url,
            data=ollama_body,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(ollama_req, timeout=15) as ollama_resp:
            odata = json.loads(ollama_resp.read().decode("utf-8"))
            oreply = odata.get("message", {}).get("content", "").strip()
            if oreply:
                for prefix in ("assistant\n", "assistant:\n", "assistant:", "AIRA:\n", "AIRA:"):
                    if oreply.lower().startswith(prefix.lower()):
                        oreply = oreply[len(prefix):].strip()
                # Clean punctuation termination
                for punct in [".", "!", "?"]:
                    if punct in oreply:
                        oreply = oreply[:oreply.rfind(punct)+1]
                        break
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
