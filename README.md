# AIRA - Advanced Cybernetic AI Voice Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-purple.svg)](https://ollama.ai/)
[![Whisper](https://img.shields.io/badge/STT-Faster--Whisper-orange.svg)](https://github.com/SYSTRAN/faster-whisper)

**AIRA** is an advanced holographic cybernetic AI voice companion featuring real-time 3D WebGL neural lip-sync, local offline speech recognition powered by OpenAI Faster-Whisper, and ultra-fast sub-second streaming conversational intelligence powered by local LLMs via Ollama.

---

## 🌟 Key Features

* **🎙️ OpenAI Faster-Whisper STT**: High-accuracy, offline local speech-to-text with real-time audio volume monitoring and automatic noise suppression.
* **⚡ Sub-Second Streaming LLM**: Local LLM intelligence (`qwen2.5:0.5b`) with Server-Sent Events (SSE) streaming for instant conversational replies (< 1s turnaround).
* **🤖 3D Holographic Cybernetic Avatar**: Three.js and WebGL shaders rendering cybernetic particle rings, interactive mouse gaze tracking, dynamic blinking, and expressive facial states (`IDLE`, `LISTENING`, `THINKING`, `SPEAKING`).
* **👄 Real-Time Viseme Neural Lip-Sync**: Talking avatar neural lip-sync engine synchronized to speech phonemes and viseme weights (`viseme-model.bin`).
* **🛡️ Self-Healing Architecture**: Zero-failure dynamic fallback brain, automatic port conflict resolution, and self-managed background engine lifecycle.
* **🚀 1-Click Desktop Launch**: Preconfigured batch launchers (`Launch_AIRA.bat`, `start_aira.bat`) that prepare Ollama, start the FastAPI backend, and launch the UI.

---

## 🏗️ Architecture

```
[ User Microphone ]
       │
       ▼
[ Web Audio PCM (16kHz WAV) ]
       │
       ▼ (POST /api/stt)
[ FastAPI Server (server.py) ] ──► [ Faster-Whisper Engine ] ──► Extracted Query Text
       │
       ▼ (POST /api/chat)
[ Local Ollama (qwen2.5:0.5b) ] ──► Real-Time Token Stream
       │
       ▼ (SSE data: ...)
[ Web UI (AIRA_JARVIS_homepage (3).html) ]
       │
       ├─► [ SpeechSynthesis / Neural TTS ]
       └─► [ Viseme Timeline & 3D WebGL Lip-Sync Mesh ]
```

---

## 📋 Prerequisites

1. **Python 3.10+**: Ensure Python is installed and added to your system PATH.
2. **Ollama**: Download and install from [ollama.ai](https://ollama.ai/).
   * Pull the fast conversational model:
     ```bash
     ollama pull qwen2.5:0.5b
     ```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Nithin21-03/Aira.git
cd Aira
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch AIRA
Double-click **`Launch_AIRA.bat`** (or run via terminal):
```bash
start_aira.bat
```
This script will:
1. Verify Ollama is running and ready.
2. Start the FastAPI backend server on `http://localhost:8000`.
3. Pre-warm the Whisper AI speech model and Ollama LLM into RAM.
4. Launch the AIRA cybernetic holographic interface in your default browser.

---

## ⚙️ Project Structure

```
Aira/
├── AIRA_JARVIS_homepage (3).html # Main 3D WebGL cybernetic interface & client engine
├── index.html                    # Mirror entrypoint for standard web servers
├── server.py                     # FastAPI backend (Whisper STT, Ollama LLM streaming)
├── viseme-model.bin              # Binary tensor weights for real-time avatar lip-sync
├── requirements.txt              # Python package dependencies
├── start_aira.bat                # Workspace launch script with auto-recovery
├── Launch_AIRA.bat               # Desktop launcher script
├── avatars/
│   ├── female.glb                # 3D female cybernetic avatar model
│   ├── male.glb                  # 3D male cybernetic avatar model
│   └── LICENSE                   # Avatar 3D asset license
└── README.md                     # Documentation
```

---

## 🎙️ Voice Interaction Tips

* **Microphone Access**: When opening in Chrome, allow microphone permissions for `http://localhost:8000`.
* **Visual VU Meter**: When you speak, the dialogue bar displays live audio input percentage (e.g. `[🎙️ Volume: 45%]`).
* **Hands-Free Chat**: Speak naturally; AIRA automatically finalizes after 1.2s of silence and transcribes using Faster-Whisper.
* **Text Fallback**: You can also type messages in the input box below the avatar and hit Enter.

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details. Avatar models retain their respective Creative Commons licenses as detailed in `avatars/LICENSE`.
