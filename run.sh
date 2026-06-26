#!/usr/bin/env bash
# Start Fingent (backend API + frontend) on http://localhost:8000
set -e
cd "$(dirname "$0")/backend"
python3 -m pip install -r requirements.txt
# optional: export GROQ_API_KEY=...  (and GROQ_MODEL=llama-3.3-70b-versatile) to use Llama
exec python3 -m uvicorn fingent.app:app --reload --port 8000
