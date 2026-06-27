@echo off
REM Start Fingent (backend API + frontend) on http://localhost:8000
cd /d "%~dp0backend"
python -m pip install -r requirements.txt
REM Persist agents/audit/traces across restarts (delete fingent.db to reset)
if "%FINGENT_DB%"=="" set FINGENT_DB=fingent.db
REM Optional: set GROQ_API_KEY=sk-... (and GROQ_MODEL=llama-3.3-70b-versatile) to use Llama
echo.
echo  Fingent is starting — open http://localhost:8000 in your browser
echo.
python -m uvicorn fingent.app:app --port 8000
