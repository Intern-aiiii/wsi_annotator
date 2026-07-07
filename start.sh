echo "Use Ctrl+C to stop the server"
.venv/bin/uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
