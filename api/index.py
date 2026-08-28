"""
Vercel serverless entrypoint. Vercel's Python runtime looks for an ASGI/WSGI
app object here. We just re-export the FastAPI app defined in app.py.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app import app  # noqa: E402