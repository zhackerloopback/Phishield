# api/index.py
# Vercel serverless entrypoint for Flask (WSGI) apps
from app import app  # import your Flask app instance
from vercel_wsgi import handle

def handler(request, *args, **kwargs):
    return handle(app, request, *args, **kwargs)

