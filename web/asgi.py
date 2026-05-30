"""ASGI adapter for running the Flask app with uvicorn."""

from uvicorn.middleware.wsgi import WSGIMiddleware

from web.server import app as flask_app

app = WSGIMiddleware(flask_app)
