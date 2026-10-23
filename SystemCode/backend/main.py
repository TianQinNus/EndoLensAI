"""Formal application entrypoint.

`app.py` remains a compatibility module for existing launch commands.
"""

from .app import app, create_app

__all__ = ["app", "create_app"]
