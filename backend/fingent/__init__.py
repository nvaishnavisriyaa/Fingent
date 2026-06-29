"""Fingent - a reusable agentic platform for financial services."""
import os as _os

# Load environment variables from a .env file BEFORE any submodule reads them.
# Looks for .env in the project root (parent of backend/) and the current dir.
# No-op if python-dotenv isn't installed or no .env exists, so demo mode still works.
try:
    from dotenv import load_dotenv as _load_dotenv

    _root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
    _load_dotenv(_os.path.join(_root, ".env"))  # project-root .env
    _load_dotenv()                               # also pick up ./.env if present
except ImportError:
    pass

from .platform import Fingent

__all__ = ["Fingent"]
__version__ = "0.1.0"
