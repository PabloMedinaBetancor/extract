"""The local interface: a small web server and the four screens. It uses `extract` and nothing else of the package."""

from .server import serve

__all__ = ["serve"]
