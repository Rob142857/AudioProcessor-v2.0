"""Small compatibility helpers for Windows console and redirected output."""

from __future__ import annotations

import sys
from typing import Any


def configure_safe_stdio(*streams: Any) -> None:
    """Prevent diagnostic Unicode from crashing a transcription run.

    Windows PowerShell may expose a legacy code page such as cp1252.  The
    transcription engine uses Unicode status symbols, so an ordinary ``print``
    must degrade safely when a caller redirects output through that code page.
    """

    selected = streams or (sys.stdout, sys.stderr)
    for stream in selected:
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(errors="replace")
        except (AttributeError, OSError, ValueError):
            # GUI/test streams can be immutable or already closed.
            continue
