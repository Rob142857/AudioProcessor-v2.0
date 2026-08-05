"""Small Windows GUI for securely configuring the cleanup service token."""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import messagebox, simpledialog

import keyring  # type: ignore

from cleanup_client import (
    CREDENTIAL_CLIENT_ID_KEY,
    CREDENTIAL_CLIENT_SECRET_KEY,
    CREDENTIAL_SERVICE,
    ACCESS_CLIENT_ID_HEADER,
    ACCESS_CLIENT_SECRET_HEADER,
    CleanupClient,
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    normalize_access_credential,
)


def main() -> int:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        client_id = simpledialog.askstring(
            "AudioProcessor cleanup access",
            "Cloudflare Access client ID:",
            parent=root,
        )
        if client_id is None:
            return 1
        client_secret = simpledialog.askstring(
            "AudioProcessor cleanup access",
            "Cloudflare Access client secret (shown only as bullets):",
            parent=root,
            show="*",
        )
        if client_secret is None:
            return 1

        client_id = normalize_access_credential(client_id, ACCESS_CLIENT_ID_HEADER)
        client_secret = normalize_access_credential(
            client_secret, ACCESS_CLIENT_SECRET_HEADER
        )
        if not client_id or not client_secret:
            messagebox.showerror(
                "Nothing saved", "Both the client ID and client secret are required.", parent=root
            )
            return 1

        keyring.set_password(CREDENTIAL_SERVICE, CREDENTIAL_CLIENT_ID_KEY, client_id)
        keyring.set_password(
            CREDENTIAL_SERVICE, CREDENTIAL_CLIENT_SECRET_KEY, client_secret
        )
        try:
            snapshot = CleanupClient(
                client_id=client_id,
                client_secret=client_secret,
                endpoint=os.environ.get("PG_CLEANUP_ENDPOINT", DEFAULT_ENDPOINT),
                model=os.environ.get("PG_CLEANUP_MODEL", DEFAULT_MODEL),
            ).ensure_glossary()
        except Exception as exc:
            messagebox.showerror(
                "Saved, but validation failed",
                "The pair was stored securely for diagnosis, but protected access "
                f"is not working yet.\n\n{type(exc).__name__}: {exc}",
                parent=root,
            )
            return 2

        messagebox.showinfo(
            "Cleanup access configured",
            f"Protected access verified and saved in Windows Credential Manager.\n\n"
            f"Editable glossary terms: {snapshot.count:,}",
            parent=root,
        )
        return 0
    finally:
        root.destroy()


if __name__ == "__main__":
    raise SystemExit(main())
