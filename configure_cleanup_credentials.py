"""Store or clear the Cloudflare Access service token in the OS keyring."""

from __future__ import annotations

import argparse
from getpass import getpass
import os
from typing import Optional

from cleanup_client import (
    CREDENTIAL_CLIENT_ID_KEY,
    CREDENTIAL_CLIENT_SECRET_KEY,
    CREDENTIAL_SERVICE,
    CleanupClient,
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Configure the protected transcript-cleanup service token"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Remove the saved token from the OS credential store",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Store without first calling the protected glossary endpoint",
    )
    args = parser.parse_args(argv)

    try:
        import keyring  # type: ignore
        from keyring.errors import PasswordDeleteError  # type: ignore
    except ImportError:
        print("The 'keyring' package is required; install requirements.txt first.")
        return 1

    if args.clear:
        for username in (CREDENTIAL_CLIENT_ID_KEY, CREDENTIAL_CLIENT_SECRET_KEY):
            try:
                keyring.delete_password(CREDENTIAL_SERVICE, username)
            except PasswordDeleteError:
                pass
        print("Saved Cloudflare Access credentials were removed.")
        return 0

    client_id = input("Cloudflare Access client ID: ").strip()
    client_secret = getpass("Cloudflare Access client secret: ").strip()
    if not client_id or not client_secret:
        print("Both values are required; nothing was stored.")
        return 1

    if not args.no_validate:
        endpoint = os.environ.get("PG_CLEANUP_ENDPOINT", DEFAULT_ENDPOINT)
        model = os.environ.get("PG_CLEANUP_MODEL", DEFAULT_MODEL)
        try:
            snapshot = CleanupClient(
                client_id=client_id,
                client_secret=client_secret,
                endpoint=endpoint,
                model=model,
            ).ensure_glossary()
        except Exception as exc:
            print(f"Credential validation failed; nothing was stored: {exc}")
            return 1
        print(f"Protected glossary access verified ({snapshot.count:,} editable terms).")

    keyring.set_password(CREDENTIAL_SERVICE, CREDENTIAL_CLIENT_ID_KEY, client_id)
    keyring.set_password(
        CREDENTIAL_SERVICE, CREDENTIAL_CLIENT_SECRET_KEY, client_secret
    )
    print("Cloudflare Access credentials were saved in the OS credential store.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
