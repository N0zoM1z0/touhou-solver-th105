"""Cross-platform singleton guard for commands that mutate TH105 input."""

from __future__ import annotations

import json
import os
import socket
import tempfile
import time
from pathlib import Path
from typing import BinaryIO


class ControllerAlreadyRunning(RuntimeError):
    """Raised before a second input controller can touch the game."""


def default_controller_lock_path() -> Path:
    """Keep Windows locking on NTFS; msvcrt byte locks reject WSL UNC files."""
    if os.name == "nt":
        return Path(tempfile.gettempdir()) / "touhou-solver-th105" / "controller.lock"
    return (
        Path(__file__).resolve().parents[2]
        / "runtime"
        / "th105_controller.lock"
    )


class ControllerLock:
    def __init__(self, path: Path, *, command: str) -> None:
        self.path = path
        self.command = command
        self._handle: BinaryIO | None = None

    def _owner_description(self, handle: BinaryIO) -> str:
        try:
            handle.seek(0)
            raw = handle.read().decode("utf-8", errors="replace").strip("\x00\r\n ")
            owner = json.loads(raw)
            if isinstance(owner, dict):
                return ", ".join(
                    f"{name}={owner[name]}"
                    for name in ("pid", "command", "host", "started_at")
                    if name in owner
                )
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
        return "owner metadata unavailable"

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        handle = os.fdopen(descriptor, "r+b", buffering=0)
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\x00")
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                owner = self._owner_description(handle)
                raise ControllerAlreadyRunning(
                    f"TH105 input controller is already running ({owner})"
                ) from exc
            metadata = json.dumps(
                {
                    "pid": os.getpid(),
                    "command": self.command,
                    "host": socket.gethostname(),
                    "started_at": time.time(),
                },
                separators=(",", ":"),
            ).encode("utf-8")
            handle.seek(0)
            handle.truncate()
            handle.write(metadata)
            self._handle = handle
        except BaseException:
            handle.close()
            raise

    def release(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self) -> ControllerLock:
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()
