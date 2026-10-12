"""Exclusive output ownership, with recovery after the owner process exits."""

from __future__ import annotations
import contextlib, ctypes, json, os
from pathlib import Path


def process_alive(pid):
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, 0, pid)
        if not handle:
            # Access denied means another process may still own the lock.
            return ctypes.get_last_error() == 5
        try:
            code = ctypes.c_ulong()
            if not kernel.GetExitCodeProcess(
                ctypes.c_void_p(handle), ctypes.byref(code)
            ):
                return True
            return code.value == 259
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@contextlib.contextmanager
def output_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".experiment.lock"
    for attempt in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                pid = int(json.loads(path.read_text(encoding="utf-8"))["pid"])
            except (ValueError, KeyError, json.JSONDecodeError):
                raise RuntimeError(
                    "output lock is unreadable; inspect it before restarting"
                )
            if process_alive(pid):
                raise RuntimeError(f"this output is already owned by running PID {pid}")
            if attempt:
                raise RuntimeError("output ownership changed during lock recovery")
            path.unlink()  # One explicitly resolved stale lock, never a directory.
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump({"pid": os.getpid()}, stream)
    try:
        yield
    finally:
        if (
            path.is_file()
            and json.loads(path.read_text(encoding="utf-8")).get("pid") == os.getpid()
        ):
            path.unlink()
