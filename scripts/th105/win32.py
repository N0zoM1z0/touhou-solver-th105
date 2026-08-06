"""Small Win32 boundary for exact-target launch, discovery, focus, and reads."""

from __future__ import annotations

import ctypes
import hashlib
import os
import struct
import time
from ctypes import wintypes
from pathlib import Path

from .constants import EXPECTED_EXE_SHA256, TARGET_EXE

TH32CS_SNAPPROCESS = 0x00000002
PROCESS_VM_READ = 0x0010
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SYNCHRONIZE = 0x00100000
CREATE_SUSPENDED = 0x00000004
PAGE_EXECUTE_READWRITE = 0x40
INFINITE = 0xFFFFFFFF
INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value
SW_RESTORE = 9
SW_SHOWNOACTIVATE = 4
STARTF_USESHOWWINDOW = 0x00000001

# Exact-build WinMain evidence (th105c.exe SHA-256 in constants.py):
#   00664D95  push 00040000h       ; WS_EX_APPWINDOW
#   00664D9A  call CreateWindowExA
# A suspended background launch changes only the immediate operand in the new
# process to replace APPWINDOW with WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW. The
# tool-window bit also keeps TH105 out of taskbar/Alt-Tab/fallback activation
# candidates when another controller destroys and recreates its foreground
# window. The executable on disk is never modified.
BACKGROUND_EXSTYLE_ADDRESS = 0x00664D96
BACKGROUND_EXSTYLE_ORIGINAL = struct.pack("<I", 0x00040000)
BACKGROUND_EXSTYLE_PATCHED = struct.pack("<I", 0x08000080)

INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
INJECTION_MARKER = 0x54483130


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_void_p),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", wintypes.WCHAR * 260),
    ]


class STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.POINTER(ctypes.c_byte)),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", INPUT_UNION)]


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


WNDENUMPROC = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)(
    wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
)


def require_windows() -> None:
    if os.name != "nt":
        raise RuntimeError("run the TH105 agent with Windows Python")


def _win_error(what: str) -> OSError:
    return ctypes.WinError(ctypes.get_last_error(), what)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Win32:
    def __init__(self) -> None:
        require_windows()
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)

        k = self.kernel32
        u = self.user32
        k.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        k.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        k.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
        k.Process32FirstW.restype = wintypes.BOOL
        k.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
        k.Process32NextW.restype = wintypes.BOOL
        k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.OpenProcess.restype = wintypes.HANDLE
        k.ReadProcessMemory.argtypes = [
            wintypes.HANDLE,
            wintypes.LPCVOID,
            wintypes.LPVOID,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        k.ReadProcessMemory.restype = wintypes.BOOL
        k.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        k.QueryFullProcessImageNameW.restype = wintypes.BOOL
        k.CreateProcessW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.LPWSTR,
            wintypes.LPVOID,
            wintypes.LPVOID,
            wintypes.BOOL,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.LPCWSTR,
            ctypes.POINTER(STARTUPINFOW),
            ctypes.POINTER(PROCESS_INFORMATION),
        ]
        k.CreateProcessW.restype = wintypes.BOOL
        k.VirtualProtectEx.argtypes = [
            wintypes.HANDLE,
            wintypes.LPVOID,
            ctypes.c_size_t,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        k.VirtualProtectEx.restype = wintypes.BOOL
        k.WriteProcessMemory.argtypes = [
            wintypes.HANDLE,
            wintypes.LPVOID,
            wintypes.LPCVOID,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        k.WriteProcessMemory.restype = wintypes.BOOL
        k.FlushInstructionCache.argtypes = [
            wintypes.HANDLE,
            wintypes.LPCVOID,
            ctypes.c_size_t,
        ]
        k.FlushInstructionCache.restype = wintypes.BOOL
        k.ResumeThread.argtypes = [wintypes.HANDLE]
        k.ResumeThread.restype = wintypes.DWORD
        k.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k.TerminateProcess.restype = wintypes.BOOL
        k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k.WaitForSingleObject.restype = wintypes.DWORD
        k.CloseHandle.argtypes = [wintypes.HANDLE]

        u.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
        u.SendInput.restype = wintypes.UINT
        u.GetForegroundWindow.restype = wintypes.HWND
        u.GetAsyncKeyState.argtypes = [ctypes.c_int]
        u.GetAsyncKeyState.restype = ctypes.c_short
        u.GetWindowThreadProcessId.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(wintypes.DWORD),
        ]
        u.GetWindowThreadProcessId.restype = wintypes.DWORD
        u.EnumWindows.argtypes = [WNDENUMPROC, wintypes.LPARAM]
        u.EnumWindows.restype = wintypes.BOOL
        u.IsWindowVisible.argtypes = [wintypes.HWND]
        u.IsWindowVisible.restype = wintypes.BOOL
        u.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(RECT)]
        u.GetWindowRect.restype = wintypes.BOOL
        u.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        u.GetWindowTextLengthW.restype = ctypes.c_int
        u.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        u.GetWindowTextW.restype = ctypes.c_int
        u.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        u.GetClassNameW.restype = ctypes.c_int
        u.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        u.BringWindowToTop.argtypes = [wintypes.HWND]
        u.SetForegroundWindow.argtypes = [wintypes.HWND]

        expected = 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28
        if ctypes.sizeof(INPUT) != expected:
            raise RuntimeError(
                f"unexpected INPUT size {ctypes.sizeof(INPUT)} (expected {expected})"
            )

    def find_pids(self, exe_name: str = TARGET_EXE) -> tuple[int, ...]:
        snapshot = self.kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
        if snapshot == INVALID_HANDLE_VALUE:
            raise _win_error("CreateToolhelp32Snapshot")
        try:
            entry = PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(entry)
            if not self.kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
                return ()
            found: list[int] = []
            while True:
                if entry.szExeFile.casefold() == exe_name.casefold():
                    found.append(int(entry.th32ProcessID))
                if not self.kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                    return tuple(found)
        finally:
            self.kernel32.CloseHandle(snapshot)

    def foreground_pid(self) -> int:
        owner = wintypes.DWORD()
        self.user32.GetWindowThreadProcessId(
            self.user32.GetForegroundWindow(), ctypes.byref(owner)
        )
        return int(owner.value)

    def windows_for_pid(self, pid: int) -> tuple[int, ...]:
        found: list[tuple[int, int, int, int]] = []

        @WNDENUMPROC
        def callback(window: int, _unused: int) -> bool:
            owner = wintypes.DWORD()
            self.user32.GetWindowThreadProcessId(window, ctypes.byref(owner))
            if owner.value == pid and self.user32.IsWindowVisible(window):
                rect = RECT()
                area = 0
                if self.user32.GetWindowRect(window, ctypes.byref(rect)):
                    area = max(0, rect.right - rect.left) * max(
                        0, rect.bottom - rect.top
                    )
                titled = int(self.user32.GetWindowTextLengthW(window) > 0)
                class_buffer = ctypes.create_unicode_buffer(256)
                self.user32.GetClassNameW(window, class_buffer, len(class_buffer))
                game_class = int(class_buffer.value.casefold() != "d3dproxywindow")
                found.append((game_class, titled, area, int(window)))
            return True

        if not self.user32.EnumWindows(callback, 0):
            raise _win_error("EnumWindows")
        # TH105 briefly owns a small DirectInput/helper window during scene
        # transitions. Prefer the largest visible window, which is the 640x480
        # render target, so focus acquisition does not race that helper.
        return tuple(
            window
            for _game_class, _titled, _area, window in sorted(found, reverse=True)
        )

    def window_rect(self, window: int) -> tuple[int, int, int, int]:
        rect = RECT()
        if not self.user32.GetWindowRect(window, ctypes.byref(rect)):
            raise _win_error("GetWindowRect")
        return (rect.left, rect.top, rect.right, rect.bottom)

    def window_text(self, window: int) -> str:
        length = self.user32.GetWindowTextLengthW(window)
        buffer = ctypes.create_unicode_buffer(max(1, length + 1))
        self.user32.GetWindowTextW(window, buffer, len(buffer))
        return buffer.value

    def window_class(self, window: int) -> str:
        buffer = ctypes.create_unicode_buffer(256)
        self.user32.GetClassNameW(window, buffer, len(buffer))
        return buffer.value

    def focus(self, pid: int, timeout: float = 8.0) -> int:
        deadline = time.perf_counter() + timeout
        last: tuple[int, ...] = ()
        while time.perf_counter() < deadline:
            last = self.windows_for_pid(pid)
            for window in last:
                self.user32.ShowWindow(window, SW_RESTORE)
                self.user32.BringWindowToTop(window)
                self.user32.SetForegroundWindow(window)
                if self.foreground_pid() == pid:
                    return window
            time.sleep(0.1)
        raise RuntimeError(f"could not focus PID {pid}; visible windows={last}")

    def _install_background_launch_patch(self, process: int) -> None:
        address = BACKGROUND_EXSTYLE_ADDRESS
        current = ctypes.create_string_buffer(len(BACKGROUND_EXSTYLE_ORIGINAL))
        transferred = ctypes.c_size_t()
        if not self.kernel32.ReadProcessMemory(
            process,
            address,
            current,
            len(current),
            ctypes.byref(transferred),
        ):
            raise _win_error("ReadProcessMemory(background launch)")
        observed = current.raw[: transferred.value]
        if observed != BACKGROUND_EXSTYLE_ORIGINAL:
            raise RuntimeError(
                "background launch patch preimage mismatch at "
                f"0x{address:08X}: {observed.hex()}"
            )

        old_protect = wintypes.DWORD()
        if not self.kernel32.VirtualProtectEx(
            process,
            address,
            len(BACKGROUND_EXSTYLE_PATCHED),
            PAGE_EXECUTE_READWRITE,
            ctypes.byref(old_protect),
        ):
            raise _win_error("VirtualProtectEx(background launch)")
        try:
            transferred = ctypes.c_size_t()
            patch = ctypes.create_string_buffer(BACKGROUND_EXSTYLE_PATCHED)
            if not self.kernel32.WriteProcessMemory(
                process,
                address,
                patch,
                len(BACKGROUND_EXSTYLE_PATCHED),
                ctypes.byref(transferred),
            ) or transferred.value != len(BACKGROUND_EXSTYLE_PATCHED):
                raise _win_error("WriteProcessMemory(background launch)")
            if not self.kernel32.FlushInstructionCache(
                process, address, len(BACKGROUND_EXSTYLE_PATCHED)
            ):
                raise _win_error("FlushInstructionCache(background launch)")
        finally:
            restored = wintypes.DWORD()
            if not self.kernel32.VirtualProtectEx(
                process,
                address,
                len(BACKGROUND_EXSTYLE_PATCHED),
                old_protect.value,
                ctypes.byref(restored),
            ):
                raise _win_error("VirtualProtectEx(restore background launch)")

    def launch(
        self,
        exe_path: Path,
        *,
        activate: bool = True,
        prevent_activation: bool = False,
    ) -> int:
        exe_path = exe_path.resolve()
        if activate and prevent_activation:
            raise ValueError(
                "activation and activation prevention are mutually exclusive"
            )
        if prevent_activation:
            observed_sha256 = sha256(exe_path)
            if observed_sha256 != EXPECTED_EXE_SHA256:
                raise RuntimeError(
                    f"refusing background launch patch for SHA-256 {observed_sha256}"
                )
        startup = STARTUPINFOW()
        startup.cb = ctypes.sizeof(startup)
        if not activate:
            # Show the GUI without granting it foreground ownership.  This is
            # required when another process-local Touhou controller is already
            # running and must not lose timing/focus authority.
            startup.dwFlags |= STARTF_USESHOWWINDOW
            startup.wShowWindow = SW_SHOWNOACTIVATE
        info = PROCESS_INFORMATION()
        creation_flags = CREATE_SUSPENDED if prevent_activation else 0
        if not self.kernel32.CreateProcessW(
            str(exe_path),
            None,
            None,
            None,
            False,
            creation_flags,
            None,
            str(exe_path.parent),
            ctypes.byref(startup),
            ctypes.byref(info),
        ):
            raise _win_error("CreateProcessW")
        try:
            if prevent_activation:
                try:
                    self._install_background_launch_patch(info.hProcess)
                    if self.kernel32.ResumeThread(info.hThread) == 0xFFFFFFFF:
                        raise _win_error("ResumeThread(background launch)")
                except Exception:
                    # This handle is the exact process created above. Never
                    # fall back to terminating by an ambiguous image name.
                    self.kernel32.TerminateProcess(info.hProcess, 1)
                    self.kernel32.WaitForSingleObject(info.hProcess, 5000)
                    raise
            return int(info.dwProcessId)
        finally:
            self.kernel32.CloseHandle(info.hThread)
            self.kernel32.CloseHandle(info.hProcess)


class ProcessReader:
    def __init__(self, api: Win32, pid: int) -> None:
        self.api = api
        self.pid = pid
        self.handle = api.kernel32.OpenProcess(
            PROCESS_VM_READ | PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        if not self.handle:
            raise _win_error("OpenProcess")

    def close(self) -> None:
        if self.handle:
            self.api.kernel32.CloseHandle(self.handle)
            self.handle = None

    def read(self, address: int, size: int) -> bytes:
        buffer = ctypes.create_string_buffer(size)
        count = ctypes.c_size_t()
        ok = self.api.kernel32.ReadProcessMemory(
            self.handle, ctypes.c_void_p(address), buffer, size, ctypes.byref(count)
        )
        if not ok or count.value != size:
            raise _win_error(f"ReadProcessMemory({address:#x}, {size})")
        return buffer.raw

    def u32(self, address: int) -> int:
        return struct.unpack("<I", self.read(address, 4))[0]

    def i32(self, address: int) -> int:
        return struct.unpack("<i", self.read(address, 4))[0]

    def u16(self, address: int) -> int:
        return struct.unpack("<H", self.read(address, 2))[0]

    def u8(self, address: int) -> int:
        return self.read(address, 1)[0]

    def i16(self, address: int) -> int:
        return struct.unpack("<h", self.read(address, 2))[0]

    def f32(self, address: int) -> float:
        return struct.unpack("<f", self.read(address, 4))[0]

    def i8(self, address: int) -> int:
        return struct.unpack("<b", self.read(address, 1))[0]

    def image_path(self) -> Path:
        buffer = ctypes.create_unicode_buffer(32768)
        length = wintypes.DWORD(len(buffer))
        if not self.api.kernel32.QueryFullProcessImageNameW(
            self.handle, 0, buffer, ctypes.byref(length)
        ):
            raise _win_error("QueryFullProcessImageNameW")
        return Path(buffer.value)

    def __enter__(self) -> "ProcessReader":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def verify_reader(
    reader: ProcessReader, expected_path: Path | None = None
) -> dict[str, object]:
    path = reader.image_path().resolve()
    digest = sha256(path)
    if path.name.casefold() != TARGET_EXE or digest != EXPECTED_EXE_SHA256:
        raise RuntimeError(f"target identity mismatch: path={path}, sha256={digest}")
    if expected_path is not None and os.path.normcase(str(path)) != os.path.normcase(
        str(expected_path.resolve())
    ):
        raise RuntimeError(
            f"target path mismatch: running={path}, expected={expected_path}"
        )
    if reader.read(0x00400000, 2) != b"MZ":
        raise RuntimeError("expected PE header is absent at 0x00400000")
    return {"pid": reader.pid, "path": str(path), "sha256": digest}


def find_exact_target(api: Win32, expected_path: Path) -> tuple[int, dict[str, object]]:
    matches: list[tuple[int, dict[str, object]]] = []
    for pid in api.find_pids():
        try:
            with ProcessReader(api, pid) as reader:
                matches.append((pid, verify_reader(reader, expected_path)))
        except (OSError, RuntimeError):
            continue
    if not matches:
        raise RuntimeError(f"exact {TARGET_EXE} target is not running")
    if len(matches) != 1:
        raise RuntimeError(f"ambiguous exact targets: {[pid for pid, _ in matches]}")
    return matches[0]


def wait_exact_target(
    api: Win32, expected_path: Path, pid: int, timeout: float
) -> dict[str, object]:
    deadline = time.perf_counter() + timeout
    last_error: Exception | None = None
    while time.perf_counter() < deadline:
        try:
            with ProcessReader(api, pid) as reader:
                return verify_reader(reader, expected_path)
        except (OSError, RuntimeError) as exc:
            last_error = exc
            time.sleep(0.1)
    raise TimeoutError(f"target PID {pid} was not ready: {last_error}")
