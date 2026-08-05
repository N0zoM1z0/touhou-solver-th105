"""Verified in-process DirectInput overlay for TH105.

TH105's DirectInput8 keyboard device does not observe Win32 SendInput events on
the supported host.  This bridge hooks the first instruction after keyboard
GetDeviceState and ORs a private 256-byte DIK overlay into the freshly polled
buffer.  Installation and removal verify exact bytes and exact target identity.
"""

from __future__ import annotations

import ctypes
import struct
import time
from contextlib import contextmanager
from ctypes import wintypes

from .constants import ADDR_RAW_KEYBOARD
from .input import KEYS
from .win32 import ProcessReader, Win32, _win_error

PROCESS_VM_WRITE = 0x0020
PROCESS_VM_OPERATION = 0x0008
PROCESS_VM_READ = 0x0010
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_SUSPEND_RESUME = 0x0800
MEM_COMMIT = 0x1000
MEM_RESERVE = 0x2000
MEM_RELEASE = 0x8000
PAGE_EXECUTE_READWRITE = 0x40
STILL_ACTIVE = 259

HOOK_ADDRESS = 0x00408218
HOOK_ORIGINAL = bytes.fromhex("A1 DC CF 6E 00")  # mov eax, [0x006ECFDC]
CAVE_SIZE = 0x200
CONTROL_OFFSET = 0x100


def rel32(source: int, target: int) -> bytes:
    displacement = target - (source + 5)
    if not -(2**31) <= displacement < 2**31:
        raise RuntimeError(f"relative branch out of range: {source:#x} -> {target:#x}")
    return struct.pack("<i", displacement)


def jump_target(source: int, instruction: bytes) -> int:
    if len(instruction) != 5 or instruction[0] != 0xE9:
        raise ValueError("expected a five-byte near jump")
    return source + 5 + struct.unpack("<i", instruction[1:])[0]


def build_stub(cave: int) -> bytes:
    control = cave + CONTROL_OFFSET
    code = bytearray()
    code += b"\x9C\x50\x51\x52\x53"  # pushfd, eax, ecx, edx, ebx
    code += b"\xB9" + struct.pack("<I", control)  # mov ecx, control
    code += b"\xBA" + struct.pack("<I", ADDR_RAW_KEYBOARD)  # mov edx, raw
    code += b"\xB8\x00\x01\x00\x00"  # mov eax, 256
    loop = len(code)
    code += b"\x8A\x19"  # mov bl, [ecx]
    code += b"\x08\x1A"  # or [edx], bl
    code += b"\x41\x42\x48"  # inc ecx; inc edx; dec eax
    branch_end = len(code) + 2
    displacement = loop - branch_end
    if not -128 <= displacement <= 127:
        raise AssertionError("overlay loop no longer fits a short branch")
    code += b"\x75" + struct.pack("b", displacement)
    code += b"\x5B\x5A\x59\x58\x9D"  # pop ebx, edx, ecx, eax, popfd
    code += HOOK_ORIGINAL
    jump_address = cave + len(code)
    code += b"\xE9" + rel32(jump_address, HOOK_ADDRESS + len(HOOK_ORIGINAL))
    return bytes(code)


class InjectedInputBridge:
    def __init__(self, api: Win32, reader: ProcessReader) -> None:
        self.api = api
        self.reader = reader
        self.handle = None
        self.cave: int | None = None
        self.control: int | None = None
        self.installed = False
        self._configure_api()

    def _configure_api(self) -> None:
        k = self.api.kernel32
        k.WriteProcessMemory.argtypes = [
            wintypes.HANDLE, wintypes.LPVOID, wintypes.LPCVOID,
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t),
        ]
        k.WriteProcessMemory.restype = wintypes.BOOL
        k.VirtualAllocEx.argtypes = [
            wintypes.HANDLE, wintypes.LPVOID, ctypes.c_size_t,
            wintypes.DWORD, wintypes.DWORD,
        ]
        k.VirtualAllocEx.restype = wintypes.LPVOID
        k.VirtualFreeEx.argtypes = [
            wintypes.HANDLE, wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD
        ]
        k.VirtualFreeEx.restype = wintypes.BOOL
        k.VirtualProtectEx.argtypes = [
            wintypes.HANDLE, wintypes.LPVOID, ctypes.c_size_t,
            wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
        ]
        k.VirtualProtectEx.restype = wintypes.BOOL
        k.FlushInstructionCache.argtypes = [
            wintypes.HANDLE, wintypes.LPCVOID, ctypes.c_size_t
        ]
        k.FlushInstructionCache.restype = wintypes.BOOL
        k.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        k.GetExitCodeProcess.restype = wintypes.BOOL
        self.ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
        self.ntdll.NtSuspendProcess.argtypes = [wintypes.HANDLE]
        self.ntdll.NtSuspendProcess.restype = wintypes.LONG
        self.ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
        self.ntdll.NtResumeProcess.restype = wintypes.LONG

    @contextmanager
    def _suspended(self):
        status = int(self.ntdll.NtSuspendProcess(self.handle))
        if status < 0:
            raise RuntimeError(f"NtSuspendProcess failed: NTSTATUS {status:#010x}")
        try:
            yield
        finally:
            status = int(self.ntdll.NtResumeProcess(self.handle))
            if status < 0:
                raise RuntimeError(f"NtResumeProcess failed: NTSTATUS {status:#010x}")

    def _write(self, address: int, data: bytes) -> None:
        written = ctypes.c_size_t()
        buffer = ctypes.create_string_buffer(data)
        if not self.api.kernel32.WriteProcessMemory(
            self.handle, ctypes.c_void_p(address), buffer, len(data), ctypes.byref(written)
        ) or written.value != len(data):
            raise _win_error(f"WriteProcessMemory({address:#x}, {len(data)})")

    def _write_code(self, address: int, data: bytes) -> None:
        old = wintypes.DWORD()
        if not self.api.kernel32.VirtualProtectEx(
            self.handle, ctypes.c_void_p(address), len(data),
            PAGE_EXECUTE_READWRITE, ctypes.byref(old)
        ):
            raise _win_error("VirtualProtectEx")
        try:
            self._write(address, data)
        finally:
            restored = wintypes.DWORD()
            if not self.api.kernel32.VirtualProtectEx(
                self.handle, ctypes.c_void_p(address), len(data),
                old.value, ctypes.byref(restored)
            ):
                raise _win_error("VirtualProtectEx restore")
        if not self.api.kernel32.FlushInstructionCache(
            self.handle, ctypes.c_void_p(address), len(data)
        ):
            raise _win_error("FlushInstructionCache")

    def _process_alive(self) -> bool:
        if not self.handle:
            return False
        exit_code = wintypes.DWORD()
        if not self.api.kernel32.GetExitCodeProcess(
            self.handle, ctypes.byref(exit_code)
        ):
            return False
        return exit_code.value == STILL_ACTIVE

    def install(self) -> None:
        if self.installed:
            return
        actual = self.reader.read(HOOK_ADDRESS, len(HOOK_ORIGINAL))
        if actual != HOOK_ORIGINAL:
            self._recover_verified_orphan(actual)
        access = (
            PROCESS_VM_READ | PROCESS_VM_WRITE | PROCESS_VM_OPERATION
            | PROCESS_SUSPEND_RESUME
            | PROCESS_QUERY_LIMITED_INFORMATION
        )
        self.handle = self.api.kernel32.OpenProcess(access, False, self.reader.pid)
        if not self.handle:
            raise _win_error("OpenProcess(input bridge)")
        try:
            allocation = self.api.kernel32.VirtualAllocEx(
                self.handle, None, CAVE_SIZE,
                MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE
            )
            if not allocation:
                raise _win_error("VirtualAllocEx(input bridge)")
            self.cave = int(allocation)
            self.control = self.cave + CONTROL_OFFSET
            self._write(self.cave, build_stub(self.cave))
            self._write(self.control, bytes(256))
            hook = b"\xE9" + rel32(HOOK_ADDRESS, self.cave)
            with self._suspended():
                if self.reader.read(HOOK_ADDRESS, len(HOOK_ORIGINAL)) != HOOK_ORIGINAL:
                    raise RuntimeError("input hook site changed before publication")
                self._write_code(HOOK_ADDRESS, hook)
                if self.reader.read(HOOK_ADDRESS, len(hook)) != hook:
                    raise RuntimeError("input hook verification failed")
            self.installed = True
        except Exception:
            if self.handle and self.cave:
                current = self.reader.read(HOOK_ADDRESS, len(HOOK_ORIGINAL))
                if current[:1] == b"\xE9":
                    with self._suspended():
                        self._write_code(HOOK_ADDRESS, HOOK_ORIGINAL)
            self._discard_allocation()
            if self.handle:
                self.api.kernel32.CloseHandle(self.handle)
                self.handle = None
            raise

    def _recover_verified_orphan(self, hook: bytes) -> None:
        """Clear an exact bridge left by a hard-killed controller process."""
        try:
            cave = jump_target(HOOK_ADDRESS, hook)
        except ValueError as exc:
            raise RuntimeError(
                f"input hook site mismatch at {HOOK_ADDRESS:#x}: {hook.hex(' ')}"
            ) from exc
        expected = build_stub(cave)
        if self.reader.read(cave, len(expected)) != expected:
            raise RuntimeError(
                f"refusing unknown hook target {cave:#x} at {HOOK_ADDRESS:#x}"
            )
        access = (
            PROCESS_VM_READ | PROCESS_VM_WRITE | PROCESS_VM_OPERATION
            | PROCESS_SUSPEND_RESUME
            | PROCESS_QUERY_LIMITED_INFORMATION
        )
        self.handle = self.api.kernel32.OpenProcess(access, False, self.reader.pid)
        if not self.handle:
            raise _win_error("OpenProcess(orphan input bridge)")
        self.cave = cave
        self.control = cave + CONTROL_OFFSET
        self.installed = True
        self.close()
        if self.reader.read(HOOK_ADDRESS, len(HOOK_ORIGINAL)) != HOOK_ORIGINAL:
            raise RuntimeError("orphan input bridge recovery did not restore hook")

    def set_keys(self, names: set[str]) -> None:
        if not self.installed or self.control is None:
            raise RuntimeError("input bridge is not installed")
        unknown = names - KEYS.keys()
        if unknown:
            raise ValueError(f"unknown keys: {sorted(unknown)}")
        overlay = bytearray(256)
        for name in names:
            overlay[KEYS[name].dik_code] = 0x80
        self._write(self.control, bytes(overlay))

    def release_all(self) -> None:
        if self.installed and self.control is not None:
            try:
                self._write(self.control, bytes(256))
            except OSError:
                # A dead target owns no live overlay or held keys.  Do not mask
                # the original process-loss error with a cleanup traceback.
                if self._process_alive():
                    raise

    def _discard_allocation(self) -> None:
        if self.cave and self.handle:
            self.api.kernel32.VirtualFreeEx(
                self.handle, ctypes.c_void_p(self.cave), 0, MEM_RELEASE
            )
        self.cave = None
        self.control = None

    def close(self) -> None:
        if not self.handle:
            return
        try:
            self.release_all()
            if self.installed and self._process_alive():
                with self._suspended():
                    self._write_code(HOOK_ADDRESS, HOOK_ORIGINAL)
                    if self.reader.read(HOOK_ADDRESS, len(HOOK_ORIGINAL)) != HOOK_ORIGINAL:
                        raise RuntimeError("input hook restoration verification failed")
                self.installed = False
                # Let any thread already inside the tiny stub retire before free.
                time.sleep(0.05)
            elif self.installed:
                self.installed = False
            self._discard_allocation()
        finally:
            self.api.kernel32.CloseHandle(self.handle)
            self.handle = None

    def __enter__(self) -> "InjectedInputBridge":
        self.install()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class InjectedKeyboard:
    def __init__(
        self,
        api: Win32,
        pid: int,
        bridge: InjectedInputBridge,
        *,
        foreground_required: bool = True,
    ) -> None:
        self.api = api
        self.pid = pid
        self.bridge = bridge
        self.foreground_required = foreground_required
        self.held: set[str] = set()

    def require_foreground(self) -> None:
        if self.foreground_required and self.api.foreground_pid() != self.pid:
            raise RuntimeError("TH105 lost foreground ownership")

    def set_chord(self, names: set[str]) -> None:
        self.require_foreground()
        self.bridge.set_keys(names)
        self.held = set(names)

    def tap(self, name: str, hold_ms: int = 65, gap_ms: int = 170) -> None:
        self.set_chord({name})
        try:
            time.sleep(hold_ms / 1000.0)
        finally:
            self.bridge.release_all()
            self.held.clear()
        time.sleep(gap_ms / 1000.0)

    def hold_chord(self, names: set[str], seconds: float) -> None:
        self.set_chord(names)
        try:
            time.sleep(seconds)
        finally:
            self.bridge.release_all()
            self.held.clear()

    def release_all(self, *, require_foreground: bool = False) -> None:
        if require_foreground:
            self.require_foreground()
        self.bridge.release_all()
        self.held.clear()
