from __future__ import annotations

import ctypes
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.win32 import (
    BACKGROUND_EXSTYLE_ADDRESS,
    BACKGROUND_EXSTYLE_ORIGINAL,
    BACKGROUND_EXSTYLE_PATCHED,
    CREATE_SUSPENDED,
    STARTF_USESHOWWINDOW,
    SW_SHOWNOACTIVATE,
    PROCESS_INFORMATION,
    STARTUPINFOW,
    Win32,
)


class _Kernel:
    def __init__(self) -> None:
        self.flags: int | None = None
        self.show: int | None = None
        self.closed: list[int] = []
        self.creation_flags: int | None = None

    def CreateProcessW(self, *_args: object) -> bool:
        startup_pointer = _args[-2]
        info_pointer = _args[-1]
        startup = ctypes.cast(startup_pointer, ctypes.POINTER(STARTUPINFOW)).contents
        info = ctypes.cast(info_pointer, ctypes.POINTER(PROCESS_INFORMATION)).contents
        self.flags = int(startup.dwFlags)
        self.show = int(startup.wShowWindow)
        self.creation_flags = int(_args[5])
        info.dwProcessId = 4242
        info.hThread = 10
        info.hProcess = 11
        return True

    def CloseHandle(self, handle: object) -> None:
        self.closed.append(int(handle))


class BackgroundLaunchTests(unittest.TestCase):
    def test_background_launch_requests_show_without_activation(self) -> None:
        api = Win32.__new__(Win32)
        kernel = _Kernel()
        api.kernel32 = kernel
        pid = api.launch(Path("th105c.exe"), activate=False)
        self.assertEqual(pid, 4242)
        self.assertEqual(kernel.flags, STARTF_USESHOWWINDOW)
        self.assertEqual(kernel.show, SW_SHOWNOACTIVATE)

    def test_exact_build_background_patch_is_installed_before_resume(self) -> None:
        class PatchKernel(_Kernel):
            def __init__(self) -> None:
                super().__init__()
                self.memory = BACKGROUND_EXSTYLE_ORIGINAL
                self.resumed = False
                self.terminated = False

            def ReadProcessMemory(self, _process, address, buffer, size, count):
                self.assert_address(address)
                ctypes.memmove(buffer, self.memory, size)
                ctypes.cast(
                    count, ctypes.POINTER(ctypes.c_size_t)
                ).contents.value = size
                return True

            def VirtualProtectEx(self, _process, address, _size, _new, old):
                self.assert_address(address)
                ctypes.cast(old, ctypes.POINTER(ctypes.c_ulong)).contents.value = 0x20
                return True

            def WriteProcessMemory(self, _process, address, buffer, size, count):
                self.assert_address(address)
                self.memory = ctypes.string_at(buffer, size)
                ctypes.cast(
                    count, ctypes.POINTER(ctypes.c_size_t)
                ).contents.value = size
                return True

            def FlushInstructionCache(self, _process, address, _size):
                self.assert_address(address)
                return True

            def ResumeThread(self, _thread):
                self.resumed = True
                return 1

            def TerminateProcess(self, _process, _code):
                self.terminated = True
                return True

            def WaitForSingleObject(self, _process, _timeout):
                return 0

            def assert_address(self, address):
                if int(address) != BACKGROUND_EXSTYLE_ADDRESS:
                    raise AssertionError(address)

        api = Win32.__new__(Win32)
        kernel = PatchKernel()
        api.kernel32 = kernel
        with (
            mock.patch("th105.win32.sha256", return_value="expected"),
            mock.patch("th105.win32.EXPECTED_EXE_SHA256", "expected"),
        ):
            pid = api.launch(
                Path("th105c.exe"), activate=False, prevent_activation=True
            )
        self.assertEqual(pid, 4242)
        self.assertEqual(kernel.creation_flags, CREATE_SUSPENDED)
        self.assertEqual(kernel.memory, BACKGROUND_EXSTYLE_PATCHED)
        self.assertTrue(kernel.resumed)
        self.assertFalse(kernel.terminated)


if __name__ == "__main__":
    unittest.main()
