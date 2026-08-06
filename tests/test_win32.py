from __future__ import annotations

import ctypes
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from th105.win32 import (
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

    def CreateProcessW(self, *_args: object) -> bool:
        startup_pointer = _args[-2]
        info_pointer = _args[-1]
        startup = ctypes.cast(startup_pointer, ctypes.POINTER(STARTUPINFOW)).contents
        info = ctypes.cast(info_pointer, ctypes.POINTER(PROCESS_INFORMATION)).contents
        self.flags = int(startup.dwFlags)
        self.show = int(startup.wShowWindow)
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


if __name__ == "__main__":
    unittest.main()
