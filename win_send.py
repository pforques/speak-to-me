import ctypes
import time
from dataclasses import dataclass
from typing import Optional

import win32api
import win32con
import win32console
import win32gui
import win32process

try:
    import psutil  # optional, for diagnostics
except Exception:  # pragma: no cover
    psutil = None  # type: ignore


TARGET_CLASSES = {"ConsoleWindowClass", "PseudoConsoleWindow", "CASCADIA_HOSTING_WINDOW_CLASS"}


@dataclass
class TargetWindow:
    hwnd: int
    title: str


def find_window_by_pid(pid: int) -> Optional[TargetWindow]:
    """Return the first visible console-like window for the given PID."""
    match: Optional[TargetWindow] = None

    def enum_cb(hwnd, _):
        nonlocal match
        if match is not None:
            return
        if not win32gui.IsWindowVisible(hwnd):
            return
        cls = win32gui.GetClassName(hwnd)
        if cls not in TARGET_CLASSES:
            return
        _, wpid = win32process.GetWindowThreadProcessId(hwnd)
        if wpid != pid:
            return
        title = win32gui.GetWindowText(hwnd) or ""
        match = TargetWindow(hwnd=hwnd, title=title)

    win32gui.EnumWindows(enum_cb, None)
    return match


def _set_foreground(hwnd: int) -> bool:
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        fg = win32gui.GetForegroundWindow()
        cur_thread = win32api.GetCurrentThreadId()
        fg_thread = win32process.GetWindowThreadProcessId(fg)[0] if fg else cur_thread
        tgt_thread = win32process.GetWindowThreadProcessId(hwnd)[0]
        user32 = ctypes.windll.user32
        user32.AttachThreadInput(cur_thread, fg_thread, True)
        user32.AttachThreadInput(cur_thread, tgt_thread, True)
        win32gui.SetForegroundWindow(hwnd)
        user32.AttachThreadInput(cur_thread, fg_thread, False)
        user32.AttachThreadInput(cur_thread, tgt_thread, False)
        return win32gui.GetForegroundWindow() == hwnd
    except Exception:
        return False


def _send_char(c: str, delay: float) -> None:
    vk = win32api.VkKeyScan(c)
    if vk == -1:
        return
    shift = bool(vk & 0x0100)
    vk_code = vk & 0xFF
    if shift:
        win32api.keybd_event(win32con.VK_SHIFT, 0, 0, 0)
    win32api.keybd_event(vk_code, 0, 0, 0)
    time.sleep(delay)
    win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)
    if shift:
        win32api.keybd_event(win32con.VK_SHIFT, 0, win32con.KEYEVENTF_KEYUP, 0)
    time.sleep(delay)


def send_text_to_console_pid(pid: int, text: str, submit: bool = True, delay: float = 0.02) -> bool:
    """Inject key events directly into a console's input buffer via WriteConsoleInput."""
    try:
        win32console.FreeConsole()
    except Exception:
        pass
    try:
        win32console.AttachConsole(pid)
    except Exception:
        return False
    try:
        h = win32console.GetStdHandle(win32console.STD_INPUT_HANDLE)
        records = []
        for ch in text:
            vk = win32api.VkKeyScan(ch)
            if vk == -1:
                continue
            vk_code = vk & 0xFF
            rec_down = win32console.PyINPUT_RECORDType(win32console.KEY_EVENT)
            rec_down.KeyDown = True
            rec_down.RepeatCount = 1
            rec_down.ControlKeyState = 0
            rec_down.VirtualKeyCode = vk_code
            rec_down.VirtualScanCode = 0
            rec_down.Char = ch
            rec_up = win32console.PyINPUT_RECORDType(win32console.KEY_EVENT)
            rec_up.KeyDown = False
            rec_up.RepeatCount = 1
            rec_up.ControlKeyState = 0
            rec_up.VirtualKeyCode = vk_code
            rec_up.VirtualScanCode = 0
            rec_up.Char = ch
            records.extend([rec_down, rec_up])
        if submit:
            for ch, vk_code in [('\r', win32con.VK_RETURN), ('\n', win32con.VK_RETURN)]:
                rec_down = win32console.PyINPUT_RECORDType(win32console.KEY_EVENT)
                rec_down.KeyDown = True
                rec_down.RepeatCount = 1
                rec_down.ControlKeyState = 0
                rec_down.VirtualKeyCode = vk_code
                rec_down.VirtualScanCode = 0
                rec_down.Char = ch
                rec_up = win32console.PyINPUT_RECORDType(win32console.KEY_EVENT)
                rec_up.KeyDown = False
                rec_up.RepeatCount = 1
                rec_up.ControlKeyState = 0
                rec_up.VirtualKeyCode = vk_code
                rec_up.VirtualScanCode = 0
                rec_up.Char = ch
                records.extend([rec_down, rec_up])
        h.WriteConsoleInput(records)
        time.sleep(delay)
        return True
    except Exception:
        return False
    finally:
        try:
            win32console.FreeConsole()
        except Exception:
            pass


def send_text_to_pid(pid: int, text: str, submit: bool = True, delay: float = 0.02) -> bool:
    target = find_window_by_pid(pid)
    if not target:
        print(f"[send] aucune fenêtre visible pour le PID {pid}")
        return False

    classname = win32gui.GetClassName(target.hwnd)
    if classname == "ConsoleWindowClass":
        ok = send_text_to_console_pid(pid, text, submit=submit, delay=delay)
        if ok:
            return True

    if not _set_foreground(target.hwnd):
        print(f"[send] impossible de mettre la fenêtre au premier plan (pid={pid})")
        return False
    time.sleep(0.05)

    try:
        for ch in text:
            _send_char(ch, delay)
        if submit:
            win32api.keybd_event(win32con.VK_RETURN, 0, 0, 0)
            time.sleep(delay)
            win32api.keybd_event(win32con.VK_RETURN, 0, win32con.KEYEVENTF_KEYUP, 0)
        return True
    except Exception as exc:  # pragma: no cover
        print(f"[send] erreur d'envoi: {exc}")
        return False
