"""
Liste les fenêtres console/pseudoconsole visibles (ConsoleWindowClass, PseudoConsoleWindow, Cascadia).
Affiche titre, classe, PID, hwnd et cmdline (si psutil dispo).
"""

import win32gui
import win32process

try:
    import psutil
except Exception:
    psutil = None  # type: ignore


TARGET_CLASSES = {"ConsoleWindowClass", "PseudoConsoleWindow", "CASCADIA_HOSTING_WINDOW_CLASS"}


def main() -> None:
    rows = []

    def enum_cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        cls = win32gui.GetClassName(hwnd)
        if cls not in TARGET_CLASSES:
            return
        title = win32gui.GetWindowText(hwnd) or "<empty>"
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        cmdline = ""
        if psutil:
            try:
                cmdline = " ".join(psutil.Process(pid).cmdline())
            except Exception:
                cmdline = ""
        rows.append((title, cls, pid, hex(hwnd), cmdline))

    win32gui.EnumWindows(enum_cb, None)

    if not rows:
        print("No visible console windows.")
        return

    for title, cls, pid, hwnd, cmd in rows:
        if cmd:
            print(f"{title} | class={cls} | pid={pid} | hwnd={hwnd} | cmdline={cmd}")
        else:
            print(f"{title} | class={cls} | pid={pid} | hwnd={hwnd}")


if __name__ == "__main__":
    main()
