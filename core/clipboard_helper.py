from pathlib import Path
import sys
import ctypes
from ctypes import wintypes
import pyperclip

# Windows Shell Clipboard Structures for CF_HDROP
CF_HDROP = 15
GHND = 0x0042

class DROPFILES(ctypes.Structure):
    _fields_ = [
        ("pFiles", wintypes.DWORD),
        ("pt", wintypes.POINT),
        ("fNC", wintypes.BOOL),
        ("fWide", wintypes.BOOL),
    ]

def copy_file_to_windows_clipboard(file_path: Path | str) -> dict:
    """
    Copies a file to the Windows clipboard both as:
    1. CF_HDROP (standard Windows file drop format for NLEs like Premiere Pro, DaVinci Resolve)
    2. Plaintext string file path (via pyperclip)
    """
    path = Path(file_path).resolve()
    if not path.exists():
        return {"success": False, "error": f"File does not exist: {path}"}
    
    path_str = str(path)
    
    # Always copy plaintext path first as universal fallback
    try:
        pyperclip.copy(path_str)
    except Exception as e:
        print(f"[Clipboard] pyperclip error: {e}")

    # Attempt native Windows CF_HDROP clipboard format
    if sys.platform == "win32":
        try:
            # Prepare double-null terminated wide string buffer
            encoded_path = (path_str + "\0\0").encode("utf-16-le")
            df_size = ctypes.sizeof(DROPFILES)
            total_size = df_size + len(encoded_path)

            kernel32 = ctypes.windll.kernel32
            user32 = ctypes.windll.user32

            h_mem = kernel32.GlobalAlloc(GHND, total_size)
            if not h_mem:
                return {"success": True, "mode": "text_path_only", "path": path_str}

            ptr = kernel32.GlobalLock(h_mem)
            if not ptr:
                kernel32.GlobalFree(h_mem)
                return {"success": True, "mode": "text_path_only", "path": path_str}

            # Fill DROPFILES struct
            df = DROPFILES()
            df.pFiles = df_size
            df.fWide = True
            ctypes.memmove(ptr, ctypes.byref(df), df_size)
            ctypes.memmove(ptr + df_size, encoded_path, len(encoded_path))
            kernel32.GlobalUnlock(h_mem)

            if user32.OpenClipboard(None):
                user32.EmptyClipboard()
                user32.SetClipboardData(CF_HDROP, h_mem)
                user32.CloseClipboard()
                return {"success": True, "mode": "native_file_drop", "path": path_str}
        except Exception as ex:
            print(f"[Clipboard] Native Windows clipboard warning: {ex}")

    return {"success": True, "mode": "text_path", "path": path_str}
