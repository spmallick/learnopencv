import pyautogui
import pyperclip
import pygetwindow as gw
import pytesseract
import psutil
import os
import time
import platform
from PIL import Image, ImageGrab

predefined_paths = {
    "whatsapp": "whatsapp://",
    "chrome": "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "edge": "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "sublime text": "C:/Program Files/Sublime Text 3/sublime_text.exe"
}

def scroll(amount):
    pyautogui.scroll(amount)


def move_mouse(x, y, duration=0.2):
    pyautogui.moveTo(x, y, duration=duration)


def click(x=None, y=None, button="left"):
    pyautogui.click(x, y, button=button)


def double_click(x=None, y=None):
    pyautogui.doubleClick(x, y)


def right_click(x=None, y=None):
    pyautogui.rightClick(x, y)


def drag_and_drop(x1, y1, x2, y2, duration=0.5):
    pyautogui.moveTo(x1, y1)
    pyautogui.dragTo(x2, y2, duration=duration)


def press_key(key):
    pyautogui.press(key)


def hotkey(*keys):
    pyautogui.hotkey(*keys)


def type_text(text):
    pyautogui.typewrite(text, interval=0.05)


def clear_field():
    pyautogui.hotkey("ctrl", "a")
    pyautogui.press("backspace")


def launch_app(path):
    resolved_path = predefined_paths.get(path.lower(), path)
    system = platform.system()

    try:
        if system == "Windows":
            os.startfile(resolved_path)
        elif system == "Darwin":  # macOS
            os.system(f"open -a '{resolved_path}'")
        elif system == "Linux":
            os.system(f"xdg-open '{resolved_path}'")
        else:
            raise Exception("Unsupported OS")

        print(f"✅ Launched {path} via predefined path")
        time.sleep(2)
        return True

    except Exception as e:
        print(f"❌ Could not launch {path} via predefined path: {e}")
        print(f"⚡ Attempting search fallback...")

        try:
            if system == "Windows":
                pyautogui.press("win")
                time.sleep(1)
                pyautogui.typewrite(path, interval=0.05)
                time.sleep(0.5)
                pyautogui.press("enter")

            elif system == "Darwin":  # macOS Spotlight
                pyautogui.hotkey("command", "space")
                time.sleep(1)
                pyautogui.typewrite(path, interval=0.05)
                time.sleep(0.5)
                pyautogui.press("enter")

            elif system == "Linux":  # Ubuntu GNOME/KDE (Super key opens Activities)
                pyautogui.press("win")
                time.sleep(1)
                pyautogui.typewrite(path, interval=0.05)
                time.sleep(0.5)
                pyautogui.press("enter")

            time.sleep(3)
            return True

        except Exception as ex:
            print(f"❌ Fallback failed for {path}: {ex}")
            return False


def is_process_running(name):
    return any(name.lower() in p.name().lower() for p in psutil.process_iter())

def kill_process(name):
    for p in psutil.process_iter():
        if name.lower() in p.name().lower():
            p.terminate()
            return True
    return False

def sleep(seconds=None):
    if seconds is not None:
        time.sleep(seconds)

