"""Subtitles, status overlay and speech for the real-time app (OpenCV + a background speech thread; no torch)."""
import queue, shutil, subprocess, sys, threading
import cv2
import numpy as np


class Speaker:
    """Speaks text on a background thread so the video never stalls. Uses `say` on macOS, pyttsx3 elsewhere,
    and falls back to espeak-ng / espeak / spd-say on Linux."""
    def __init__(self, enabled=True, rate=170):
        self.enabled, self.rate, self.q = enabled, rate, queue.Queue()
        if enabled: threading.Thread(target=self._worker, daemon=True).start()

    def say(self, text):
        if not (self.enabled and text): return
        while self.q.qsize() > 3:                              # never fall far behind the video
            try: self.q.get_nowait()
            except queue.Empty: break
        self.q.put(text)

    def _backend(self):
        if sys.platform == "darwin" and shutil.which("say"):
            return lambda s: subprocess.run(["say", "-r", str(self.rate), s], check=False)
        if sys.platform == "win32":
            try:
                import pythoncom; pythoncom.CoInitialize()
            except Exception:
                pass
        try:
            import pyttsx3
            eng = pyttsx3.init(); eng.setProperty("rate", self.rate)
            def speak(s): eng.say(s); eng.runAndWait()
            return speak
        except Exception as e:
            print("[tts] pyttsx3 unavailable:", repr(e)[:120])
        for exe in ("espeak-ng", "espeak", "spd-say"):
            if shutil.which(exe): return lambda s, exe=exe: subprocess.run([exe, s], check=False)
        print("[tts] no speech engine found: pip install pyttsx3 (or install espeak-ng on Linux)")
        return lambda s: None

    def _worker(self):
        speak = self._backend()
        while True:
            text = self.q.get()
            if text is None: return
            try: speak(text)
            except Exception as e: print("[tts] failed:", repr(e)[:120])


def _outlined(img, text, org, scale, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def wrap_tail(words, max_w, scale, max_lines=2):
    """The most recent words that fit in `max_lines` lines of width `max_w` pixels."""
    width = lambda s: cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0]
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if width(trial) <= max_w or not cur: cur = trial
        else: lines.append(cur); cur = w
    lines.append(cur)
    return [l for l in lines[-max_lines:] if l]


def draw_subtitles(frame, words, candidate=None, scale=None):
    """Subtitle band at the bottom: recent words in white, the candidate that is being recognised in grey."""
    h, w = frame.shape[:2]
    scale = scale or max(0.8, w / 900)
    shown = list(words) + ([candidate] if candidate else [])
    lines = wrap_tail(shown, int(w * 0.92), scale)
    if not lines: return frame
    lh = int(cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][1] * 1.7)
    top = h - lh * len(lines) - 24
    band = frame.copy(); cv2.rectangle(band, (0, top), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(band, 0.55, frame, 0.45, 0)
    for i, line in enumerate(lines):
        tw = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0]
        last = candidate is not None and i == len(lines) - 1
        _outlined(frame, line, ((w - tw) // 2, top + 12 + lh * (i + 1) - 8), scale, (190, 190, 190) if last else (255, 255, 255))
    return frame


def draw_hud(frame, lines, scale=0.55):
    for i, l in enumerate(lines): _outlined(frame, l, (10, 22 + 22 * i), scale, (80, 255, 120), 1)
    return frame
