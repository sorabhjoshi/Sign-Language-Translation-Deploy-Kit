"""Browser version of the real-time sign recognizer, for free hosting on Hugging Face Spaces.

The visitor's webcam frames stream to this app, MediaPipe and the hybrid model run on the server CPU, and the page shows
the subtitle and speaks each recognised word with the browser's own speech synthesis (so no audio is generated on the server).
"""
import collections, html, os, threading, time
import gradio as gr
import mediapipe as mp
import torch

import slt_realtime as rt
from slt_landmarks import SignSegmenter, holistic_to_frame, process_window

MODEL = os.environ.get("SLT_MODEL", "slt_deploy_50signs.pt")
CONF, MARGIN, MAX_SESSIONS = 0.45, 0.10, 4
torch.set_num_threads(2)
pred = rt.Predictor(MODEL, "cpu")
mph = mp.solutions.holistic


class Session:
    """One visitor: own MediaPipe tracker, segmenter and subtitle state."""
    def __init__(self):
        self.holistic = mph.Holistic(static_image_mode=False, model_complexity=1, smooth_landmarks=True,        # 1 ships inside the wheel; 0 and 2 download a file at run time
                                     refine_face_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        # frames arrive at only a few per second over the network, so the timing thresholds are looser than on the desktop
        self.seg = SignSegmenter(rest_y=1.4, start_s=0.15, end_gap_s=0.6, min_sign_s=0.4, max_sign_s=4.0)
        self.words, self.last_word, self.last_t = collections.deque(maxlen=12), None, -1e9
        self.n, self.last_event, self.top3, self.lock = 0, time.perf_counter(), [], threading.Lock()

    def close(self):
        try: self.holistic.close()
        except Exception: pass


SESSIONS, SLOCK = collections.OrderedDict(), threading.Lock()
def get_session(key):
    with SLOCK:
        if key in SESSIONS:
            SESSIONS.move_to_end(key); return SESSIONS[key]
        s = SESSIONS[key] = Session()
        while len(SESSIONS) > MAX_SESSIONS:
            SESSIONS.popitem(last=False)[1].close()
        return s


def render(s):
    text = html.escape(" ".join(s.words)) or "&nbsp;"
    sub = f"<div style='font-size:2.2rem;line-height:1.3;min-height:3rem;padding:.5rem;border-radius:8px;background:#111;color:#fff'>{text}</div>"
    state = {"idle": "waiting for a sign", "signing": "capturing a sign..."}[s.seg.state]
    top = "".join(f"<br>{html.escape(w)}: {p:.2f}" for w, p in s.top3)
    return sub, f"**{state}**{top}"


def process_frame(frame, request: gr.Request):
    if frame is None: return gr.update(), gr.update(), gr.update()
    s = get_session(request.session_hash if request is not None else "local")
    with s.lock:
        t = time.perf_counter()
        res = s.holistic.process(frame)                                  # Gradio gives RGB uint8
        seg = s.seg.update(t, holistic_to_frame(res))
        speak = gr.update()
        if seg is not None:
            probs = pred.predict(*process_window(seg["frames"]))
            ok, word, s.top3 = rt.decide(probs, pred.signs, CONF, MARGIN); word = rt.pretty(word)
            s.last_event = time.perf_counter()
            if ok and not (word == s.last_word and t - s.last_t < 0.8):
                s.words.append(word); s.last_word, s.last_t, s.n = word, t, s.n + 1
                speak = f"{s.n}|{word}"                                  # the counter makes repeated words fire the change event again
        if s.words and time.perf_counter() - s.last_event > 6 and s.seg.state == "idle": s.words.clear()
        sub, status = render(s)
    return sub, speak, status


SPEAK_JS = """(v, on) => { if (!v || !on) return; const w = v.split('|').slice(1).join('|');
  if (w) { const u = new SpeechSynthesisUtterance(w); u.rate = 1.0; speechSynthesis.cancel(); speechSynthesis.speak(u); } }"""

with gr.Blocks(css="#speak {display:none}", title="Sign language translator") as demo:
    gr.Markdown("# Real-time sign language translator\nAllow the camera, sign one word at a time and lower your hands between signs. "
                f"Recognised words appear as a subtitle and are spoken. Vocabulary ({len(pred.signs)} ASL signs): {', '.join(pred.signs)}.\n\n"
                "Your video is processed on the server and is not stored.")
    with gr.Row():
        cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Camera", mirror_webcam=False)
        with gr.Column():
            sub = gr.HTML()
            status = gr.Markdown()
            voice = gr.Checkbox(True, label="Speak the words")
            test = gr.Button("Test voice")
    speak_box = gr.Textbox(elem_id="speak", show_label=False)
    cam.stream(process_frame, [cam], [sub, speak_box, status], show_progress="hidden", trigger_mode="always_last", concurrency_limit=4)
    speak_box.change(None, [speak_box, voice], None, js=SPEAK_JS)
    test.click(None, None, None, js="() => { speechSynthesis.speak(new SpeechSynthesisUtterance('Voice enabled')); }")

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
