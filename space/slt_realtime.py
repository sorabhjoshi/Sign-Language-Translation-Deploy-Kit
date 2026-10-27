"""Real-time sign recognition: webcam -> MediaPipe Holistic -> isolated-sign segmentation -> hybrid quantum-classical
model -> subtitles on screen and speech.

    python slt_realtime.py --model slt_deploy_50signs.pt --selftest          # checks the whole path without a camera
    python slt_realtime.py --model slt_deploy_50signs.pt --video clip.mp4    # offline test on a video file
    python slt_realtime.py --model slt_deploy_50signs.pt                     # webcam

Model code and feature code below are copied verbatim from the training notebook so that the saved weights load unchanged.
"""
import argparse, math, queue, sys, threading, time
from collections import deque
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

from slt_landmarks import N_PTS, SignSegmenter, holistic_to_frame, process_window

T_CACHE = 64

# ================= copied verbatim from the training notebook (batch pipeline) =================
MIRROR_PERM = None
def set_lip_perm(lip_perm):
    """The left/right partner of each lip point is learned from the data at training time and stored in the checkpoint."""
    global MIRROR_PERM
    LIP_PERM = lip_perm
    MIRROR_PERM = list(range(21, 42)) + list(range(0, 21)) + [42, 44, 43, 46, 45, 48, 47] + [49 + j for j in LIP_PERM]

def mirror(x, p):
    x = x[:, :, MIRROR_PERM].clone(); x[..., 0] = -x[..., 0]
    return x, p[:, :, [1, 0]]

def temporal_sample(x, p, T_out, frac=None):
    B, T = x.shape[:2]; dev = x.device
    ar = torch.arange(T_out, device=dev, dtype=torch.float32) / (T_out - 1)
    if frac is None:
        idx = ar.unsqueeze(0).expand(B, -1) * (T - 1)
    else:
        L = torch.empty(B, 1, device=dev).uniform_(*frac) * (T - 1)
        idx = torch.rand(B, 1, device=dev) * ((T - 1) - L) + L * ar
    i0 = idx.floor().long().clamp(0, T - 1); i1 = (i0 + 1).clamp(max=T - 1)
    w = (idx - i0.float()).view(B, T_out, 1, 1); bi = torch.arange(B, device=dev).unsqueeze(1)
    return x[bi, i0] * (1 - w) + x[bi, i1] * w, p[bi, idx.round().long().clamp(0, T - 1)]

def spatial_aug(x, p, rot_deg, scale, shift, jitter, mirror_p, shear=0.0):
    B, dev = x.shape[0], x.device
    if mirror_p > 0:
        xm, pm = mirror(x, p); f = torch.rand(B, device=dev) < mirror_p
        x = torch.where(f.view(B, 1, 1, 1), xm, x); p = torch.where(f.view(B, 1, 1), pm, p)
    a = torch.empty(B, device=dev).uniform_(-rot_deg, rot_deg) * math.pi / 180
    s = torch.empty(B, device=dev).uniform_(*scale)
    c, sn = (torch.cos(a) * s).view(B, 1, 1), (torch.sin(a) * s).view(B, 1, 1)
    x = torch.stack([x[..., 0] * c - x[..., 1] * sn, x[..., 0] * sn + x[..., 1] * c, x[..., 2] * s.view(B, 1, 1)], -1)
    if shear > 0:
        sh = torch.empty(B, 1, 1, device=dev).uniform_(-shear, shear)
        x = torch.stack([x[..., 0] + sh * x[..., 1], x[..., 1], x[..., 2]], -1)
    if shift > 0:  x = x + torch.empty(B, 1, 1, 3, device=dev).uniform_(-shift, shift)
    if jitter > 0:
        nz = torch.randn_like(x) * jitter; nz[:, :, 49:] *= 0.2; x = x + nz   # lips live at a small scale: less jitter
    return x, p

def time_mask(x, p, prob, max_frac=0.12):
    B, T = x.shape[:2]; dev = x.device
    ln = (torch.rand(B, 1, device=dev) * max_frac * T).long()
    st = (torch.rand(B, 1, device=dev) * (T - ln)).long()
    t = torch.arange(T, device=dev).unsqueeze(0)
    m = ((t >= st) & (t < st + ln) & (torch.rand(B, 1, device=dev) < prob))
    return x * (~m).view(B, T, 1, 1), p & (~m).unsqueeze(-1)

VEL = 10.0
TIPS, TRIU = [0, 4, 8, 12, 16, 20], torch.triu_indices(6, 6, 1)
HAND_D, POSE_D, LIP_D = 205, 42, 246
def hand_feats(h, ph):
    loc = h - h[:, :, :1]
    size = (h[:, :, 9] - h[:, :, 0]).norm(dim=-1).clamp(min=1e-2)
    loc = loc / size[..., None, None]
    v = torch.zeros_like(h); v[:, 1:] = (h[:, 1:] - h[:, :-1]) * VEL * (ph[:, 1:] * ph[:, :-1])[..., None, None]
    tp = h[:, :, TIPS]                                                   # wrist + 5 fingertips
    iu = TRIU.to(h.device)
    dd = (tp.unsqueeze(3) - tp.unsqueeze(2)).norm(dim=-1)[:, :, iu[0], iu[1]] / size[..., None]   # 15 pairwise distances
    return torch.cat([h.flatten(2), loc.flatten(2), v.flatten(2), ph.unsqueeze(-1), dd], -1)

def pose_feats(q):
    v = torch.zeros_like(q); v[:, 1:] = (q[:, 1:] - q[:, :-1]) * VEL
    return torch.cat([q.flatten(2), v.flatten(2)], -1)

def lip_feats(q, lp):
    """Lip shape (centred, mouth-width normalised) + its motion + mouth position/motion. lp = frame has lips."""
    c = q.mean(2, keepdim=True)
    size = (q[:, :, 0] - q[:, :, 10]).norm(dim=-1).clamp(min=1e-2)
    loc = (q - c) / size[..., None, None] * lp[..., None, None]
    both = (lp[:, 1:] * lp[:, :-1])[..., None, None]
    v = torch.zeros_like(loc); v[:, 1:] = (loc[:, 1:] - loc[:, :-1]) * VEL * both
    cv = torch.zeros_like(c); cv[:, 1:] = (c[:, 1:] - c[:, :-1]) * VEL * both
    return torch.cat([loc.flatten(2), v.flatten(2), c.flatten(2), cv.flatten(2)], -1)

def canon_dominant(x, p):
    """Mirror clips where the left hand is the more-used hand, so every signer looks right-handed."""
    flip = p[..., 0].float().sum(1) > p[..., 1].float().sum(1)
    xm, pm = mirror(x, p)
    return torch.where(flip.view(-1, 1, 1, 1), xm, x), torch.where(flip.view(-1, 1, 1), pm, p)

def make_inputs(cfg, xb, pb, train):
    x, p = xb.float(), pb
    x, p = temporal_sample(x, p, T_CACHE, cfg["warp"] if train else None)
    x, p = canon_dominant(x, p)
    lp = x[:, :, 49:].abs().sum((2, 3)) > 0                                   # frames where lips really exist
    if train:
        x, p = spatial_aug(x, p, **cfg["aug"]); x, p = time_mask(x, p, cfg["tmask"])
    lp = lp & (x[:, :, 49:].abs().sum((2, 3)) > 0)
    x = torch.cat([x[:, :, :42] * p.repeat_interleave(21, dim=2).unsqueeze(-1).to(x.dtype), x[:, :, 42:49],
                   x[:, :, 49:] * lp[..., None, None].to(x.dtype)], 2)
    pf = p.to(x.dtype)
    out = [hand_feats(x[:, :, :21], pf[..., 0]), hand_feats(x[:, :, 21:42], pf[..., 1])]
    if cfg.get("use_pose", False): out.append(pose_feats(x[:, :, 42:49]))
    if cfg.get("use_lips", False): out.append(lip_feats(x[:, :, 49:], lp.to(x.dtype)))
    return tuple(out)


# ================= copied verbatim from the training notebook (model) =================
class ConvBlock(nn.Module):
    def __init__(self, d, k=5, drop=0.1):
        super().__init__()
        self.norm, self.dw = nn.LayerNorm(d), nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.pw, self.drop = nn.Linear(d, d), nn.Dropout(drop)
    def forward(self, x):
        y = self.dw(self.norm(x).transpose(1, 2)).transpose(1, 2)
        return x + self.drop(F.gelu(self.pw(y)))

class StreamEncoder(nn.Module):
    def __init__(self, d_in, d, drop):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(drop))
        self.conv = nn.Sequential(ConvBlock(d, 5, drop), ConvBlock(d, 5, drop))
        self.rnn, self.norm = nn.LSTM(d, d // 2, batch_first=True, bidirectional=True), nn.LayerNorm(d)
    def forward(self, x):
        h = self.conv(self.proj(x))
        return self.norm(h + self.rnn(h)[0])


class StatevectorCircuit(nn.Module):
    """Batched statevector simulation of a data-reuploading variational quantum circuit."""
    def __init__(self, n_qubits, q_layers):
        super().__init__()
        self.n, self.L = int(n_qubits), int(q_layers)
        self.weights = nn.Parameter(torch.rand(self.L, self.n, 3) * 2 * math.pi)
        N = 2 ** self.n
        f = torch.arange(N)
        for i in range(self.n):
            t = self.n - 1 - (i + 1) % self.n
            f = f ^ (((f >> (self.n - 1 - i)) & 1) << t)
        inv = torch.empty_like(f); inv[f] = torch.arange(N)
        self.register_buffer("perm", inv)

    def _rot(self):
        phi, th, om = self.weights.unbind(-1)
        c, s = torch.cos(th / 2), torch.sin(th / 2)
        e = lambda a: torch.polar(torch.ones_like(a), a)
        return torch.stack([torch.stack([e(-(phi + om) / 2) * c, -e((phi - om) / 2) * s], -1),
                            torch.stack([e(-(phi - om) / 2) * s,  e((phi + om) / 2) * c], -1)], -2)

    def forward(self, angles):
        B = angles.shape[0]
        a = angles.float().view(B, self.L, self.n)
        ca, sa = torch.cos(a / 2), torch.sin(a / 2)
        ry = torch.stack([torch.stack([ca, -sa], -1), torch.stack([sa, ca], -1)], -2).to(torch.complex64)
        U = self._rot().unsqueeze(0) @ ry
        sv = torch.zeros(B, 2 ** self.n, dtype=torch.complex64, device=angles.device)
        sv[:, 0] = 1
        for l in range(self.L):
            for w in range(self.n):
                sv = torch.einsum("bij,bljr->blir", U[:, l, w],
                                 sv.view(B, 2 ** w, 2, 2 ** (self.n - w - 1))).reshape(B, -1)
            sv = sv[:, self.perm]
        return sv.real.square() + sv.imag.square()


class ClassicalHead(nn.Module):
    """Classical stand-in for one circuit (control arm): angles -> probability-like vector, small MLP."""
    def __init__(self, n_in, n_out, hid=8):
        super().__init__(); self.net = nn.Sequential(nn.Linear(n_in, hid), nn.GELU(), nn.Linear(hid, n_out))
    def forward(self, a): return torch.softmax(self.net(a.float()), -1)


class QuantumFusionLayer(nn.Module):
    """Multi-head quantum fusion: the pooled feature is projected to angles for `heads` parallel VQCs,
    their measurement probabilities are concatenated, projected back and added as a gated residual."""
    def __init__(self, d_in, n_qubits=4, q_layers=2, heads=8, kind="quantum"):
        super().__init__()
        self.H, self.n_angles, self.n_outcomes = heads, n_qubits * q_layers, 2 ** n_qubits
        self.angle_proj = nn.Sequential(nn.Linear(d_in, 2 * heads * self.n_angles), nn.GELU(),
                                        nn.Linear(2 * heads * self.n_angles, heads * self.n_angles), nn.Tanh())
        self.circuits = nn.ModuleList([StatevectorCircuit(n_qubits, q_layers) if kind == "quantum" else ClassicalHead(self.n_angles, self.n_outcomes)
                                       for _ in range(heads)])
        self.post_proj = nn.Linear(heads * self.n_outcomes, d_in)
        self.norm = nn.LayerNorm(d_in)
        self.gate = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        ang = (self.angle_proj(x) * math.pi).view(x.shape[0], self.H, self.n_angles)
        probs = torch.cat([c(ang[:, h].contiguous()) for h, c in enumerate(self.circuits)], -1)   # (B, H * 2^n)
        return self.norm(x + self.gate * self.post_proj(probs * self.n_outcomes - 1.0))


class HybridQuantumClassicalModel(nn.Module):
    """Hands (x2) + pose + lips stream encoders -> Transformer fusion -> attention pool -> quantum fusion -> head."""
    def __init__(self, n_cls, use_pose=False, use_lips=False, d=192, layers=3, heads=4, T=T_CACHE,
                 drop=0.15, n_qubits=4, q_layers=2, q_heads=8, use_quantum=True, circuit_kind="quantum"):
        super().__init__()
        self.use_quantum = use_quantum
        self.enc_l, self.enc_r = StreamEncoder(HAND_D, d, drop), StreamEncoder(HAND_D, d, drop)
        self.extra = nn.ModuleList(([StreamEncoder(POSE_D, d, drop)] if use_pose else []) +
                                   ([StreamEncoder(LIP_D, d, drop)] if use_lips else []))
        self.stream_emb = nn.Parameter(torch.randn(2 + len(self.extra), 1, d) * 0.02)
        self.time_emb = nn.Parameter(torch.randn(1, T, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, activation="gelu", batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.pool = nn.Linear(d, 1)
        self.quantum_fusion = QuantumFusionLayer(d, n_qubits, q_layers, q_heads, circuit_kind)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, n_cls))

    def forward(self, l, r, *more):
        s = [self.enc_l(l), self.enc_r(r)] + [e(t) for e, t in zip(self.extra, more)]
        z = self.fusion(torch.cat([t + self.time_emb + self.stream_emb[i] for i, t in enumerate(s)], 1))
        a = torch.softmax(self.pool(z).squeeze(-1), 1).unsqueeze(-1)
        pooled = (a * z).sum(1)
        return self.head(self.quantum_fusion(pooled) if self.use_quantum else pooled)


def build_model(n_cls, cfg):
    """Same arguments as `build` in the notebook."""
    return HybridQuantumClassicalModel(n_cls, cfg.get("use_pose", False), cfg.get("use_lips", False),
                                       d=cfg.get("d", 192), layers=cfg.get("layers", 3),
                                       n_qubits=cfg.get("n_qubits", 4), q_layers=cfg.get("q_layers", 2),
                                       q_heads=cfg.get("q_heads", 8), use_quantum=cfg.get("use_quantum", True),
                                       circuit_kind=cfg.get("circuit_kind", "quantum"))


# ================= new code: inference wrapper, decision rule, application =================
class Predictor:
    def __init__(self, path, device):
        b = torch.load(path, map_location="cpu", weights_only=True)
        assert b["T"] == T_CACHE, f"checkpoint was trained with T={b['T']}"
        self.signs, self.cfg, self.device = list(b["signs"]), dict(b["model_cfg"]), device
        self.info = b.get("info", {})
        set_lip_perm(b["lip_perm"])
        self.model = build_model(len(self.signs), self.cfg)
        self.model.load_state_dict(b["state"]); self.model.to(device).eval()

    @torch.no_grad()
    def predict(self, X, P):
        """X (T, 89, 3) float16, P (T, 2) bool, from process_window  ->  class probabilities (n_classes,)"""
        xb = torch.from_numpy(X[None]).to(self.device); pb = torch.from_numpy(P[None]).to(self.device)
        logits = self.model(*make_inputs(self.cfg, xb, pb, False))
        return logits.float().softmax(-1)[0].cpu().numpy()


def decide(probs, signs, conf, margin):
    o = np.argsort(-probs)
    top3 = [(signs[i], float(probs[i])) for i in o[:3]]
    ok = probs[o[0]] >= conf and (probs[o[0]] - probs[o[1]]) >= margin
    return ok, signs[o[0]], top3


DISPLAY = {"weus": "we", "glasswindow": "window", "shhh": "shh"}              # dataset labels that read or sound odd
def pretty(w): return DISPLAY.get(w, w)


def selftest(pred):
    """One synthetic clip through the complete inference path (no camera, no MediaPipe)."""
    rs = np.random.RandomState(0); n = 40; kp = np.full((n, N_PTS, 3), np.nan, np.float32)
    for f in range(n):
        kp[f, 43], kp[f, 44] = [0.60, 0.50, 0], [0.40, 0.50, 0]                       # shoulders
        kp[f, 42] = [0.50, 0.30, 0]; kp[f, 45:49] = [[0.65, 0.65, 0], [0.35, 0.65, 0], [0.7, 0.8, 0], [0.3, 0.8, 0]]
        kp[f, 21:42] = np.array([0.35 + 0.1 * math.sin(f / 6), 0.45, 0]) + rs.randn(21, 3) * 0.02   # right hand moving
        kp[f, 49:] = np.array([0.5, 0.35, 0]) + rs.randn(N_PTS - 49, 3) * 0.01
    t0 = time.perf_counter(); X, P = process_window(kp); t1 = time.perf_counter()
    probs = pred.predict(X, P); t2 = time.perf_counter()
    print("window:", X.shape, X.dtype, "| hands present:", f"{P.mean():.0%}", "| probs sum:", round(float(probs.sum()), 4))
    print("top-3:", [(s, round(p, 3)) for s, p in sorted(zip(pred.signs, probs), key=lambda a: -a[1])[:3]])
    print(f"latency: preprocessing {1000*(t1-t0):.1f} ms, model {1000*(t2-t1):.1f} ms on {pred.device}")
    assert np.isfinite(probs).all(), "non-finite probabilities"
    print("self-test OK")


def main():
    ap = argparse.ArgumentParser(description="Real-time sign recognition with subtitles and speech")
    ap.add_argument("--model", required=True, help="slt_deploy_*.pt written by the export notebook")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", help="run on a video file instead of the webcam")
    ap.add_argument("--selftest", action="store_true", help="run one synthetic clip through the model and exit")
    ap.add_argument("--list-signs", action="store_true", help="print the vocabulary and exit")
    ap.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    ap.add_argument("--width", type=int, default=960); ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--conf", type=float, default=0.45, help="minimum top-1 probability to accept a word")
    ap.add_argument("--margin", type=float, default=0.10, help="minimum gap between top-1 and top-2 probability")
    ap.add_argument("--rest-y", type=float, default=1.4, help="wrists lower than this many shoulder widths below the shoulders count as resting (99 = use hand detection only)")
    ap.add_argument("--end-gap", type=float, default=0.35, help="seconds with hands down that end a sign")
    ap.add_argument("--min-sign", type=float, default=0.30); ap.add_argument("--max-sign", type=float, default=4.0)
    ap.add_argument("--repeat-s", type=float, default=0.8, help="ignore the same word again within this many seconds")
    ap.add_argument("--clear-after", type=float, default=6.0, help="clear the subtitle after this many idle seconds")
    ap.add_argument("--max-words", type=int, default=12)
    ap.add_argument("--no-tts", action="store_true"); ap.add_argument("--speak", choices=["word", "sentence"], default="word")
    ap.add_argument("--flip-input", action="store_true", help="mirror the camera image before MediaPipe (try it if recognition is poor)")
    ap.add_argument("--no-landmarks", action="store_true", help="do not draw the landmarks")
    ap.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2], help="MediaPipe model complexity: 0 is fastest and 2 most accurate, but both download a model file on first use; 1 works offline")
    ap.add_argument("--save-video", help="write the annotated video to this file (mp4)")
    a = ap.parse_args()

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu": torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    pred = Predictor(a.model, device)
    print(f"loaded {len(pred.signs)} signs, {sum(p.numel() for p in pred.model.parameters())/1e6:.2f}M parameters, device {device}", pred.info or "")
    if a.list_signs: print(", ".join(pred.signs)); return
    if a.selftest: selftest(pred); return

    import cv2, mediapipe as mp
    from slt_ui import Speaker, draw_hud, draw_subtitles
    mph, mpd = mp.solutions.holistic, mp.solutions.drawing_utils
    holistic = mph.Holistic(static_image_mode=False, model_complexity=a.complexity, smooth_landmarks=True, refine_face_landmarks=False,
                            min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(a.video if a.video else a.camera)
    if not cap.isOpened(): sys.exit("could not open the video source")
    if not a.video: cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    vid_fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) if a.video else None
    writer = None

    seg = SignSegmenter(rest_y=a.rest_y, end_gap_s=a.end_gap, min_sign_s=a.min_sign, max_sign_s=a.max_sign)
    speaker = Speaker(enabled=not a.no_tts)
    jobs, results = queue.Queue(maxsize=3), queue.Queue()

    def worker():
        while True:
            s = jobs.get()
            if s is None: return
            t0 = time.perf_counter()
            try:
                X, P = process_window(s["frames"]); probs = pred.predict(X, P)
                results.put(dict(probs=probs, seg=s, ms=1000 * (time.perf_counter() - t0)))
            except Exception as e:
                print("[inference] failed:", repr(e)[:200])
    threading.Thread(target=worker, daemon=True).start()

    words, pending = deque(maxlen=a.max_words), []
    last_word, last_word_t, last_event_t = None, -1e9, time.perf_counter()
    top3, last_ms, fps, t_prev, frame_i, show_lm, paused = [], 0.0, 0.0, time.perf_counter(), 0, not a.no_landmarks, False
    print("running. keys: q quit | c clear | t toggle speech | l toggle landmarks | r reset segmenter | space pause")

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                s = seg.flush()
                if s is not None: jobs.put(s)
                break
            t_now = frame_i / vid_fps if a.video else time.perf_counter()
            frame_i += 1
            inp = cv2.flip(frame, 1) if a.flip_input else frame
            res = holistic.process(cv2.cvtColor(inp, cv2.COLOR_BGR2RGB))
            s = seg.update(t_now, holistic_to_frame(res))
            if s is not None:
                try: jobs.put_nowait(s)
                except queue.Full: print("[skip] inference is behind, dropped one sign")
            view = inp.copy()
            if show_lm:
                for lms, con in ((res.left_hand_landmarks, mph.HAND_CONNECTIONS), (res.right_hand_landmarks, mph.HAND_CONNECTIONS), (res.pose_landmarks, mph.POSE_CONNECTIONS)):
                    if lms is not None: mpd.draw_landmarks(view, lms, con)
            now = time.perf_counter(); inst = 1 / max(now - t_prev, 1e-6); fps = 0.9 * fps + 0.1 * inst if fps else inst; t_prev = now
        view_disp = cv2.flip(view, 1) if not a.flip_input and not a.video else view      # selfie-style preview for the webcam

        while not results.empty():                                           # recognised signs coming back from the worker
            r = results.get(); last_ms = r["ms"]
            accepted, word, top3 = decide(r["probs"], pred.signs, a.conf, a.margin); word = pretty(word)
            tnow = time.perf_counter(); last_event_t = tnow
            print(f"{'ACCEPT' if accepted else 'reject'} {word:<14s} {top3[0][1]:.2f} | top-3 {[(w, round(p, 2)) for w, p in top3]} | {r['seg']['duration']:.2f}s sign, {last_ms:.0f} ms")
            if accepted and not (word == last_word and tnow - last_word_t < a.repeat_s):
                words.append(word); pending.append(word); last_word, last_word_t = word, tnow
                if a.speak == "word": speaker.say(word)
        if a.speak == "sentence" and pending and time.perf_counter() - last_event_t > 1.5 and seg.state == "idle":
            speaker.say(" ".join(pending)); pending = []
        if words and time.perf_counter() - last_event_t > a.clear_after and seg.state == "idle": words.clear()

        cand = "..." if seg.state == "signing" else None                      # shown in grey while a sign is being captured
        out = draw_subtitles(view_disp.copy(), list(words), cand)
        hud = [f"{fps:4.1f} FPS | {seg.state} | speech {'on' if speaker.enabled else 'off'} | model {last_ms:.0f} ms"]
        hud += [f"{w}: {p:.2f}" for w, p in top3]
        out = draw_hud(out, hud)
        if a.save_video:
            if writer is None: writer = cv2.VideoWriter(a.save_video, cv2.VideoWriter_fourcc(*"mp4v"), 25, (out.shape[1], out.shape[0]))
            writer.write(out)
        cv2.imshow("Sign recognition (q to quit)", out)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27): break
        elif k == ord("c"): words.clear(); pending.clear()
        elif k == ord("t"): speaker.enabled = not speaker.enabled
        elif k == ord("l"): show_lm = not show_lm
        elif k == ord("r"): seg.reset()
        elif k == ord(" "): paused = not paused

    jobs.put(None); cap.release(); holistic.close()
    if writer is not None: writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
