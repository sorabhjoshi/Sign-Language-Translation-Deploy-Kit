"""Landmark side of the real-time sign recognizer: MediaPipe result -> 89-point layout, window processing, sign segmentation.

Numpy only. The layout constants, `normalize_body` and `fill_hands` are copied verbatim from the training notebook,
and `process_window` repeats the steps of the notebook's `process` on a live window instead of a parquet file,
so the model sees the same kind of input it was trained on.
"""
import warnings
from collections import deque
import numpy as np

# ---- copied verbatim from the training notebook -------------------------------------------------------------
LH0, POSE0, RH0 = 468, 489, 522
POSE_KEEP = [0, 11, 12, 13, 14, 15, 16]
LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 95, 88, 178, 87, 14, 317, 402, 318, 324]   # 40 lip points; [0]/[10] = mouth corners
N_PTS = 49 + len(LIPS)   # layout: 0-41 hands, 42-48 pose, 49-88 lips

def normalize_body(kp):
    """Body frame: origin = mid-shoulder, unit = shoulder width."""
    ls, rs = kp[:, 43], kp[:, 44]
    mid = np.nanmean((ls + rs) / 2, axis=0)
    width = np.nanmean(np.linalg.norm((ls - rs)[:, :2], axis=-1))
    if not (np.isfinite(mid).all() and np.isfinite(width)) or width < 1e-3:
        mid = np.nanmean(kp[:, :42].reshape(-1, 3), axis=0)
        width = 0.25
        if not np.isfinite(mid).all():
            mid = np.zeros(3, np.float32)
    return (kp - mid) / width

def fill_hands(kpn, pres):
    """Linearly interpolate dropped hand detections between the first and last detected frame."""
    kpn, pres = kpn.copy(), pres.copy()
    for h in range(2):
        ok = np.where(pres[:, h])[0]
        if len(ok) < 2: continue
        t = np.arange(ok[0], ok[-1] + 1)
        for j in range(h * 21, h * 21 + 21):
            for d in range(3): kpn[t, j, d] = np.interp(t, ok, kpn[ok, j, d])
        pres[t, h] = True
    return kpn, pres

# ---- end of verbatim part ------------------------------------------------------------------------------------


def holistic_to_frame(res):
    """MediaPipe Holistic result -> (89, 3) float32 in the training layout. NaN = landmark not detected.
    Layout: 0-20 left hand | 21-41 right hand | 42-48 pose (nose, shoulders, elbows, wrists) | 49-88 lips."""
    out = np.full((N_PTS, 3), np.nan, np.float32)
    def lm(l): return None if l is None else np.array([[p.x, p.y, p.z] for p in l.landmark], np.float32)
    lh, rh, po, fa = lm(res.left_hand_landmarks), lm(res.right_hand_landmarks), lm(res.pose_landmarks), lm(res.face_landmarks)
    if lh is not None: out[0:21] = lh
    if rh is not None: out[21:42] = rh
    if po is not None: out[42:49] = po[POSE_KEEP]
    if fa is not None: out[49:] = fa[LIPS]
    return out


def process_window(kp, T=64):
    """kp: (n, 89, 3) raw landmarks of one sign (NaN = missing)  ->  X (T, 89, 3) float16, P (T, 2) bool.
    Same steps as `process` in the training notebook: body frame, hand-gap interpolation, crop to the hand-present
    segment, resample to T frames."""
    n = len(kp)
    if n == 0:
        return np.zeros((T, N_PTS, 3), np.float16), np.zeros((T, 2), bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pres = (~np.isnan(kp[:, :42, 0])).reshape(n, 2, 21).any(-1)
        kpn = normalize_body(kp)
        kpn, pres = fill_hands(kpn, pres)
    act = np.where(pres.any(1))[0]                       # crop to the hand-present segment
    lo, hi = (max(act[0] - 2, 0), min(act[-1] + 2, n - 1)) if len(act) else (0, n - 1)
    idx = np.linspace(lo, hi, T).round().astype(int)
    c = np.clip(np.nan_to_num(kpn[idx], nan=0.0), -8, 8)
    return c.astype(np.float16), pres[idx]


class SignSegmenter:
    """Cuts the live landmark stream into isolated signs, the unit the model was trained on.

    A sign starts when a hand has been raised into signing space for `start_s` seconds and ends when both hands have
    been down or out of view for `end_gap_s`. "Raised" means the wrist is less than `rest_y` shoulder widths below the
    shoulder line; set rest_y very large to rely on hand detection only. Times come from the caller, so the same code
    works for a webcam (wall clock) and for a video file (frame index / fps).
    """
    def __init__(self, rest_y=1.4, start_s=0.10, end_gap_s=0.35, min_sign_s=0.30, max_sign_s=4.0, preroll_s=0.15):
        self.rest_y, self.start_s, self.end_gap_s = rest_y, start_s, end_gap_s
        self.min_sign_s, self.max_sign_s, self.preroll_s = min_sign_s, max_sign_s, preroll_s
        self.reset()

    def reset(self):
        self.state, self.buf, self.pre = "idle", [], deque()
        self.active_since = self.last_active = self.t_start = None

    def hand_up(self, pts):
        ls, rs = pts[43], pts[44]
        have_sh = np.isfinite(ls[:2]).all() and np.isfinite(rs[:2]).all()
        for h0 in (0, 21):
            wrist = pts[h0]
            if not np.isfinite(wrist[:2]).all(): continue
            if not have_sh: return True
            width = max(float(np.linalg.norm(ls[:2] - rs[:2])), 1e-3)
            if (wrist[1] - (ls[1] + rs[1]) / 2) / width < self.rest_y: return True
        return False

    def update(self, t, pts):
        """Feed one frame. Returns None, or a dict(frames, t0, t1, duration) when a sign has just ended."""
        act = self.hand_up(pts)
        if self.state == "idle":
            self.pre.append((t, pts))
            while self.pre and t - self.pre[0][0] > self.preroll_s: self.pre.popleft()
            if act:
                if self.active_since is None: self.active_since = t
                if t - self.active_since >= self.start_s:
                    self.state, self.buf = "signing", list(self.pre)
                    self.t_start, self.last_active = self.buf[0][0], t
            else:
                self.active_since = None
            return None
        self.buf.append((t, pts))
        if act: self.last_active = t
        if t - self.last_active >= self.end_gap_s or t - self.t_start >= self.max_sign_s:
            return self._finish()
        return None

    def flush(self):
        return self._finish() if self.state == "signing" else None

    def _finish(self):
        keep = [(t, p) for t, p in self.buf if t <= self.last_active + 0.15]      # drop the frames after the hands went down
        t0, t1, dur = self.t_start, self.last_active, self.last_active - self.t_start
        self.reset()
        if dur < self.min_sign_s or len(keep) < 5: return None
        return dict(frames=np.stack([p for _, p in keep]), t0=t0, t1=t1, duration=dur)
