"""
live_annotate_web.py — Browser UI for live EMG + labels + LDA QC.

Run:  python live_annotate_web.py
Open: http://127.0.0.1:5000

Requires: flask, flask-socketio, python-socketio, pyserial, joblib, numpy, scikit-learn
  (see requirements.txt)
"""

from __future__ import annotations

import copy
import csv
import threading
import time
from collections import deque
from pathlib import Path

import joblib
import numpy as np
import serial
import serial.tools.list_ports
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BAUD_RATE       = 921600
NUM_FEATURES    = 24
DEFAULT_LABEL   = "unlabeled"
RING_MAXLEN     = 24000   # ~12 s at 2 kHz
DISPLAY_STRIDE  = 4       # plot every 4th sample → ~500 Hz effective
BROADCAST_HZ    = 20      # emit interval (seconds = 1/BROADCAST_HZ)
HOST            = "127.0.0.1"
PORT            = 5000

PKL_DIR = Path(__file__).parent

PALETTE = [
    "#d4edda", "#f8d7da", "#fff3cd", "#cce5ff",
    "#f5c6cb", "#d1ecf1", "#ffeeba", "#e2e3e5",
]

try:
    _lda    = joblib.load(PKL_DIR / "lda_model.pkl")
    _scaler = joblib.load(PKL_DIR / "scaler.pkl")
    LDA_READY = True
except Exception:
    _lda = _scaler = None
    LDA_READY = False

LDA_THRESHOLD       = 0.5
LDA_CONFIRM_WINDOWS = 10
_reach_streak       = 0


def lda_predict(feat_row: list) -> int:
    global _reach_streak
    if not LDA_READY or _lda is None or _scaler is None:
        return -1
    lda, scaler = _lda, _scaler
    x = np.array(feat_row, dtype=np.float64).reshape(1, -1)
    score = float(lda.decision_function(scaler.transform(x))[0])
    if score > LDA_THRESHOLD:
        _reach_streak += 1
    else:
        _reach_streak = 0
    return 1 if _reach_streak >= LDA_CONFIRM_WINDOWS else 0


# ─────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────
_lock = threading.Lock()

_current_label: str = DEFAULT_LABEL
_label_colors: dict[str, str] = {}
_raw_buf: list = []
_feat_buf: list = []
_held_pred: int = -1

_running_serial = False
_recording = False

_ring: deque = deque(maxlen=RING_MAXLEN)  # (t_s, r1, r2, r3, lda, label)
_segments: list[dict] = []                 # {t0, t1, label, color}
_first_ts_us: int | None = None
_last_t_s: float = 0.0

_ser: serial.Serial | None = None
_serial_thread: threading.Thread | None = None
_bcast_stop = threading.Event()
_bcast_thread: threading.Thread | None = None

_session_name = "session_web"


def _join_stream_threads() -> None:
    global _serial_thread, _bcast_thread
    _bcast_stop.set()
    if _serial_thread is not None and _serial_thread.is_alive():
        _serial_thread.join(timeout=2.5)
    if _bcast_thread is not None and _bcast_thread.is_alive():
        _bcast_thread.join(timeout=2.5)
    _serial_thread = None
    _bcast_thread = None


def _color_for_label(lbl: str) -> str:
    if lbl not in _label_colors:
        _label_colors[lbl] = PALETTE[len(_label_colors) % len(PALETTE)]
    return _label_colors[lbl]


def _ensure_open_segment() -> None:
    if not _segments:
        c = _color_for_label(_current_label)
        _segments.append({"t0": 0.0, "t1": 0.0, "label": _current_label, "color": c})


def reset_session_state() -> None:
    global _reach_streak
    global _first_ts_us, _held_pred, _segments, _label_colors
    global _raw_buf, _feat_buf, _ring, _last_t_s, _current_label
    with _lock:
        _reach_streak = 0
        _first_ts_us = None
        _held_pred = -1
        _raw_buf.clear()
        _feat_buf.clear()
        _ring.clear()
        _last_t_s = 0.0
        _current_label = DEFAULT_LABEL
        _label_colors = {}
        _segments = []
        _ensure_open_segment()


def get_label() -> str:
    with _lock:
        return _current_label


def set_label_web(new: str) -> None:
    global _current_label
    with _lock:
        t_cut = _last_t_s
        if _segments:
            _segments[-1]["t1"] = t_cut
        _current_label = new
        c = _color_for_label(new)
        _segments.append({"t0": t_cut, "t1": t_cut, "label": new, "color": c})


def save_csvs(base: str) -> tuple[str, str]:
    raw_path  = f"{base}_annotated_raw.csv"
    feat_path = f"{base}_annotated_feat.csv"
    with _lock:
        raw_snap  = list(_raw_buf)
        feat_snap = list(_feat_buf)
    with open(raw_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_us", "raw1", "raw2", "raw3", "lda_pred", "label"])
        w.writerows(raw_snap)
    with open(feat_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"f{i}" for i in range(NUM_FEATURES)] + ["lda_pred", "label"])
        w.writerows(feat_snap)
    return raw_path, feat_path


def serial_reader() -> None:
    global _running_serial, _first_ts_us, _held_pred, _last_t_s
    assert _ser is not None
    while _running_serial:
        try:
            line = _ser.readline().decode("utf-8", errors="ignore").strip()
        except Exception:
            break
        if not line or line.startswith("#"):
            continue

        if line.startswith("R:"):
            parts = line[2:].split(",")
            if len(parts) != 4:
                continue
            try:
                ts_us = int(parts[0])
                r1, r2, r3 = float(parts[1]), float(parts[2]), float(parts[3])
            except (ValueError, IndexError):
                continue
            with _lock:
                if _first_ts_us is None:
                    _first_ts_us = ts_us
                t_s = (ts_us - _first_ts_us) / 1e6
                _last_t_s = t_s
                pred = _held_pred
                lbl = _current_label
                _ring.append((t_s, r1, r2, r3, pred, lbl))
                if _recording:
                    _raw_buf.append([ts_us, r1, r2, r3, pred, lbl])
                if _segments:
                    _segments[-1]["t1"] = t_s

        elif line.startswith("F:"):
            parts = line[2:].split(",")
            if len(parts) != NUM_FEATURES:
                continue
            try:
                feats = [float(p) for p in parts]
            except ValueError:
                continue
            pred = lda_predict(feats)
            with _lock:
                _held_pred = pred
                if _recording:
                    _feat_buf.append(feats + [pred, _current_label])


def broadcaster_loop(socketio: SocketIO, app: Flask) -> None:
    interval = 1.0 / BROADCAST_HZ
    while not _bcast_stop.is_set():
        _bcast_stop.wait(interval)
        if _bcast_stop.is_set():
            break
        with _lock:
            if not _ring:
                continue
            data = list(_ring)
            t_all = [row[0] for row in data]
            # downsample for browser
            if len(data) > 1:
                idx = range(0, len(data), DISPLAY_STRIDE)
                data_ds = [data[i] for i in idx]
            else:
                data_ds = data
            t_ds = [r[0] for r in data_ds]
            ch1 = [r[1] for r in data_ds]
            ch2 = [r[2] for r in data_ds]
            ch3 = [r[3] for r in data_ds]
            lda = [r[4] for r in data_ds]
            t_max = t_all[-1]
            segs = copy.deepcopy(_segments)
            if segs:
                segs[-1]["t1"] = t_max
            n_raw = len(_raw_buf)
            n_feat = len(_feat_buf)
            lbl = _current_label
            pred = _held_pred
            rec = _recording

        payload = {
            "t": t_ds,
            "ch1": ch1,
            "ch2": ch2,
            "ch3": ch3,
            "lda": lda,
            "t_max": t_max,
            "segments": segs,
            "status": {
                "recording": rec,
                "n_raw": n_raw,
                "n_feat": n_feat,
                "label": lbl,
                "lda_pred": pred,
                "lda_ready": LDA_READY,
            },
        }
        with app.app_context():
            socketio.emit("emg_batch", payload, namespace="/")


# ─────────────────────────────────────────────
# FLASK + SOCKETIO
# ─────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = "emg-live-dev"
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")


@socketio.on("connect")
def on_socket_connect():
    """New browser tab/page: align UI with actual serial state (survives refresh)."""
    ser_open = _ser is not None and _ser.is_open
    sid = getattr(request, "sid", None)
    if sid is not None:
        socketio.emit("serial_state", {"open": ser_open}, to=sid, namespace="/")


@app.route("/")
def index():
    return render_template("live_emg.html")


@app.route("/api/ports")
def api_ports():
    ports = serial.tools.list_ports.comports()
    return jsonify([{"device": p.device, "description": p.description or ""}
                    for p in ports])


@socketio.on("connect_serial")
def on_connect_serial(data):
    global _ser, _running_serial, _serial_thread, _bcast_thread, _bcast_stop
    port = (data or {}).get("port")
    if not port:
        socketio.emit("log", {"msg": "No port specified"})
        return
    if _ser is not None and _ser.is_open:
        # Refresh / second tab: port is already open — sync UI instead of refusing.
        socketio.emit(
            "log",
            {"msg": "Serial already open — using existing connection (e.g. after refresh)."},
        )
        socketio.emit("serial_state", {"open": True}, namespace="/")
        return

    _join_stream_threads()
    _bcast_stop.clear()

    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=0.01)
        time.sleep(2)
        ser.reset_input_buffer()
        ser.write(b"r")
        ser.flush()
        time.sleep(0.1)
        ser.reset_input_buffer()
    except serial.SerialException as e:
        socketio.emit("log", {"msg": f"Serial error: {e}"})
        return

    reset_session_state()
    with _lock:
        _ensure_open_segment()

    _ser = ser
    _running_serial = True
    _serial_thread = threading.Thread(target=serial_reader, daemon=True)
    _serial_thread.start()

    _bcast_thread = threading.Thread(
        target=broadcaster_loop, args=(socketio, app), daemon=True
    )
    _bcast_thread.start()
    socketio.emit("log", {"msg": f"Connected to {port}"})
    socketio.emit("serial_state", {"open": True})


@socketio.on("disconnect_serial")
def on_disconnect_serial():
    global _ser, _running_serial
    _running_serial = False
    if _ser is not None:
        try:
            _ser.close()
        except Exception:
            pass
        _ser = None
    _join_stream_threads()
    _bcast_stop.clear()
    socketio.emit("log", {"msg": "Serial closed"})
    socketio.emit("serial_state", {"open": False})


@socketio.on("set_label")
def on_set_label(data):
    lbl = (data or {}).get("label", "").strip()
    if not lbl:
        return
    set_label_web(lbl)
    socketio.emit("log", {"msg": f"Label → {lbl}"})


@socketio.on("start_recording")
def on_start():
    global _recording
    with _lock:
        _recording = True
    socketio.emit("log", {"msg": "Recording ON"})


@socketio.on("stop_recording")
def on_stop():
    global _recording
    with _lock:
        _recording = False
        n = len(_raw_buf)
    socketio.emit("log", {"msg": f"Recording OFF ({n:,} raw rows buffered)"})


@socketio.on("save_csv")
def on_save(data):
    global _session_name
    name = (data or {}).get("session", "").strip() or _session_name
    _session_name = name
    try:
        raw_p, feat_p = save_csvs(name)
        socketio.emit("log", {"msg": f"Saved {raw_p} + {feat_p}"})
    except Exception as e:
        socketio.emit("log", {"msg": f"Save failed: {e}"})


def main() -> None:
    print(f"LDA ready: {LDA_READY}")
    print(f"Open http://{HOST}:{PORT}")
    socketio.run(app, host=HOST, port=PORT, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
