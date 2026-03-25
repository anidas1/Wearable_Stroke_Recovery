"""
live_annotate.py — Free-running EMG recorder with live terminal labelling.

ESP32 streams R: (2000 Hz) and F: (10 Hz) lines.
Type a label in the terminal at any time → all rows from that point use it.
LDA prediction is computed in Python from each F: window and held until the
next window; it appears as an extra column in the raw CSV alongside your label.

Output CSVs:
  {name}_annotated_raw.csv   — timestamp_us, raw1, raw2, raw3, lda_pred, label
  {name}_annotated_feat.csv  — f0..f23, lda_pred, label

Commands (type and press ENTER):
  start        begin recording rows to CSV buffers
  stop         pause recording (serial + LDA stay live)
  <anything>   set label for all future rows (works paused or recording)
  save         write current buffers to CSV without stopping
  q / quit     save CSVs, show plot, exit
"""

import csv
import sys
import threading
import time
from pathlib import Path

import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import serial
import serial.tools.list_ports

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BAUD_RATE    = 921600
NUM_FEATURES = 24          # 8 features × 3 channels
DEFAULT_LABEL = "unlabeled"

PKL_DIR = Path(__file__).parent   # same folder as this script

# ─────────────────────────────────────────────
# LOAD LDA MODEL (optional — graceful fallback)
# ─────────────────────────────────────────────
try:
    _lda    = joblib.load(PKL_DIR / "lda_model.pkl")
    _scaler = joblib.load(PKL_DIR / "scaler.pkl")
    LDA_READY = True
    print("  LDA model loaded.")
except Exception:
    LDA_READY = False
    print("  WARNING: lda_model.pkl / scaler.pkl not found — lda_pred will be -1")


# Debouncing only here — Arduino firmware still uses raw score > 0 for GPIO inference.
LDA_THRESHOLD        = 0.5   # raise above 0 to reduce false REACH triggers
LDA_CONFIRM_WINDOWS  = 10  # consecutive windows above threshold needed for REACH
_reach_streak        = 0     # counts consecutive above-threshold windows

def lda_predict(feat_row: list) -> int:
    """Return 1 (REACH) only after LDA_CONFIRM_WINDOWS consecutive high-score windows."""
    global _reach_streak
    if not LDA_READY:
        return -1
    x     = np.array(feat_row, dtype=np.float64).reshape(1, -1)
    score = float(_lda.decision_function(_scaler.transform(x))[0])
    if score > LDA_THRESHOLD:
        _reach_streak += 1
    else:
        _reach_streak = 0
    return 1 if _reach_streak >= LDA_CONFIRM_WINDOWS else 0


# ─────────────────────────────────────────────
# SHARED STATE (protected by lock where needed)
# ─────────────────────────────────────────────
_lock          = threading.Lock()
_current_label: str = DEFAULT_LABEL
_raw_buf:  list = []
_feat_buf: list = []
_held_pred: int = -1       # last LDA prediction — held between F: windows
_running    = True
_recording  = False        # gating flag — buffers only filled when True
_t_start    = time.time()


def get_label() -> str:
    with _lock:
        return _current_label


def set_label(new: str) -> str:
    global _current_label
    with _lock:
        old = _current_label
        _current_label = new
    return old


def get_held_pred() -> int:
    with _lock:
        return _held_pred


def set_held_pred(p: int) -> None:
    global _held_pred
    with _lock:
        _held_pred = p


def is_recording() -> bool:
    with _lock:
        return _recording


def set_recording(state: bool) -> None:
    global _recording
    with _lock:
        _recording = state


# ─────────────────────────────────────────────
# PORT SELECTION
# ─────────────────────────────────────────────
def select_port() -> str:
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found.")
        sys.exit(1)
    if len(ports) == 1:
        print(f"  Auto-selecting: {ports[0].device}")
        return ports[0].device
    print("\n  Available ports:")
    for i, p in enumerate(ports):
        print(f"    [{i}] {p.device} — {p.description}")
    return ports[int(input("\n  Select port number: "))].device


# ─────────────────────────────────────────────
# SERIAL THREAD
# ─────────────────────────────────────────────
def serial_thread(ser: serial.Serial) -> None:
    global _running
    last_status = time.time()

    while _running:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
        except Exception:
            break

        if not line or line.startswith("#"):
            continue

        if line.startswith("R:"):
            parts = line[2:].split(",")
            if len(parts) == 4 and is_recording():
                try:
                    row = [int(parts[0]),
                           float(parts[1]), float(parts[2]), float(parts[3]),
                           get_held_pred(),
                           get_label()]
                    with _lock:
                        _raw_buf.append(row)
                except (ValueError, IndexError):
                    pass

        elif line.startswith("F:"):
            parts = line[2:].split(",")
            if len(parts) == NUM_FEATURES:
                try:
                    feats = [float(p) for p in parts]
                    pred  = lda_predict(feats)
                    set_held_pred(pred)         # LDA always updates
                    if is_recording():
                        row = feats + [pred, get_label()]
                        with _lock:
                            _feat_buf.append(row)
                except (ValueError, IndexError):
                    pass

        # Print status every 10 seconds as a clean newline (no \r overwrite)
        now = time.time()
        if now - last_status >= 10.0:
            last_status = now
            with _lock:
                n_raw  = len(_raw_buf)
                n_feat = len(_feat_buf)
                lbl    = _current_label
                pred   = _held_pred
                rec    = _recording
            pred_str = {0: "REST", 1: "REACH"}.get(pred, "?")
            rec_str  = "RECORDING" if rec else "PAUSED"
            elapsed  = now - _t_start
            print(f"\n  [{rec_str}]  label={lbl}  pred={pred_str}  |  "
                  f"{n_raw:,} raw  {n_feat} feat  |  {elapsed:.0f}s")


# ─────────────────────────────────────────────
# CSV SAVE
# ─────────────────────────────────────────────
def save_csvs(base: str) -> None:
    raw_path  = f"{base}_annotated_raw.csv"
    feat_path = f"{base}_annotated_feat.csv"

    with _lock:
        raw_snap  = list(_raw_buf)
        feat_snap = list(_feat_buf)

    with open(raw_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_us", "raw1", "raw2", "raw3", "lda_pred", "label"])
        w.writerows(raw_snap)
    print(f"\n  Saved: {raw_path}  ({len(raw_snap):,} rows)")

    with open(feat_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"f{i}" for i in range(NUM_FEATURES)] + ["lda_pred", "label"])
        w.writerows(feat_snap)
    print(f"  Saved: {feat_path}  ({len(feat_snap)} rows)")


# ─────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────
def plot_session() -> None:
    with _lock:
        raw_snap = list(_raw_buf)

    if not raw_snap:
        print("  No data to plot.")
        return

    t_abs  = np.array([r[0] for r in raw_snap], dtype=float)
    t      = (t_abs - t_abs[0]) / 1e6
    ch     = [np.array([r[i] for r in raw_snap], dtype=float) for i in (1, 2, 3)]
    labels = [r[5] for r in raw_snap]   # string labels

    # Assign a consistent colour to each unique label
    unique_labels = list(dict.fromkeys(labels))   # preserve order
    palette = ["#d4edda", "#f8d7da", "#fff3cd", "#cce5ff",
               "#f5c6cb", "#d1ecf1", "#ffeeba", "#e2e3e5"]
    label_color = {lbl: palette[i % len(palette)]
                   for i, lbl in enumerate(unique_labels)}

    ch_colors = ["#e74c3c", "#2ecc71", "#3498db"]
    ch_names  = ["CH1 Ant.Deltoid", "CH2 Tricep", "CH3 Forearm"]
    lda_pred  = np.array([r[4] for r in raw_snap], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    fig.suptitle("Live Annotate — EMG + LDA QC", fontsize=13, fontweight="bold")

    for ax, sig, name, col in zip(axes[:3], ch, ch_names, ch_colors):
        added: set = set()
        i = 0
        while i < len(labels):
            j = i
            while j < len(labels) and labels[j] == labels[i]:
                j += 1
            lbl = labels[i]
            bg  = label_color.get(lbl, "#ffffff")
            ax.axvspan(t[i], t[min(j, len(t) - 1)], color=bg, alpha=0.45,
                       label=lbl if lbl not in added else "_nolegend_")
            ax.plot(t[i:j], sig[i:j], color=col, linewidth=0.6, alpha=0.9)
            added.add(lbl)
            i = j
        ax.set_ylabel(name, fontsize=10)
        ax.grid(True, alpha=0.25, linestyle="--")

    axes[0].legend(
        handles=[mpatches.Patch(color=label_color[l], label=l, alpha=0.7)
                 for l in unique_labels],
        loc="upper right", fontsize=8, framealpha=0.85
    )

    # Binary LDA prediction as a step plot in the 4th row
    ax_pred = axes[3]
    ax_pred.step(t, lda_pred, where="post", color="#8e44ad", linewidth=1.2)
    ax_pred.fill_between(t, lda_pred, step="post", alpha=0.25, color="#8e44ad")
    ax_pred.set_yticks([0, 1])
    ax_pred.set_yticklabels(["REST (0)", "REACH (1)"], fontsize=9)
    ax_pred.set_ylim(-0.1, 1.3)
    ax_pred.set_ylabel("LDA pred", fontsize=10)
    ax_pred.grid(True, alpha=0.25, linestyle="--")

    axes[-1].set_xlabel("Time (s)", fontsize=10)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main() -> None:
    global _running, _t_start

    print("\n" + "=" * 52)
    print("  Live EMG Annotator")
    print("=" * 52)
    name = input("  Session name (e.g. ani_test): ").strip() or "session"

    port = select_port()
    print(f"\n  Connecting to {port} at {BAUD_RATE} baud...")
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=0.01)
        time.sleep(2)
        ser.reset_input_buffer()
        ser.write(b"r")   # MODE_NORMAL → R: + F: stream
        ser.flush()
        time.sleep(0.1)
        ser.reset_input_buffer()
        print("  Connected.\n")
    except serial.SerialException as e:
        print(f"  Could not open port: {e}")
        sys.exit(1)

    _t_start = time.time()

    # Start serial reader thread
    t = threading.Thread(target=serial_thread, args=(ser,), daemon=True)
    t.start()

    print(f"  Current label : '{DEFAULT_LABEL}'")
    print("  Commands:")
    print("    start          — begin recording to CSV buffers")
    print("    stop           — pause recording (serial + LDA stay live)")
    print("    <label>        — set label for all future rows")
    print("    save           — write CSVs without stopping")
    print("    q / quit       — save, plot, exit\n")

    # Main thread: blocking input loop
    while True:
        try:
            cmd = input().strip()
        except (EOFError, KeyboardInterrupt):
            cmd = "q"

        if not cmd:
            continue

        low = cmd.lower()

        if low in ("q", "quit"):
            break

        if low == "start":
            if is_recording():
                print("\n  Already recording.")
            else:
                set_recording(True)
                with _lock:
                    t_now = time.time() - _t_start
                print(f"\n  Recording started  [{t_now:.1f}s]")
            continue

        if low == "stop":
            if not is_recording():
                print("\n  Already paused.")
            else:
                set_recording(False)
                with _lock:
                    n_raw = len(_raw_buf)
                    t_now = time.time() - _t_start
                print(f"\n  Recording paused  [{t_now:.1f}s, {n_raw:,} raw samples buffered]")
            continue

        if low == "save":
            save_csvs(name)
            continue

        # Anything else → new label
        old = set_label(cmd)
        with _lock:
            n_raw = len(_raw_buf)
            t_now = time.time() - _t_start
        rec_note = " (recording)" if is_recording() else " (paused — type 'start' to record)"
        print(f"\n  Label: '{old}' → '{cmd}'  [{t_now:.1f}s, {n_raw:,} rows]{rec_note}")

    _running = False
    ser.close()
    print("\n  Stopping...")
    t.join(timeout=2)

    save_csvs(name)
    print("\n  Plotting session...")
    plot_session()

    print("\n" + "=" * 52)
    print("  Done")
    with _lock:
        print(f"  Raw samples : {len(_raw_buf):,}")
        print(f"  Feat windows: {len(_feat_buf)}")
    print("=" * 52)


if __name__ == "__main__":
    main()
