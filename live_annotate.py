"""
live_annotate.py
Reads ESP32 CSV stream, lets you type labels in real time
Saves two CSVs:
  {name}{session}_feat.csv  — features + lda_pred + user_label
  {name}{session}_log.csv   — timestamp + lda_pred + user_label only
"""

import serial
import serial.tools.list_ports
import csv
import time
import sys
import threading
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────
BAUD_RATE    = 921600
NUM_FEATURES = 24

# ─────────────────────────────────────────────────────────────────────
# SHARED STATE
# ─────────────────────────────────────────────────────────────────────
feat_buffer   = []
log_buffer    = []
current_label = "unlabeled"
running       = True
lock          = threading.Lock()

# ─────────────────────────────────────────────────────────────────────
# PORT SELECT
# ─────────────────────────────────────────────────────────────────────
def select_port():
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No ports found"); sys.exit(1)
    if len(ports) == 1:
        print(f"Auto-selecting: {ports[0].device}")
        return ports[0].device
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device} — {p.description}")
    return ports[int(input("Select port: "))].device

# ─────────────────────────────────────────────────────────────────────
# ANNOTATION THREAD — type labels while recording
# ─────────────────────────────────────────────────────────────────────
def annotation_thread():
    global current_label, running
    print("\n  ── Label controls ──────────────────────────")
    print("  Type any label + Enter to set it")
    print("  Common: 'rest'  'reach'  'jerk'  'noise'")
    print("  Type 'done' to stop recording")
    print("  ────────────────────────────────────────────\n")
    while running:
        try:
            text = input().strip()
            if not text:
                continue
            if text.lower() == "done":
                running = False
                break
            with lock:
                current_label = text.lower()
            print(f"  → [{datetime.now().strftime('%H:%M:%S')}] Label: '{current_label}'")
        except EOFError:
            break

# ─────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────
def save_feat_csv(filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["timestamp_us"] +
            [f"f{i}" for i in range(NUM_FEATURES)] +
            ["lda_pred", "user_label"]
        )
        for row in feat_buffer:
            writer.writerow(row)
    print(f"✓ Feat saved : {filename}  ({len(feat_buffer)} windows)")

def save_log_csv(filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_us", "lda_pred", "user_label"])
        for row in log_buffer:
            writer.writerow(row)
    print(f"✓ Log  saved : {filename}  ({len(log_buffer)} entries)")

# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    global running

    print("\n" + "="*55)
    print("  Live EMG Annotator")
    print("="*55)

    subject  = input("  Subject name:   ").strip().lower() or "subject"
    session  = input("  Session number: ").strip() or "1"

    feat_filename = f"{subject}{session}_feat.csv"
    log_filename  = f"{subject}{session}_log.csv"

    port = select_port()

    print(f"\nConnecting to {port} at {BAUD_RATE}...")
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
        time.sleep(2)
        ser.reset_input_buffer()
        print("✓ Connected\n")
    except serial.SerialException as e:
        print(f"✗ {e}"); sys.exit(1)

    # Start annotation input thread
    ann_thread = threading.Thread(target=annotation_thread, daemon=True)
    ann_thread.start()

    print(f"  {'TIME':>8}  {'LDA':>6}  {'LABEL':<15}  WINDOW")
    print("  " + "─"*50)

    header_skipped = False
    window_count   = 0

    try:
        while running:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            # Skip header line
            if not header_skipped:
                if line.startswith("timestamp"):
                    header_skipped = True
                continue

            parts = line.split(",")

            # Expect: timestamp, f0..f23, prediction = 26 fields
            if len(parts) != NUM_FEATURES + 2:
                continue

            try:
                ts   = int(parts[0])
                feat = [float(p) for p in parts[1:NUM_FEATURES+1]]
                pred = parts[NUM_FEATURES + 1].strip()
            except (ValueError, IndexError):
                continue

            with lock:
                label = current_label

            # Save to buffers
            feat_buffer.append([ts] + feat + [pred, label])
            log_buffer.append([ts, pred, label])
            window_count += 1

            # Auto-save every 100 windows
            if window_count % 100 == 0:
                save_feat_csv(feat_filename)
                save_log_csv(log_filename)
                print(f"  💾 Auto-saved ({window_count} windows)")

            # Live display
            t_str = datetime.now().strftime("%H:%M:%S")
            print(f"  {t_str}  {pred:>6}  {label:<15}  #{window_count}")

    except KeyboardInterrupt:
        print("\n\nCtrl+C — stopping...")
    finally:
        running = False
        ser.close()
        save_feat_csv(feat_filename)
        save_log_csv(log_filename)

    print("\n" + "="*55)
    print("  DONE")
    print("="*55)
    print(f"  Windows recorded : {window_count}")
    print(f"  Labels used      : {sorted(set(r[-1] for r in log_buffer))}")
    print(f"  Feat file        : {feat_filename}")
    print(f"  Log  file        : {log_filename}")

if __name__ == "__main__":
    main()
```

**Usage:**
1. Flash ESP32, open this script
2. Enter name and session number
3. Terminal shows live stream — type labels freely:
```
  TIME      LDA    LABEL            WINDOW
  ──────────────────────────────────────────
  14:23:01  REST   unlabeled        #1
  14:23:01  REST   unlabeled        #2
→ reach          ← you type this
  → Label: 'reach'
  14:23:02  REACH  reach            #3
  14:23:02  REACH  reach            #4
→ rest
  → Label: 'rest'
  14:23:03  REST   rest             #5
