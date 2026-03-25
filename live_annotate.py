"""
live_annotate.py
Reads ESP32 dual-stream (R: raw + F: features), lets you type labels in real time
Saves two CSVs:
  {name}{session}_feat.csv  — features + lda_pred + user_label
  {name}{session}_raw.csv   — raw ADC + user_label
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
raw_buffer    = []
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
# ANNOTATION THREAD
# ─────────────────────────────────────────────────────────────────────
def annotation_thread():
    global current_label, running
    print("\n  ── Label controls ──────────────────────────")
    print("  Type any label + Enter to set it instantly")
    print("  Common: 'rest'  'reach'  'jerk'  'noise'  'misc'")
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
            print(f"  → [{datetime.now().strftime('%H:%M:%S')}] "
                  f"Label set to: '{current_label}'")
        except EOFError:
            break

# ─────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────
def save_feat_csv(filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [f"f{i}" for i in range(NUM_FEATURES)] +
            ["lda_pred", "user_label"]
        )
        for row in feat_buffer:
            writer.writerow(row)
    print(f"✓ Feat saved : {filename}  ({len(feat_buffer)} windows)")

def save_raw_csv(filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_us", "raw1", "raw2", "raw3", "user_label"])
        for row in raw_buffer:
            writer.writerow(row)
    print(f"✓ Raw  saved : {filename}  ({len(raw_buffer):,} samples)")

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
    raw_filename  = f"{subject}{session}_raw.csv"

    port = select_port()

    print(f"\nConnecting to {port} at {BAUD_RATE}...")
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
        time.sleep(2)
        ser.reset_input_buffer()
        print("✓ Connected\n")
    except serial.SerialException as e:
        print(f"✗ {e}"); sys.exit(1)

    # Sniff to confirm stream
    print("  Sniffing stream (2s)...")
    sniff_end  = time.time() + 2.0
    r_count, f_count = 0, 0
    while time.time() < sniff_end:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line and not line.startswith("#"):
            if line.startswith("R:"): r_count += 1
            if line.startswith("F:"): f_count += 1
            if r_count + f_count <= 5:
                print(f"    {repr(line)}")
    print(f"  R: lines={r_count}  F: lines={f_count}")
    if r_count == 0 and f_count == 0:
        print("  WARNING: No data — check firmware and baud rate")
    ser.reset_input_buffer()

    # Start annotation thread
    ann_thread = threading.Thread(target=annotation_thread, daemon=True)
    ann_thread.start()

    print(f"\n  {'TIME':>8}  {'LDA':>6}  {'LABEL':<15}  {'WINDOWS':>7}  {'SAMPLES':>8}")
    print("  " + "─"*58)

    window_count = 0
    sample_count = 0

    try:
        while running:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line or line.startswith("#"):
                continue

            with lock:
                label = current_label

            # ── Raw sample ────────────────────────────────────────
            if line.startswith("R:"):
                parts = line[2:].split(",")
                if len(parts) == 4:
                    try:
                        raw_buffer.append([
                            int(parts[0]),
                            float(parts[1]),
                            float(parts[2]),
                            float(parts[3]),
                            label
                        ])
                        sample_count += 1
                    except ValueError:
                        pass

            # ── Feature window ────────────────────────────────────
            elif line.startswith("F:"):
                parts = line[2:].split(",")
                if len(parts) == NUM_FEATURES + 1:
                    try:
                        feats = [float(p) for p in parts[:NUM_FEATURES]]
                        pred  = parts[NUM_FEATURES].strip()
                        feat_buffer.append(feats + [pred, label])
                        window_count += 1

                        # Auto-save every 100 windows
                        if window_count % 100 == 0:
                            save_feat_csv(feat_filename)
                            save_raw_csv(raw_filename)
                            print(f"  💾 Auto-saved "
                                  f"({window_count} windows, "
                                  f"{sample_count:,} samples)")

                        # Live display — update on every feature window
                        t_str = datetime.now().strftime("%H:%M:%S")
                        print(f"  {t_str}  {pred:>6}  "
                              f"{label:<15}  "
                              f"{window_count:>7}  "
                              f"{sample_count:>8,}")

                    except ValueError:
                        pass

    except KeyboardInterrupt:
        print("\n\nCtrl+C — stopping...")
    finally:
        running = False
        ser.close()
        save_feat_csv(feat_filename)
        save_raw_csv(raw_filename)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  DONE")
    print("="*55)
    print(f"  Windows recorded : {window_count}")
    print(f"  Samples recorded : {sample_count:,}")
    labels_used = sorted(set(r[-1] for r in feat_buffer))
    print(f"  Labels used      : {labels_used}")
    print(f"  Feat file        : {feat_filename}")
    print(f"  Raw  file        : {raw_filename}")

if __name__ == "__main__":
    main()
