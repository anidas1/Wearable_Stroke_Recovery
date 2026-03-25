"""
EMG Recording Script — REST / REACH
3-channel MyoWare 2.0 via ESP32 serial stream (sampler_features.ino)

Protocol per trial:
  5s SETTLE(discarded) → 1s REST(0) → 1s TRANSITION(-1) → 1.5s REACH(1) → 1s BREAK
  Trial duration: 4.5s + 1s break

Labels saved:
  0  = REST
  1  = REACH
 -1  = TRANSITION — discarded before saving

Output files:
  {name}{trial}trial_raw.csv   — raw ADC samples with labels
  {name}{trial}trial_feat.csv  — ESP32-extracted features + labels
"""

import serial
import serial.tools.list_ports
import csv
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BAUD_RATE           = 921600
NUM_TRIALS          = 50

INITIAL_SETTLE      = 5.0
REST_DURATION       = 1.0
TRANSITION_DURATION = 1.0
REACH_DURATION      = 1.5
INTER_TRIAL_BREAK   = 1.0

LABEL_REST          =  0
LABEL_REACH         =  1
LABEL_TRANSITION    = -1

NUM_FEATURES        = 24

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def select_port():
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found.")
        sys.exit(1)
    if len(ports) == 1:
        print(f"\nAuto-selecting: {ports[0].device}")
        return ports[0].device
    print("\nAvailable ports:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device} — {p.description}")
    return ports[int(input("\nSelect port number: "))].device


def collect_samples_with_progress(ser, duration_s, label,
                                   raw_buffer, feat_buffer,
                                   bar_label, color_code="\033[94m"):
    reset      = "\033[0m"
    start_time = time.time()
    end_time   = start_time + duration_s
    last_draw  = 0.0

    while True:
        now = time.time()
        if now >= end_time:
            break

        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("R:"):
                parts = line[2:].split(",")
                if len(parts) == 4:
                    raw_buffer.append([
                        int(parts[0]),
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                        label
                    ])

            elif line.startswith("F:"):
                parts = line[2:].split(",")
                if len(parts) == NUM_FEATURES + 1:
                    feats = [float(p) for p in parts[:NUM_FEATURES]]
                    feat_buffer.append(feats + [label])

        except (ValueError, UnicodeDecodeError, IndexError):
            pass

        now = time.time()
        if (now - last_draw) >= 0.05:
            remaining = max(0.0, end_time - now)
            frac      = min(1.0, max(0.0, (now - start_time) / duration_s))
            filled    = int(frac * 20)
            bar       = "█" * filled + "░" * (20 - filled)
            sys.stdout.write(
                f"\r{color_code}  [{bar_label:<12}]  {reset}"
                f"|{bar}|  {remaining:.1f}s remaining   "
            )
            sys.stdout.flush()
            last_draw = now

    sys.stdout.write("\r" + " " * 70 + "\r")
    sys.stdout.flush()


def break_with_progress(duration_s):
    start_time = time.time()
    end_time   = start_time + duration_s
    last_draw  = 0.0
    while time.time() < end_time:
        now = time.time()
        if (now - last_draw) >= 0.05:
            remaining = max(0.0, end_time - now)
            frac      = min(1.0, (now - start_time) / duration_s)
            filled    = int(frac * 20)
            bar       = "█" * filled + "░" * (20 - filled)
            sys.stdout.write(
                f"\r\033[90m  [BREAK]      \033[0m"
                f"|{bar}|  {remaining:.1f}s   "
            )
            sys.stdout.flush()
            last_draw = now
        time.sleep(0.01)
    sys.stdout.write("\r" + " " * 70 + "\r")
    sys.stdout.flush()


def halfway_break(ser, raw_buffer, feat_buffer):
    print("\n  ⏸  HALFWAY — rest your arm\n")
    ser.reset_input_buffer()
    input("  Press ENTER when ready to continue...\n")
    drain_end = time.time() + 0.5
    while time.time() < drain_end:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("R:"):
                parts = line[2:].split(",")
                if len(parts) == 4:
                    raw_buffer.append([int(parts[0]),
                        float(parts[1]), float(parts[2]),
                        float(parts[3]), LABEL_TRANSITION])
            elif line.startswith("F:"):
                parts = line[2:].split(",")
                if len(parts) == NUM_FEATURES + 1:
                    feats = [float(p) for p in parts[:NUM_FEATURES]]
                    feat_buffer.append(feats + [LABEL_TRANSITION])
        except (ValueError, UnicodeDecodeError, IndexError):
            pass
    ser.reset_input_buffer()
    print("  ✓ Resuming\n")


def save_raw_csv(raw_buffer, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_us", "raw1", "raw2", "raw3", "label"])
        for row in raw_buffer:
            writer.writerow(row)
    print(f"✓ Raw saved:  {filename}  ({len(raw_buffer):,} samples)")


def save_feat_csv(feat_buffer, filename):
    clean     = [row for row in feat_buffer if row[-1] != LABEL_TRANSITION]
    discarded = len(feat_buffer) - len(clean)
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"f{i}" for i in range(NUM_FEATURES)] + ["label"])
        for row in clean:
            writer.writerow(row)
    print(f"✓ Feat saved: {filename}  "
          f"({len(clean)} windows, {discarded} transition discarded)")


# ─────────────────────────────────────────────
# SEGMENTATION PLOT
# ─────────────────────────────────────────────
def plot_segmentation(raw_buffer, subject_name, trial_num):
    if not raw_buffer:
        print("No data to plot.")
        return

    t_abs  = np.array([row[0] for row in raw_buffer], dtype=float)
    t      = (t_abs - t_abs[0]) / 1_000_000
    ch     = [np.array([row[i] for row in raw_buffer], dtype=float)
              for i in (1, 2, 3)]
    labels = np.array([int(row[4]) for row in raw_buffer])

    label_style = {
        LABEL_REST:       ('#d4edda', 'REST'),
        LABEL_REACH:      ('#f8d7da', 'REACH'),
        LABEL_TRANSITION: ('#fff3cd', 'TRANSITION'),
    }
    ch_colors = ['#e74c3c', '#2ecc71', '#3498db']
    ch_names  = ['CH1 Ant.Deltoid', 'CH2 Tricep', 'CH3 Forearm']

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f'EMG Segmentation — {subject_name} Trial {trial_num}',
                 fontsize=13, fontweight='bold')

    for ax, sig, name, col in zip(axes, ch, ch_names, ch_colors):
        added = set()
        i = 0
        while i < len(labels):
            lbl = labels[i]
            j   = i
            while j < len(labels) and labels[j] == lbl:
                j += 1
            bg, txt = label_style.get(lbl, ('#ffffff', '?'))
            ax.axvspan(t[i], t[min(j, len(t) - 1)],
                       color=bg, alpha=0.45,
                       label=txt if txt not in added else '_nolegend_')
            added.add(txt)
            ax.plot(t[i:j], sig[i:j], color=col, linewidth=0.7, alpha=0.9)
            i = j
        ax.set_ylabel(name, fontsize=10)
        ax.grid(True, alpha=0.25, linestyle='--')

    shown = set(labels)
    patches = [mpatches.Patch(color=label_style[k][0],
                               label=label_style[k][1], alpha=0.6)
               for k in label_style if k in shown]
    axes[0].legend(handles=patches, loc='upper right',
                   fontsize=8, framealpha=0.85)
    axes[-1].set_xlabel('Time (s)', fontsize=10)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("  EMG Recording — REST / REACH")
    print("="*55)

    subject_name = input("  Subject name  (e.g. javen): ").strip().lower()
    trial_num    = input("  Trial number  (e.g. 1):     ").strip()

    if not subject_name: subject_name = "subject"
    if not trial_num:    trial_num    = "1"

    base_name     = f"{subject_name}{trial_num}trial"
    raw_filename  = f"{base_name}_raw.csv"
    feat_filename = f"{base_name}_feat.csv"

    trial_duration = REST_DURATION + TRANSITION_DURATION + REACH_DURATION
    total_min      = (INITIAL_SETTLE +
                      NUM_TRIALS * (trial_duration + INTER_TRIAL_BREAK)) / 60

    print("\n" + "="*55)
    print(f"  Subject      : {subject_name}")
    print(f"  Trial number : {trial_num}")
    print(f"  Output files : {raw_filename}")
    print(f"               : {feat_filename}")
    print(f"  Trials       : {NUM_TRIALS}")
    print(f"\n  Per trial:")
    print(f"    {REST_DURATION}s REST → {TRANSITION_DURATION}s prep → "
          f"{REACH_DURATION}s REACH → {INTER_TRIAL_BREAK}s break")
    print(f"  Trial duration : {trial_duration:.1f}s")
    print(f"  Total session  : ~{total_min:.1f} min")
    print("="*55)

    port = select_port()

    print(f"\nConnecting to {port} at {BAUD_RATE} baud...")
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=0.01)
        time.sleep(2)
        ser.reset_input_buffer()
        print("✓ Connected\n")
    except serial.SerialException as e:
        print(f"✗ Could not open port: {e}")
        sys.exit(1)

    # Sniff
    print("  Sniffing stream (2s)...")
    sniff_end = time.time() + 2.0
    r_count, f_count, shown = 0, 0, 0
    while time.time() < sniff_end:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line:
            if shown < 6:
                print(f"    {repr(line)}")
                shown += 1
            if line.startswith("R:"): r_count += 1
            if line.startswith("F:"): f_count += 1
    print(f"  R: lines={r_count}  F: lines={f_count}")
    if r_count == 0 and f_count == 0:
        print("  WARNING: No data — check firmware and baud rate")
    ser.reset_input_buffer()
    print()

    raw_buffer  = []
    feat_buffer = []

    print("─"*55)
    print("  SETUP")
    print("  • Place target ~30cm in front at table height")
    print("  • Arm relaxed at side to start")
    print("")
    print("  INSTRUCTIONS")
    print("  • REACH = move arm forward to target, same every trial")
    print("  • Keep motion fluid and consistent")
    print("─"*55)
    input("\n  Press ENTER when ready...\n")

    for i in range(3, 0, -1):
        print(f"  Starting in {i}...")
        time.sleep(1)
    ser.reset_input_buffer()
    print()

    # Initial settle
    print("  ▸ Hold still — sensors settling (5s)...")
    _r, _f = [], []
    collect_samples_with_progress(
        ser, INITIAL_SETTLE, LABEL_TRANSITION,
        _r, _f, "SETTLE", "\033[90m")
    ser.reset_input_buffer()
    print()

    for trial in range(1, NUM_TRIALS + 1):
        progress = ("█" * (trial * 20 // NUM_TRIALS) +
                    "░" * (20 - trial * 20 // NUM_TRIALS))
        print(f"\n{'─'*55}")
        print(f"  TRIAL {trial}/{NUM_TRIALS}  [{progress}]  "
              f"{trial * 100 // NUM_TRIALS}%")
        print(f"{'─'*55}")

        # REST
        print("  ▸ RELAX — arm still at side")
        collect_samples_with_progress(
            ser, REST_DURATION, LABEL_REST,
            raw_buffer, feat_buffer, "REST", "\033[92m")

        # TRANSITION
        print("  ▸ Get ready...")
        collect_samples_with_progress(
            ser, TRANSITION_DURATION, LABEL_TRANSITION,
            raw_buffer, feat_buffer, "PREPARE", "\033[93m")

        # REACH
        print("  ▸ REACH — move forward to target!")
        collect_samples_with_progress(
            ser, REACH_DURATION, LABEL_REACH,
            raw_buffer, feat_buffer, "REACH", "\033[91m")

        # BREAK
        if trial < NUM_TRIALS:
            break_with_progress(INTER_TRIAL_BREAK)

        # Auto-save every 5 trials
        if trial % 5 == 0:
            save_raw_csv(raw_buffer,   raw_filename)
            save_feat_csv(feat_buffer, feat_filename)
            print(f"  💾 Auto-saved at trial {trial}")

        # Halfway break
        if trial == NUM_TRIALS // 2:
            halfway_break(ser, raw_buffer, feat_buffer)

    ser.close()

    save_raw_csv(raw_buffer,   raw_filename)
    save_feat_csv(feat_buffer, feat_filename)

    print("\n  Plotting segmentation...")
    plot_segmentation(raw_buffer, subject_name, trial_num)

    feat_clean = [r for r in feat_buffer if r[-1] != LABEL_TRANSITION]
    rest_n     = sum(1 for r in feat_clean if r[-1] == LABEL_REST)
    reach_n    = sum(1 for r in feat_clean if r[-1] == LABEL_REACH)

    print("\n" + "="*55)
    print("  DONE")
    print("="*55)
    print(f"  Subject        : {subject_name}")
    print(f"  Trial          : {trial_num}")
    print(f"  Raw samples    : {len(raw_buffer):,}")
    print(f"  Feature windows: {len(feat_clean)}")
    print(f"    REST  (0)    : {rest_n}")
    print(f"    REACH (1)    : {reach_n}")
    print(f"  Raw file       : {raw_filename}")
    print(f"  Feat file      : {feat_filename}")

if __name__ == "__main__":
    main()
