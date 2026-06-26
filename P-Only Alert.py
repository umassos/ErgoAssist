import os
import time
import csv
import threading
import subprocess
from datetime import datetime

import mido

# =========================
# CONFIG (UPDATED)
# =========================
BASELINE_SEC = 60
TASK_SEC = 180

PITCH_THRESH_DEG = 10.0         # bad posture if |dev| > 10 deg
SAMPLE_HZ_EST = 50              # UPDATED: assume 50 Hz for persistence counting

PERSIST_SEC = 3.0               # NEW: must sustain bad posture for 3 seconds
PERSIST_SAMPLES = int(PERSIST_SEC * SAMPLE_HZ_EST)  # 150 samples at 50 Hz

COOLDOWN_SEC = 10.0             # cooldown after each alert

LOG_DIR = "logs"
STRETCH_URL = "https://www.youtube.com/watch?v=BPlCatqZRPI"

# =========================
# Head Tracker decoding
# =========================
def convert(msb, lsb, degrees=True):
    i = (128 * msb) + lsb
    if i >= 8192:
        i -= 16384
    return i * (0.02797645484 if degrees else 0.00048828125)

def read_headtracker_stream():
    with mido.open_input("Head Tracker") as inport:
        for msg in inport:
            if hasattr(msg, "data") and len(msg.data) > 10:
                if msg.data[3] == 64 and msg.data[4] == 0:
                    t = time.time()
                    yaw = convert(msg.data[5], msg.data[6])
                    pitch = convert(msg.data[7], msg.data[8])
                    roll = convert(msg.data[9], msg.data[10])
                    yield (t, yaw, pitch, roll)

# =========================
# Sound loop (macOS)
# =========================
def start_sound_loop():
    """
    Plays a short macOS system sound repeatedly using afplay.
    Returns (stop_event, thread).
    """
    stop_evt = threading.Event()
    sound_path = "/System/Library/Sounds/Ping.aiff"

    def _loop():
        while not stop_evt.is_set():
            try:
                subprocess.run(
                    ["afplay", sound_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except Exception:
                break
            time.sleep(0.05)

    th = threading.Thread(target=_loop, daemon=True)
    th.start()
    return stop_evt, th

# =========================
# macOS alert dialog
# =========================
def show_alert_dialog(deviation: float, threshold: float):
    """
    Blocking system alert.
    Buttons:
      - Open video
      - Dismiss
    Returns (dismiss_time_epoch, button_name)
    """
    stop_evt, _ = start_sound_loop()

    title = "Posture Check"
    message = (
        "Bad posture is sustaining.\n"
        "Please adjust your posture now.\n\n"
        f"Pitch deviation: {deviation:+.1f}°   (limit: ±{threshold:.1f}°)\n\n"
        "Stretch for 5 minutes:\n"
        f"{STRETCH_URL}"
    )

    script = (
        'set r to display alert "{title}" message "{msg}" as critical '
        'buttons {{"Dismiss","Open video"}} default button "Open video"\n'
        'return "button returned:" & (button returned of r)'
    ).format(
        title=title.replace('"', '\\"'),
        msg=message.replace('"', '\\"'),
    )

    button_name = "Dismiss"
    try:
        proc = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        out = (proc.stdout or "").strip()
        if out.startswith("button returned:"):
            button_name = out.split("button returned:", 1)[1].strip() or "Dismiss"
    finally:
        stop_evt.set()
        time.sleep(0.2)

    if button_name == "Open video":
        subprocess.run(["open", STRETCH_URL], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return time.time(), button_name

# =========================
# Helpers
# =========================
def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)

def median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return float(s[n // 2])
    return float(0.5 * (s[n // 2 - 1] + s[n // 2]))

# =========================
# MAIN
# =========================
def main():
    participant = input("Participant code (e.g., P01): ").strip() or "PXX"

    out_dir = os.path.join(LOG_DIR, participant, "alert1_posture_only")
    safe_mkdir(out_dir)
    out_csv = os.path.join(out_dir, f"posture_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    if "Head Tracker" not in mido.get_input_names():
        print("Head Tracker not found. Connect/pair it first.")
        return

    rows = []

    # -------------------------
    # Baseline
    # -------------------------
    print(f"Baseline: sit neutral for {BASELINE_SEC} seconds.")
    baseline_pitches = []
    t0 = time.time()
    stream = read_headtracker_stream()

    while time.time() - t0 < BASELINE_SEC:
        _, _, pitch, _ = next(stream)
        baseline_pitches.append(float(pitch))

    P_base = median(baseline_pitches)
    print(f"Baseline pitch (median): {P_base:.2f} deg")
    print(f"Monitoring for {TASK_SEC} seconds...")

    # -------------------------
    # Monitoring state
    # -------------------------
    last_alert_time = -1e9
    alert_active = False
    bad_counter = 0  # counts consecutive bad samples (|dev| > 10°)

    task_start = time.time()

    try:
        for _, yaw, pitch, roll in stream:
            now = time.time()
            if now - task_start > TASK_SEC:
                break

            deviation = float(pitch - P_base)
            dev_abs = abs(deviation)

            # -------------------------
            # Persistence (3 seconds at 50 Hz => 150 samples)
            # If posture returns to within ±10°, reset persistence counter.
            # -------------------------
            if dev_abs > PITCH_THRESH_DEG:
                bad_counter += 1
            else:
                bad_counter = 0

            sustained_bad = (bad_counter >= PERSIST_SAMPLES)

            # Cooldown
            can_alert = (now - last_alert_time) >= COOLDOWN_SEC

            # Log continuous stream
            rows.append({
                "t_epoch": now,
                "t_from_task_start_sec": now - task_start,
                "yaw": float(yaw),
                "pitch": float(pitch),
                "roll": float(roll),
                "P_base": float(P_base),
                "pitch_deviation": deviation,
                "pitch_dev_abs": dev_abs,
                "bad_counter_samples": int(bad_counter),
                "sustained_bad_3s": int(sustained_bad),
                "in_cooldown": int(not can_alert),
                "alert_active": int(alert_active),
            })

            # Fire alert only when sustained bad posture AND not in cooldown AND no overlapping alert
            if sustained_bad and can_alert and not alert_active:
                alert_active = True
                last_alert_time = now
                alert_started_at = now

                dismissed_at, button = show_alert_dialog(deviation, PITCH_THRESH_DEG)

                rows.append({
                    "t_epoch": dismissed_at,
                    "t_from_task_start_sec": dismissed_at - task_start,
                    "EVENT": "ALERT_DISMISSED",
                    "button": button,
                    "alert_started_at": alert_started_at,
                    "alert_dismissed_at": dismissed_at,
                    "P_base": float(P_base),
                    "pitch_deviation_at_alert": deviation,
                    "pitch_threshold": float(PITCH_THRESH_DEG),
                    "persist_sec": float(PERSIST_SEC),
                    "persist_samples": int(PERSIST_SAMPLES),
                    "cooldown_sec": float(COOLDOWN_SEC),
                    "stretch_url": STRETCH_URL,
                })

                alert_active = False

                # Reset persistence counter after an alert so user gets a fresh 3s window
                bad_counter = 0

            # Minimal terminal status (~every 2 seconds at 50Hz -> every 100 samples)
            if len(rows) % 100 == 0:
                print(
                    f"t={now-task_start:5.1f}s pitch={pitch:7.2f} dev={deviation:+6.1f} "
                    f"bad_cnt={bad_counter:3d}/{PERSIST_SAMPLES} cooldown={int(not can_alert)}",
                    end="\r"
                )

    except KeyboardInterrupt:
        print("\nStopped.")

    # -------------------------
    # Save
    # -------------------------
    if rows:
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved: {out_csv}")
    else:
        print("\nNo data collected, nothing saved.")

if __name__ == "__main__":
    main()
