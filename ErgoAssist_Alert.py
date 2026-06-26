import os
import time
import math
import json
import queue
import threading
import subprocess
from collections import deque

import numpy as np
import pandas as pd
import joblib
import mido
from frenztoolkit import Streamer

from scipy.signal import welch
from scipy.integrate import simpson

# =========================================================================
# USER CONFIG
# =========================================================================
DEVICE_ID   = os.environ.get("FRENZ_DEVICE_ID", "FRENZM74")
PRODUCT_KEY = os.environ.get("FRENZ_PRODUCT_KEY", "")

EEG_MODEL_PATH = "eeg_models/eeg_rest_active_8020.pkl"
IMU_MODEL_PATH = "imu_models/imu_posture_good_bad.pkl"
IMU_BAD_LABEL  = "Bad"

FS_IMU   = 50.0
T_BASE   = 60.0
T_TASK   = 3 * 60.0
T_IMU    = 1.0
T_P      = 3.0
T_CONT   = 30.0
T_I      = 10.0
T_E      = 5.0

EEG_FS_ASSUMED = 125
BEEP_PERIOD_SEC = 0.35
LOG_ROOT = "logs"
SAVE_EVERY_N = 1500

FRENZ_COLS = ["ts", "LF", "OTEL", "REF1", "RF", "OTER", "REF2"]
USE_CH = ["LF", "OTEL", "RF", "OTER"]
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}

# =========================================================================
# Algorithm core
# =========================================================================
class AlertScheduler:
    def __init__(self, fs, T_p, T_cont, T_i, T_e):
        self.dt = 1.0 / fs
        self.N_p = math.ceil(T_p * fs)
        self.T_cont = T_cont
        self.T_i = T_i
        self.T_e = T_e
        self.bad_count = 0
        self.bad_dur = 0.0
        self.last_alert = -math.inf
        self.eeg_state = "Unknown"
        self.last_eeg = -math.inf
        self.events = []

    def step(self, t, is_bad, eeg_fn):
        if is_bad:
            self.bad_count += 1
            self.bad_dur += self.dt
        else:
            self.bad_count = 0
            self.bad_dur = 0.0

        sustained = self.bad_count >= self.N_p
        continuous = self.bad_dur >= self.T_cont

        if t - self.last_eeg >= self.T_e:
            self.eeg_state = eeg_fn()
            self.last_eeg = t

        interruptible = (self.eeg_state == "Rest")
        in_interval = (t - self.last_alert) < self.T_i

        trigger, reason = False, None
        if not in_interval:
            if continuous:
                trigger, reason = True, "continuous_bad"
            elif sustained and interruptible:
                trigger, reason = True, "sustained_and_interruptible"

        if trigger:
            self.events.append(t)
            self.last_alert = t
            self.bad_count = 0
            self.bad_dur = 0.0

        return trigger, reason


def extract_imu_features(yaw, pitch_dev, roll):
    def stats(x, name):
        x = np.asarray(x, dtype=float)
        return {
            f"{name}_mean": float(np.mean(x)),
            f"{name}_std": float(np.std(x)),
            f"{name}_min": float(np.min(x)),
            f"{name}_max": float(np.max(x)),
            f"{name}_range": float(np.max(x) - np.min(x)),
            f"{name}_median": float(np.median(x)),
            f"{name}_rms": float(np.sqrt(np.mean(x ** 2))),
        }

    feats = {}
    feats.update(stats(yaw, "yaw"))
    feats.update(stats(pitch_dev, "pitch_dev"))
    feats.update(stats(np.abs(pitch_dev), "abs_pitch_dev"))
    feats.update(stats(roll, "roll"))
    return feats


class IMUClassifier:
    def __init__(self, bundle, bad_label=IMU_BAD_LABEL):
        self.feature_cols = bundle["feature_cols"]
        self.scaler = bundle["scaler"]
        self.clf = bundle["clf"]
        self.classes = list(bundle["classes"])
        self.bad_label = bad_label
        self.last_label = "Unknown"

    def __call__(self, yaw, pitch_dev, roll):
        feats = extract_imu_features(yaw, pitch_dev, roll)
        row_f = {c: 0.0 for c in self.feature_cols}
        for k, v in feats.items():
            if k in row_f:
                row_f[k] = float(v)
        Xs = self.scaler.transform(pd.DataFrame([row_f], columns=self.feature_cols))
        self.last_label = str(self.clf.predict(Xs)[0])
        return self.last_label == self.bad_label


def _convert(msb, lsb, degrees=True):
    i = (128 * msb) + lsb
    if i >= 8192:
        i -= 16384
    return i * (0.02797645484 if degrees else 0.00048828125)


class HeadTrackerThread(threading.Thread):
    def __init__(self, port_name="Head Tracker", maxlen=20000):
        super().__init__(daemon=True)
        self.port_name = port_name
        self.sample_q = queue.Queue()
        self._stop = threading.Event()
        self._ready = threading.Event()

    def stop(self):
        self._stop.set()

    def ready(self):
        return self._ready.is_set()

    def run(self):
        if self.port_name not in mido.get_input_names():
            print("ERROR: Head Tracker MIDI not found.")
            return
        print("HeadTracker connected:", self.port_name)
        self._ready.set()
        with mido.open_input(self.port_name) as inport:
            for msg in inport:
                if self._stop.is_set():
                    break
                if not hasattr(msg, "data") or msg.data is None:
                    continue
                d = msg.data
                if len(d) < 11:
                    continue
                if d[3] == 64 and d[4] == 0:
                    t = time.time()
                    yaw = _convert(d[5], d[6])
                    pitch = _convert(d[7], d[8])
                    roll = _convert(d[9], d[10])
                    self.sample_q.put((t, yaw, pitch, roll))


def bandpowers(sig, fs=EEG_FS_ASSUMED):
    freqs, psd = welch(sig, fs=fs, nperseg=min(256, len(sig)))
    out = {}
    for b, (lo, hi) in BANDS.items():
        m = (freqs >= lo) & (freqs <= hi)
        out[f"{b}_p"] = float(simpson(psd[m], x=freqs[m])) if np.any(m) else 0.0
    total = out["theta_p"] + out["alpha_p"] + out["beta_p"] + 1e-12
    out["theta_rel"] = out["theta_p"] / total
    out["alpha_rel"] = out["alpha_p"] / total
    out["beta_rel"] = out["beta_p"] / total
    out["tbr"] = out["theta_p"] / (out["beta_p"] + 1e-12)
    return out


def extract_eeg_features(window_4ch):
    feats, tbr_list, ei_list = {}, [], []
    for i in range(4):
        bp = bandpowers(window_4ch[:, i])
        feats[f"eeg_Channel_{i+1}_Theta_power"] = bp["theta_p"]
        feats[f"eeg_Channel_{i+1}_Alpha_power"] = bp["alpha_p"]
        feats[f"eeg_Channel_{i+1}_Beta_power"] = bp["beta_p"]
        feats[f"eeg_Channel_{i+1}_Theta_rel_power"] = bp["theta_rel"]
        feats[f"eeg_Channel_{i+1}_Alpha_rel_power"] = bp["alpha_rel"]
        feats[f"eeg_Channel_{i+1}_Beta_rel_power"] = bp["beta_rel"]
        feats[f"eeg_Channel_{i+1}_theta_beta_ratio"] = bp["tbr"]
        ei = bp["beta_p"] / (bp["alpha_p"] + bp["theta_p"] + 1e-12)
        feats[f"TBR_ch{i+1}"] = bp["tbr"]
        feats[f"EngagementIndex_ch{i+1}"] = float(ei)
        tbr_list.append(bp["tbr"])
        ei_list.append(ei)
    feats["TBR_mean"] = float(np.mean(tbr_list))
    feats["EngagementIndex_mean"] = float(np.mean(ei_list))
    feats["TBR_std"] = float(np.std(tbr_list))
    feats["EngagementIndex_std"] = float(np.std(ei_list))
    return feats


class EEGClassifier:
    def __init__(self, streamer, bundle):
        self.streamer = streamer
        self.feature_cols = bundle["feature_cols"]
        self.scaler = bundle["scaler"]
        self.clf = bundle["clf"]
        self.classes = list(bundle["classes"])
        self.last_prob_rest = float("nan")

    def __call__(self):
        self.last_prob_rest = float("nan")
        raw = self.streamer.get_raw_data_realtime(window_in_seconds=T_E)
        eeg = raw.get("eeg", None)
        if eeg is None or len(eeg) == 0:
            return "Unknown"
        X4 = pd.DataFrame(eeg, columns=FRENZ_COLS)[USE_CH].to_numpy(dtype=float)
        if X4.shape[0] < 125:
            return "Unknown"
        feats = extract_eeg_features(X4)
        row_f = {c: 0.0 for c in self.feature_cols}
        for k, v in feats.items():
            if k in row_f:
                row_f[k] = float(v)
        Xs = self.scaler.transform(pd.DataFrame([row_f], columns=self.feature_cols))
        state = str(self.clf.predict(Xs)[0])
        if "Rest" in self.classes:
            self.last_prob_rest = float(self.clf.predict_proba(Xs)[0][self.classes.index("Rest")])
        return state


def show_alert_popup_blocking(message, title="TechNeck Alert"):
    script = f'display dialog "{message}" with title "{title}" buttons {{"OK"}} default button "OK"'
    subprocess.run(["osascript", "-e", script], check=False)


def beep_once():
    subprocess.run(["osascript", "-e", "beep 1"], check=False)


def alert_with_beep_until_ok(message):
    stop_flag = {"stop": False}

    def beeper():
        while not stop_flag["stop"]:
            beep_once()
            time.sleep(BEEP_PERIOD_SEC)

    th = threading.Thread(target=beeper, daemon=True)
    th.start()
    try:
        show_alert_popup_blocking(message)
    finally:
        stop_flag["stop"] = True
        th.join(timeout=1.0)


def incremental_save(rows, csv_path):
    if not rows:
        return
    pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def main():
    participant = input("Participant code (e.g., P01): ").strip()
    if not participant:
        print("Participant code required.")
        return

    if not PRODUCT_KEY:
        print("ERROR: set FRENZ_PRODUCT_KEY in the environment.")
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    session_id = f"{participant}_{ts}"
    out_dir = os.path.join(LOG_ROOT, participant, f"eeg_posture_alert_{ts}")
    ensure_dir(out_dir)

    print("Loading EEG model:", EEG_MODEL_PATH)
    eeg_bundle = joblib.load(EEG_MODEL_PATH)
    print("Loading IMU model:", IMU_MODEL_PATH)
    imu_bundle = joblib.load(IMU_MODEL_PATH)

    ht = HeadTrackerThread()
    ht.start()
    time.sleep(0.5)
    if not ht.ready():
        print("ERROR: Head Tracker not ready.")
        return

    streamer = Streamer(
        device_id=DEVICE_ID,
        product_key=PRODUCT_KEY,
        data_folder=os.path.join(out_dir, "frenz_data"),
        turn_off_light=True,
    )
    print("Starting Frenz session...")
    streamer.start()
    time.sleep(1.0)

    eeg_clf = EEGClassifier(streamer, eeg_bundle)
    imu_clf = IMUClassifier(imu_bundle)
    sched = AlertScheduler(fs=FS_IMU, T_p=T_P, T_cont=T_CONT, T_i=T_I, T_e=T_E)

    head_csv = os.path.join(out_dir, "headtracker_stream.csv")
    decision_csv = os.path.join(out_dir, "decision_log.csv")
    alert_csv = os.path.join(out_dir, "alerts.csv")
    metrics_json = os.path.join(out_dir, "ergonomic_metrics.json")

    head_rows, decision_rows, alert_rows_data = [], [], []
    unsaved_head, unsaved_decision = [], []

    baseline_ready = False
    baseline_yaw = 0.0
    baseline_pitch = 0.0
    baseline_roll = 0.0
    baseline_yaws, baseline_pitches, baseline_rolls = [], [], []

    imu_win = deque()
    t0 = time.time()
    start_task = None
    N_IMU = max(1, math.ceil(T_IMU * FS_IMU))

    print(f"BASELINE {T_BASE:.0f}s then MONITORING {T_TASK:.0f}s ...")

    running = True
    while running:
        try:
            t_wall, yaw, pitch, roll = ht.sample_q.get(timeout=0.5)
        except queue.Empty:
            if start_task is not None and time.time() - start_task >= T_TASK:
                break
            continue

        if not baseline_ready:
            baseline_yaws.append(yaw)
            baseline_pitches.append(pitch)
            baseline_rolls.append(roll)
            row = {
                "t_wall": t_wall, "phase": "baseline",
                "yaw": yaw, "pitch": pitch, "roll": roll,
                "baseline_yaw": 0.0, "baseline_pitch": 0.0, "baseline_roll": 0.0,
                "yaw_dev": 0.0, "pitch_dev": 0.0, "roll_dev": 0.0,
                "bad": 0, "bad_counter": 0, "bad_posture_dur_sec": 0.0,
            }
            head_rows.append(row)
            unsaved_head.append(row)
            if t_wall - t0 >= T_BASE:
                baseline_yaw = float(np.median(baseline_yaws)) if baseline_yaws else 0.0
                baseline_pitch = float(np.median(baseline_pitches)) if baseline_pitches else 0.0
                baseline_roll = float(np.median(baseline_rolls)) if baseline_rolls else 0.0
                baseline_ready = True
                start_task = t_wall
                with open(os.path.join(out_dir, "baseline.json"), "w") as f:
                    json.dump({
                        "baseline_yaw": baseline_yaw,
                        "baseline_pitch": baseline_pitch,
                        "baseline_roll": baseline_roll,
                        "baseline_sec": T_BASE,
                    }, f, indent=2)
                print("Baseline set. MONITORING...")
                incremental_save(unsaved_head, head_csv)
                unsaved_head.clear()
            continue

        yaw_dev = yaw - baseline_yaw
        pitch_dev = pitch - baseline_pitch
        roll_dev = roll - baseline_roll

        imu_win.append((t_wall, yaw_dev, pitch_dev, roll_dev))
        while imu_win and t_wall - imu_win[0][0] > T_IMU:
            imu_win.popleft()

        if len(imu_win) < N_IMU:
            continue

        win = np.array(imu_win)
        is_bad = imu_clf(win[:, 1], win[:, 2], win[:, 3])

        trigger, reason = sched.step(t_wall, is_bad, eeg_fn=eeg_clf)
        elapsed = t_wall - start_task

        hrow = {
            "t_wall": t_wall,
            "phase": "task",
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "baseline_yaw": baseline_yaw,
            "baseline_pitch": baseline_pitch,
            "baseline_roll": baseline_roll,
            "yaw_dev": yaw_dev,
            "pitch_dev": pitch_dev,
            "roll_dev": roll_dev,
            "bad": int(is_bad),
            "bad_counter": int(sched.bad_count),
            "bad_posture_dur_sec": float(sched.bad_dur),
        }
        head_rows.append(hrow)
        unsaved_head.append(hrow)

        drow = {
            "t_wall": t_wall,
            "elapsed_sec": elapsed,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "baseline_yaw": baseline_yaw,
            "baseline_pitch": baseline_pitch,
            "baseline_roll": baseline_roll,
            "yaw_dev": yaw_dev,
            "pitch_dev": pitch_dev,
            "roll_dev": roll_dev,
            "bad": int(is_bad),
            "bad_counter": int(sched.bad_count),
            "sustained_bad": int(sched.bad_count >= sched.N_p),
            "bad_posture_dur_sec": float(sched.bad_dur),
            "continuous_bad": int(sched.bad_dur >= sched.T_cont),
            "eeg_state": sched.eeg_state,
            "prob_rest": eeg_clf.last_prob_rest,
            "interruptible": int(sched.eeg_state == "Rest"),
            "in_cooldown": int((t_wall - sched.last_alert) < sched.T_i),
            "trigger": int(trigger),
            "reason": reason,
        }
        decision_rows.append(drow)
        unsaved_decision.append(drow)

        if trigger:
            alert_start = time.time()
            alert_with_beep_until_ok("Wrong posture sustained.\nPlease adjust your posture.")
            alert_end = time.time()
            alert_data = {
                "participant_id": participant,
                "session_id": session_id,
                "alert_num": len(alert_rows_data) + 1,
                "t_alert_start": alert_start,
                "t_alert_end": alert_end,
                "elapsed_alert_start": alert_start - start_task,
                "elapsed_alert_end": alert_end - start_task,
                "alert_duration_sec": alert_end - alert_start,
                "reason": reason,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "baseline_yaw": baseline_yaw,
                "baseline_pitch": baseline_pitch,
                "baseline_roll": baseline_roll,
                "yaw_deviation": yaw_dev,
                "pitch_deviation": pitch_dev,
                "roll_deviation": roll_dev,
                "eeg_state": sched.eeg_state,
                "prob_rest": eeg_clf.last_prob_rest,
            }
            alert_rows_data.append(alert_data)
            with open(os.path.join(out_dir, f"alert_{len(alert_rows_data):03d}_{int(alert_start)}.json"), "w") as f:
                json.dump(alert_data, f, indent=2)
            print(f"Alert #{len(alert_rows_data)}: {reason}")

        if len(unsaved_head) >= SAVE_EVERY_N:
            incremental_save(unsaved_head, head_csv)
            unsaved_head.clear()
        if len(unsaved_decision) >= SAVE_EVERY_N:
            incremental_save(unsaved_decision, decision_csv)
            unsaved_decision.clear()

        if elapsed >= T_TASK:
            running = False

    streamer.stop()
    ht.stop()

    incremental_save(unsaved_head, head_csv)
    incremental_save(unsaved_decision, decision_csv)
    pd.DataFrame(alert_rows_data).to_csv(alert_csv, index=False)
    for a in alert_rows_data:
        with open(os.path.join(out_dir, f"alert_{a['alert_num']:03d}_{int(a['t_alert_start'])}.json"), "w") as f:
            json.dump(a, f, indent=2)

    summary = {
        "participant_id": participant,
        "session_id": session_id,
        "condition": "P+C",
        "n_alerts": len(alert_rows_data),
        "task_duration_sec": round((time.time() - start_task), 1) if start_task else 0.0,
    }
    with open(metrics_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== SESSION SUMMARY =====")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nSaved to: {out_dir}")


if __name__ == "__main__":
    main()