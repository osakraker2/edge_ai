from flask import Flask, render_template, jsonify, send_from_directory
import os
import time
import csv
import threading
import numpy as np
from collections import deque

from max30102 import MAX30102
from pan_tompkins import PanTompkins

from arrhythmia_model import (
    ensure_model_async,
    score_latest_session_async,
    get_train_status,
    get_score_status,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARR_PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arrhythmia_model", "plots")

app = Flask(__name__)

PHASE_WAIT_FINGER = "WAIT_FINGER"
PHASE_CALIBRATION = "CALIBRATION"
PHASE_STABILIZING = "STABILIZING"
PHASE_RECORDING = "RECORDING"
PHASE_DONE = "DONE"
PHASE_ERROR = "ERROR"

FINGER_THRESHOLD = 160000
FINGER_STABLE_SAMPLES = 48
FINGER_STABLE_RATIO = 0.75
FINGER_LOST_GRACE_SEC = 1.0

CALIBRATION_SEC = 5.0
RECORDING_SEC = 60.0
SAMPLE_HZ = 100.0

SETTLE_MIN_SEC = 5.0
POST_SETTLE_DELAY_SEC = 0.30

BPM_MIN = 55.0
BPM_MAX = 110.0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
NABIZ_CSV_PATH = os.path.join(OUTPUT_DIR, "nabiz.csv")

PT_PLOTS_DIR = os.path.join(BASE_DIR, "pan_tompkins", "plots")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

_state_lock = threading.Lock()
_state = {
    "phase": PHASE_WAIT_FINGER,
    "message": "Parmağınızı lütfen sensöre koyun",
    "phase_started_at": None,
    "finger_present": False,
    "last_ir": 0,
    "last_red": 0,
    "bpm": None,
    "bpm_raw": None,
    "bpm_avg": None,
    "bpm_quality": 0.0,
    "settling_sec": None,
    "settled_at": None,
    "last_sample_at": None,
    "last_finger_seen_at": None,
    "output_path": None,
    "_file": None,
    "_writer": None,
    "samples_written": 0,
}

_finger_window = deque(maxlen=FINGER_STABLE_SAMPLES)
_ir_window = deque(maxlen=int(SAMPLE_HZ * 10))
_last_bpm_write_at = 0.0
_last_bpm_compute_at = 0.0
_bpm_ema = None
_last_bpm_valid = None
_bpm_sum = 0
_bpm_count = 0
_good_bpm_streak = 0
_bpm_stable_window = deque(maxlen=6)
_worker_started = False
_sensor = None
_stop_event = threading.Event()
_worker_thread = None

_plot_worker_started = False
_plot_worker_thread = None

_csv_analysis_lock = threading.Lock()
_csv_analysis_state = {
    "running": False,
    "status": "idle",
    "message": "",
    "started_at": None,
    "finished_at": None,
    "session_path": None,
}
_csv_analysis_thread = None


def _set_phase_locked(phase: str, message: str) -> None:
    _state["phase"] = phase
    _state["message"] = message
    _state["phase_started_at"] = time.time()


def _close_file_locked() -> None:
    f = _state.get("_file")
    if not f:
        return
    try:
        f.flush()
        f.close()
    except Exception:
        pass
    _state["_file"] = None
    _state["_writer"] = None


def _reset_session_locked() -> None:
    _close_file_locked()
    _state["output_path"] = None
    _state["samples_written"] = 0
    _state["bpm"] = None
    _state["bpm_raw"] = None
    _state["bpm_avg"] = None
    _state["bpm_quality"] = 0.0
    _state["settling_sec"] = None
    _state["settled_at"] = None
    global _bpm_ema, _last_bpm_valid, _last_bpm_compute_at, _last_bpm_write_at
    _bpm_ema = None
    _last_bpm_valid = None
    _last_bpm_compute_at = 0.0
    _last_bpm_write_at = 0.0
    _set_phase_locked(PHASE_WAIT_FINGER, "Parmağınızı lütfen sensöre koyun")


def _start_recording_locked() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"session_{ts}.csv")

    _cleanup_outputs_locked(keep_session_path=path)
    _reset_nabiz_csv_locked()

    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["t", "red", "ir"])

    _state["_file"] = f
    _state["_writer"] = w
    _state["output_path"] = path
    _state["samples_written"] = 0
    global _bpm_sum, _bpm_count, _good_bpm_streak
    _bpm_sum = 0
    _bpm_count = 0
    _good_bpm_streak = 0
    _bpm_stable_window.clear()
    global _last_bpm_write_at, _last_bpm_compute_at
    _last_bpm_write_at = 0.0
    _last_bpm_compute_at = 0.0
    _state["bpm_avg"] = None
    # settling_sec / settled_at bilgisi STABILIZING fazında ölçülür ve burada sıfırlanmaz.
    _set_phase_locked(PHASE_RECORDING, "Kayıt başladı. Lütfen parmağınızı çekmeyin")


def _finish_recording_locked() -> None:
    _close_file_locked()
    global _bpm_sum, _bpm_count
    if _bpm_count > 0:
        _state["bpm_avg"] = int(round(_bpm_sum / float(_bpm_count)))
    _set_phase_locked(PHASE_DONE, "Kayıt tamamlandı. Lütfen parmağınızı sensörden çekiniz")
    _cleanup_outputs_locked(keep_session_path=_state.get("output_path"))
    _stop_event.set()


def _estimate_bpm_from_ir_window(ir_values, fs: float, prev_bpm: float | None):
    # Autocorrelation tabanlı daha stabil BPM tahmini
    n = len(ir_values)
    if n < int(fs * 6):
        return None, 0.0

    mean = sum(ir_values) / n
    x = [v - mean for v in ir_values]

    # Basit high-pass: yavaş drift'i azalt
    win = max(1, int(fs * 1.2))
    if win > 1 and n > win:
        ma = []
        s = sum(x[:win])
        ma.append(s / win)
        for i in range(win, n):
            s += x[i] - x[i - win]
            ma.append(s / win)
        # ma uzunluğu n-win+1; merkezleyip çıkar
        offset = (win - 1) // 2
        y = []
        for i in range(n):
            j = min(max(i - offset, 0), len(ma) - 1)
            y.append(x[i] - ma[j])
        x = y

    energy = sum(v * v for v in x)
    if energy <= 0:
        return None, 0.0

    bpm_min = BPM_MIN
    bpm_max = BPM_MAX
    lag_min = int(fs * 60.0 / bpm_max)
    lag_max = int(fs * 60.0 / bpm_min)
    lag_min = max(lag_min, 2)
    lag_max = min(lag_max, n - 2)
    if lag_max <= lag_min:
        return None, 0.0

    def _corr_at(lag: int) -> float:
        c = 0.0
        for i in range(n - lag):
            c += x[i] * x[i + lag]
        return c

    # Aday lag'ları tara ve en iyi birkaçını sakla
    candidates = []
    best_lag = None
    best_corr = -1e18
    for lag in range(lag_min, lag_max + 1):
        c = _corr_at(lag)
        if c > best_corr:
            best_corr = c
            best_lag = lag
        candidates.append((lag, c))

    if best_lag is None:
        return None, 0.0

    # En güçlü adayların alt kümesi
    candidates.sort(key=lambda t: t[1], reverse=True)
    top = candidates[:12]

    # Harmonik adayları da dahil et (2x ve 1/2x)
    expanded = {lag: corr for lag, corr in top}
    for lag, corr in list(top):
        lag2 = lag * 2
        if lag2 <= lag_max:
            expanded.setdefault(lag2, _corr_at(lag2))
        if lag % 2 == 0:
            lagh = lag // 2
            if lagh >= lag_min:
                expanded.setdefault(lagh, _corr_at(lagh))

    # Seçim: önceki BPM varsa ona en yakın güçlü adayı seç
    chosen_lag = None
    chosen_corr = None
    if prev_bpm is not None and prev_bpm > 0:
        best_dist = None
        for lag, corr in expanded.items():
            # sadece yeterince güçlüleri değerlendir
            if corr < 0.85 * best_corr:
                continue
            bpm = 60.0 * fs / float(lag)
            if bpm < 35 or bpm > 220:
                continue
            dist = abs(bpm - float(prev_bpm))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                chosen_lag = lag
                chosen_corr = corr

    if chosen_lag is None:
        best_score = -1e18
        for lag, corr in expanded.items():
            if corr < 0.85 * best_corr:
                continue
            bpm = 60.0 * fs / float(lag)
            if bpm < 35 or bpm > 220:
                continue
            score = corr / (lag ** 0.06)
            if score > best_score:
                best_score = score
                chosen_lag = lag
                chosen_corr = corr

    if chosen_lag is None:
        chosen_lag = best_lag
        chosen_corr = best_corr

    best_lag = chosen_lag
    best_corr = float(chosen_corr)

    quality = best_corr / energy
    if quality < 0.15:
        return None, float(quality)

    bpm = 60.0 * fs / float(best_lag)
    if bpm < 35 or bpm > 220:
        return None, float(quality)
    return int(round(bpm)), float(quality)


def _smooth_bpm(raw_bpm: int):
    global _bpm_ema
    if raw_bpm is None:
        return _bpm_ema
    if _bpm_ema is None:
        _bpm_ema = float(raw_bpm)
        return int(round(_bpm_ema))

    # Ani sıçrama limit: tek update'de max ±8 bpm
    delta = float(raw_bpm) - _bpm_ema
    if delta > 8:
        raw_bpm = int(round(_bpm_ema + 8))
    elif delta < -8:
        raw_bpm = int(round(_bpm_ema - 8))

    alpha = 0.18
    _bpm_ema = (1 - alpha) * _bpm_ema + alpha * float(raw_bpm)
    return int(round(_bpm_ema))


def _update_bpm_estimate(phase: str, bpm_raw: int | None, q: float) -> int | None:
    global _bpm_ema
    # Kalite düşükse yeni tahmini kabul etme
    if bpm_raw is None or q < 0.20:
        return int(round(_bpm_ema)) if _bpm_ema is not None else None

    # Fazlara göre daha kontrollü güncelle
    if _bpm_ema is None:
        _bpm_ema = float(bpm_raw)
        return int(round(_bpm_ema))

    # outlier reddi: aşırı uzaksa ve kalite çok yüksek değilse güncelleme
    if abs(float(bpm_raw) - _bpm_ema) > 20 and q < 0.35:
        return int(round(_bpm_ema))

    # Clamp + EMA
    if phase == PHASE_RECORDING:
        max_step = 4.0
        alpha = 0.10
    else:
        max_step = 6.0
        alpha = 0.18

    target = float(bpm_raw)
    delta = target - _bpm_ema
    if delta > max_step:
        target = _bpm_ema + max_step
    elif delta < -max_step:
        target = _bpm_ema - max_step

    _bpm_ema = (1 - alpha) * _bpm_ema + alpha * target
    return int(round(_bpm_ema))


def _reset_nabiz_csv_locked() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        with open(NABIZ_CSV_PATH, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "bpm"])
    except Exception:
        pass


def _cleanup_outputs_locked(keep_session_path: str | None) -> None:
    # outputs altında sadece iki dosya kalsın: nabiz.csv ve en son session_*.csv
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for name in os.listdir(OUTPUT_DIR):
            path = os.path.join(OUTPUT_DIR, name)
            if not os.path.isfile(path):
                continue
            if path == NABIZ_CSV_PATH:
                continue
            if keep_session_path is not None and path == keep_session_path:
                continue
            if name.startswith("session_") and name.endswith(".csv"):
                try:
                    os.remove(path)
                except Exception:
                    pass
            else:
                # diğer tüm dosyaları da temizle
                try:
                    os.remove(path)
                except Exception:
                    pass
    except Exception:
        return


def _get_latest_session_path() -> str | None:
    try:
        if not os.path.isdir(OUTPUT_DIR):
            return None
        best = None
        best_mtime = None
        for name in os.listdir(OUTPUT_DIR):
            if not (name.startswith("session_") and name.endswith(".csv")):
                continue
            path = os.path.join(OUTPUT_DIR, name)
            try:
                mtime = os.path.getmtime(path)
            except Exception:
                continue
            if best is None or (best_mtime is not None and mtime > best_mtime) or best_mtime is None:
                best = path
                best_mtime = mtime
        return best
    except Exception:
        return None


def _append_bpm_csv(now: float, bpm: int) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(NABIZ_CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([now, bpm])


def _update_finger_presence_locked(ir: int, now: float) -> None:
    is_present = ir >= FINGER_THRESHOLD
    _finger_window.append(1 if is_present else 0)

    stable_ratio = (sum(_finger_window) / len(_finger_window)) if _finger_window else 0.0
    finger_present = stable_ratio >= FINGER_STABLE_RATIO and len(_finger_window) == _finger_window.maxlen

    if is_present:
        _state["last_finger_seen_at"] = now

    _state["finger_present"] = finger_present


def _tick_state_machine_locked(now: float) -> None:
    phase = _state["phase"]
    phase_started_at = _state.get("phase_started_at")

    last_seen = _state.get("last_finger_seen_at")
    finger_recent = last_seen is not None and (now - last_seen) <= FINGER_LOST_GRACE_SEC

    if phase == PHASE_WAIT_FINGER:
        if _state["finger_present"]:
            _set_phase_locked(PHASE_CALIBRATION, "Kalibrasyon (5 sn). Veri kaydedilmiyor")
        return

    if phase == PHASE_CALIBRATION:
        if not finger_recent:
            _reset_session_locked()
            return
        if phase_started_at is not None and (now - phase_started_at) >= CALIBRATION_SEC:
            _bpm_stable_window.clear()
            _state["settling_sec"] = None
            _state["settled_at"] = None
            _set_phase_locked(PHASE_STABILIZING, "Sinyal oturuyor. Lütfen parmağınızı çekmeyin")
        return

    if phase == PHASE_STABILIZING:
        if not finger_recent:
            _reset_session_locked()
            return
        # Settled olduysa küçük bir gecikmeden sonra kaydı başlat
        settled_at = _state.get("settled_at")
        if settled_at is not None and (now - settled_at) >= POST_SETTLE_DELAY_SEC:
            _start_recording_locked()
        return

    if phase == PHASE_RECORDING:
        if not finger_recent:
            _reset_session_locked()
            return
        if phase_started_at is not None and (now - phase_started_at) >= RECORDING_SEC:
            _finish_recording_locked()
        return

    if phase in (PHASE_DONE, PHASE_ERROR):
        if _state["finger_present"]:
            return
        _reset_session_locked()


def _maybe_write_sample_locked(now: float) -> None:
    if _state["phase"] != PHASE_RECORDING:
        return

    f = _state.get("_file")
    if not f:
        return

    w = _state.get("_writer")
    if not w:
        return

    try:
        w.writerow([now, _state["last_red"], _state["last_ir"]])
        _state["samples_written"] += 1
    except Exception:
        _set_phase_locked(PHASE_ERROR, "Dosyaya yazma hatası")
        _close_file_locked()


def _ensure_sensor_started() -> None:
    global _sensor
    if _sensor is not None:
        return
    _sensor = MAX30102()


def _worker_loop() -> None:
    interval = 1.0 / SAMPLE_HZ

    try:
        _ensure_sensor_started()
    except Exception:
        with _state_lock:
            _set_phase_locked(PHASE_ERROR, "Sensör başlatılamadı (I2C/izin/kablo?)")
        return

    global _last_bpm_write_at, _last_bpm_compute_at, _last_bpm_valid, _bpm_sum, _bpm_count, _good_bpm_streak
    next_tick = time.perf_counter()
    while not _stop_event.is_set():
        # Hassas 25Hz scheduler
        next_tick += interval
        now = time.time()
        try:
            red, ir = _sensor.read_fifo()
            red = int(red) if red is not None else 0
            ir = int(ir) if ir is not None else 0
        except Exception:
            red, ir = 0, 0

        with _state_lock:
            _state["last_red"] = red
            _state["last_ir"] = ir
            _state["last_sample_at"] = now

            _ir_window.append(ir)
            # BPM hesaplamasını her örnekte değil ~1Hz yap (25Hz'i tutturmak için)
            if (now - _last_bpm_compute_at) >= 1.0:
                bpm_raw, q = _estimate_bpm_from_ir_window(list(_ir_window), SAMPLE_HZ, _last_bpm_valid)
                _state["bpm_raw"] = bpm_raw
                _state["bpm_quality"] = float(q)
                bpm = _update_bpm_estimate(_state["phase"], bpm_raw, float(q))
                _state["bpm"] = bpm
                if bpm is not None and float(q) >= 0.20:
                    _last_bpm_valid = float(bpm)
                _last_bpm_compute_at = now
            else:
                bpm = _state.get("bpm")
                q = _state.get("bpm_quality", 0.0)

            # Stabilizasyon: (kayıt başlamadan önce) STABILIZING fazında yakala
            if _state["phase"] == PHASE_STABILIZING:
                started = _state.get("phase_started_at")
                elapsed = (now - started) if started is not None else None

                if bpm_raw is not None and q >= 0.20:
                    _bpm_stable_window.append(int(bpm_raw))
                else:
                    _bpm_stable_window.clear()

                if _state.get("settled_at") is None and elapsed is not None and elapsed >= SETTLE_MIN_SEC:
                    if len(_bpm_stable_window) == _bpm_stable_window.maxlen:
                        vmin = min(_bpm_stable_window)
                        vmax = max(_bpm_stable_window)
                        if (vmax - vmin) <= 6:
                            _state["settling_sec"] = round(elapsed, 2)
                            _state["settled_at"] = now
                            # Oturma anında BPM'i ankrajla (başlangıç hatalarını engeller)
                            try:
                                vals = sorted(_bpm_stable_window)
                                mid = vals[len(vals) // 2]
                            except Exception:
                                mid = None
                            if mid is not None:
                                _state["bpm"] = int(mid)
                                _state["bpm_raw"] = int(mid)
                                _last_bpm_valid = float(mid)
                                _bpm_ema = float(mid)

            # Avg BPM: sadece gerçek kayıt fazında biriktir
            if _state["phase"] == PHASE_RECORDING:
                started = _state.get("phase_started_at")
                rec_elapsed = (now - started) if started is not None else None
                # ilk 5sn ve düşük kaliteyi avg hesabına katma
                if bpm is not None and rec_elapsed is not None and rec_elapsed >= 5.0 and float(q) >= 0.20:
                    _bpm_sum += int(bpm)
                    _bpm_count += 1

            if bpm is not None and _state["phase"] == PHASE_RECORDING:
                if now - _last_bpm_write_at >= 1.0:
                    try:
                        _append_bpm_csv(now, bpm)
                        _last_bpm_write_at = now
                    except Exception:
                        pass

            _update_finger_presence_locked(ir, now)
            _tick_state_machine_locked(now)
            _maybe_write_sample_locked(now)

        # Sleep remaining time (if we're behind, don't sleep)
        remaining = next_tick - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)

    try:
        if _sensor is not None:
            _sensor.bus.close()
    except Exception:
        pass


def _ensure_worker_started() -> None:
    global _worker_started, _worker_thread
    if _worker_started:
        return

    _stop_event.clear()
    t = threading.Thread(target=_worker_loop, daemon=True)
    t.start()
    _worker_thread = t
    _worker_started = True


def _restart_worker() -> None:
    global _worker_started, _sensor, _worker_thread
    with _state_lock:
        _state["bpm"] = None
        _state["last_ir"] = 0
        _state["last_red"] = 0
        _state["last_sample_at"] = None
        _state["last_finger_seen_at"] = None
        _state["finger_present"] = False
        _state["samples_written"] = 0
        _set_phase_locked(PHASE_WAIT_FINGER, "Parmağınızı lütfen sensöre koyun")

    try:
        if _sensor is not None:
            _sensor.bus.close()
    except Exception:
        pass

    _sensor = None
    _worker_started = False
    _worker_thread = None
    _ensure_worker_started()


def _save_session_raw_plot(session_path: str) -> None:
    t, red, ir = _load_csv_data(session_path)
    if len(ir) == 0 or len(t) == 0:
        return

    if len(t) < 5:
        return

    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return

    fs_est = 1.0 / float(np.median(dt))
    if not np.isfinite(fs_est) or fs_est <= 1.0:
        return

    win_n = int(round(fs_est * 10.0))
    win_n = max(50, min(win_n, len(ir)))

    # pick best window by band-limited energy ratio
    best_i = 0
    best_score = -1.0
    step = max(1, win_n // 2)
    for i0 in range(0, max(1, len(ir) - win_n + 1), step):
        x = ir[i0 : i0 + win_n]
        if len(x) < win_n:
            continue
        if np.std(x) < 1e-6:
            continue

        x = x.astype(float)
        x = x - np.mean(x)
        rfft = np.fft.rfft(x)
        pxx = (np.abs(rfft) ** 2)
        freqs = np.fft.rfftfreq(len(x), d=1.0 / fs_est)

        band = (freqs >= 0.7) & (freqs <= 3.0)
        total = float(np.sum(pxx) + 1e-12)
        score = float(np.sum(pxx[band]) / total)

        # prefer windows with more dynamic range (avoid flat segments)
        score *= float(np.std(x))

        if score > best_score:
            best_score = score
            best_i = i0

    i0 = best_i
    x_t = t[i0 : i0 + win_n]
    x_ir = ir[i0 : i0 + win_n].astype(float)

    # resample to 100Hz, 10 seconds => 1000 points
    t0 = float(x_t[0])
    t1 = float(x_t[-1])
    target_dur = 10.0
    if t1 - t0 < 0.2:
        return
    if (t1 - t0) > target_dur:
        t1 = t0 + target_dur

    target_t = np.linspace(t0, t1, 1000)
    # ensure monotonic x_t for interp
    order = np.argsort(x_t)
    x_t_sorted = x_t[order]
    x_ir_sorted = x_ir[order]
    y = np.interp(target_t, x_t_sorted, x_ir_sorted)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig = plt.figure(figsize=(10.24, 3.84), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(len(y)), y, color="blue", linewidth=1.0)
    ax.set_title("Yeni Oturum: Ham IR Sinyali (100Hz, 10sn, En Uygun Segment)")
    ax.set_xlabel("Örnek Sayısı")
    ax.set_ylabel("Ham Değer (IR)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "session_raw.png"))
    plt.close(fig)


def _save_nabiz_plot() -> None:
    if not os.path.isfile(NABIZ_CSV_PATH):
        return
    t_vals = []
    bpm_vals = []
    try:
        with open(NABIZ_CSV_PATH, "r", newline="") as f:
            r = csv.reader(f)
            _ = next(r, None)
            for row in r:
                if len(row) < 2:
                    continue
                try:
                    t_vals.append(float(row[0]))
                    bpm_vals.append(float(row[1]))
                except Exception:
                    continue
    except Exception:
        return

    if len(bpm_vals) == 0:
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig = plt.figure(figsize=(9.6, 3.6), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(bpm_vals, linewidth=1.4, color="#ffb703")
    ax.set_title("Nabiz (nabiz.csv)")
    ax.set_xlabel("sample")
    ax.set_ylabel("bpm")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "nabiz.png"))
    plt.close(fig)


def _plot_worker_loop() -> None:
    last_session_path = None
    last_session_mtime = None
    last_nabiz_mtime = None

    while True:
        try:
            session_path = _get_latest_session_path()
            if session_path and os.path.isfile(session_path):
                mtime = os.path.getmtime(session_path)
                if last_session_path != session_path or last_session_mtime != mtime:
                    last_session_path = session_path
                    last_session_mtime = mtime
                    _save_session_raw_plot(session_path)

            if os.path.isfile(NABIZ_CSV_PATH):
                nm = os.path.getmtime(NABIZ_CSV_PATH)
                if last_nabiz_mtime != nm:
                    last_nabiz_mtime = nm
                    _save_nabiz_plot()
        except Exception:
            pass

        time.sleep(0.6)


def _ensure_plot_worker_started() -> None:
    global _plot_worker_started, _plot_worker_thread
    if _plot_worker_started:
        return
    _plot_worker_started = True
    _plot_worker_thread = threading.Thread(target=_plot_worker_loop, daemon=True)
    _plot_worker_thread.start()


def _csv_analysis_worker(session_path: str) -> None:
    global _csv_analysis_state
    try:
        _save_session_raw_plot(session_path)
        _save_nabiz_plot()
        with _csv_analysis_lock:
            _csv_analysis_state["running"] = False
            _csv_analysis_state["status"] = "done"
            _csv_analysis_state["message"] = "ok"
            _csv_analysis_state["finished_at"] = time.time()
    except Exception as e:
        with _csv_analysis_lock:
            _csv_analysis_state["running"] = False
            _csv_analysis_state["status"] = "error"
            _csv_analysis_state["message"] = str(e)
            _csv_analysis_state["finished_at"] = time.time()


@app.route("/csv_analysis/start")
def csv_analysis_start():
    global _csv_analysis_thread
    session_path = _get_latest_session_path()
    if not session_path or not os.path.isfile(session_path):
        with _csv_analysis_lock:
            _csv_analysis_state["running"] = False
            _csv_analysis_state["status"] = "no_data"
            _csv_analysis_state["message"] = "No session CSV found"
            _csv_analysis_state["session_path"] = session_path
        return jsonify(status="no_data", message="No session CSV found", session_path=session_path)

    with _csv_analysis_lock:
        if _csv_analysis_state.get("running"):
            return jsonify(status="running")
        _csv_analysis_state["running"] = True
        _csv_analysis_state["status"] = "running"
        _csv_analysis_state["message"] = "running"
        _csv_analysis_state["started_at"] = time.time()
        _csv_analysis_state["finished_at"] = None
        _csv_analysis_state["session_path"] = session_path

    _csv_analysis_thread = threading.Thread(target=_csv_analysis_worker, args=(session_path,), daemon=True)
    _csv_analysis_thread.start()
    return jsonify(status="started", session_path=session_path)


@app.route("/csv_analysis/status")
def csv_analysis_status():
    with _csv_analysis_lock:
        st = dict(_csv_analysis_state)

    return jsonify(
        status=st.get("status", "idle"),
        running=bool(st.get("running")),
        message=st.get("message", ""),
        started_at=st.get("started_at"),
        finished_at=st.get("finished_at"),
        session_path=st.get("session_path"),
        plots={
            "session_raw": "/plots/session_raw.png",
            "nabiz": "/plots/nabiz.png",
        },
    )


@app.route("/arrhythmia/train")
def arrhythmia_train():
    ensure_model_async()
    return jsonify(status="started")


@app.route("/arrhythmia/train_status")
def arrhythmia_train_status():
    return jsonify(get_train_status())


@app.route("/arrhythmia/score_latest")
def arrhythmia_score_latest():
    # Requires a trained RR model. Uses current session data -> PanTompkins RR -> anomaly score.
    session_path = _get_latest_session_path()
    if not session_path or not os.path.isfile(session_path):
        return jsonify(status="no_data", message="No session data found")

    t, red, ir = _load_csv_data(session_path)
    if len(ir) == 0:
        return jsonify(status="empty", message="No IR data in session")

    processor = PanTompkins(fs=SAMPLE_HZ)
    pt = processor.process(ir)
    rr = pt.get("rr_intervals_ms") or []
    score_latest_session_async(rr)
    return jsonify(status="started", session_path=session_path, rr_count=len(rr))


@app.route("/arrhythmia/score_status")
def arrhythmia_score_status():
    return jsonify(get_score_status())


@app.route("/arr_plots/<path:filename>")
def arr_plots(filename: str):
    return send_from_directory(ARR_PLOTS_DIR, filename)


@app.route("/plots/<path:filename>")
def plots(filename: str):
    return send_from_directory(PLOTS_DIR, filename)


@app.route("/pt_plots/<path:filename>")
def pt_plots(filename: str):
    return send_from_directory(PT_PLOTS_DIR, filename)


@app.route("/")
def index():
    _ensure_worker_started()
    return render_template("index.html")


@app.route("/status")
def status():
    _ensure_worker_started()

    with _state_lock:
        now = time.time()
        phase = _state["phase"]
        started = _state.get("phase_started_at")

        remaining = None
        if phase == PHASE_CALIBRATION and started is not None:
            remaining = max(0.0, CALIBRATION_SEC - (now - started))
        elif phase == PHASE_RECORDING and started is not None:
            remaining = max(0.0, RECORDING_SEC - (now - started))

        return jsonify(
            phase=phase,
            message=_state["message"],
            finger_present=_state["finger_present"],
            red=_state["last_red"],
            ir=_state["last_ir"],
            bpm=_state["bpm"],
            bpm_avg=_state["bpm_avg"],
            bpm_quality=_state["bpm_quality"],
            settling_sec=_state["settling_sec"],
            remaining=remaining,
            samples_written=_state["samples_written"],
            output_path=_state["output_path"],
            nabiz_path=NABIZ_CSV_PATH,
        )


@app.route("/restart")
def restart():
    _restart_worker()
    return jsonify(status="ok")


@app.route("/cleanup")
def cleanup():
    with _state_lock:
        keep = _state.get("output_path") or _get_latest_session_path()
        _cleanup_outputs_locked(keep_session_path=keep)
    return jsonify(status="ok", keep=keep)


@app.route("/session_data")
def session_data():
    _ensure_worker_started()
    with _state_lock:
        path = _state.get("output_path") or _get_latest_session_path()

    if not path or not os.path.isfile(path):
        return jsonify(status="empty", path=path, t=[], ir=[], red=[])

    t_vals = []
    ir_vals = []
    red_vals = []

    try:
        with open(path, "r", newline="") as f:
            r = csv.reader(f)
            header = next(r, None)
            for row in r:
                if len(row) < 3:
                    continue
                try:
                    t_vals.append(float(row[0]))
                    red_vals.append(int(float(row[1])))
                    ir_vals.append(int(float(row[2])))
                except Exception:
                    continue
    except Exception:
        return jsonify(status="error", path=path, t=[], ir=[], red=[])

    n = len(t_vals)
    if n <= 0:
        return jsonify(status="empty", path=path, t=[], ir=[], red=[])

    max_points = 900
    step = max(1, n // max_points)
    if step > 1:
        t_vals = t_vals[::step]
        ir_vals = ir_vals[::step]
        red_vals = red_vals[::step]

    return jsonify(status="ok", path=path, t=t_vals, ir=ir_vals, red=red_vals)


@app.route("/nabiz_data")
def nabiz_data():
    """Return BPM time series from nabiz.csv (for UI plotting)."""
    _ensure_worker_started()
    t_vals = []
    bpm_vals = []
    try:
        if not os.path.isfile(NABIZ_CSV_PATH):
            return jsonify(status="no_data", path=NABIZ_CSV_PATH, t=[], bpm=[])
        with open(NABIZ_CSV_PATH, "r", newline="") as f:
            r = csv.reader(f)
            _ = next(r, None)
            for row in r:
                if len(row) < 2:
                    continue
                try:
                    t_vals.append(float(row[0]))
                    bpm_vals.append(float(row[1]))
                except Exception:
                    continue
    except Exception:
        return jsonify(status="error", path=NABIZ_CSV_PATH, t=[], bpm=[])

    n = min(len(t_vals), len(bpm_vals))
    if n == 0:
        return jsonify(status="empty", path=NABIZ_CSV_PATH, t=[], bpm=[])

    max_points = 900
    step = max(1, n // max_points)
    if step > 1:
        t_vals = t_vals[::step]
        bpm_vals = bpm_vals[::step]

    return jsonify(status="ok", path=NABIZ_CSV_PATH, t=t_vals, bpm=bpm_vals)


def _load_csv_data(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load CSV data and return t, red, ir as numpy arrays."""
    t_vals = []
    red_vals = []
    ir_vals = []
    try:
        with open(csv_path, "r", newline="") as f:
            r = csv.reader(f)
            header = next(r, None)
            for row in r:
                if len(row) < 3:
                    continue
                try:
                    t_vals.append(float(row[0]))
                    red_vals.append(float(row[1]))
                    ir_vals.append(float(row[2]))
                except Exception:
                    continue
    except Exception:
        pass
    return np.array(t_vals), np.array(red_vals), np.array(ir_vals)


def _detrend_signal(x: np.ndarray, fs: float) -> np.ndarray:
    win = int(max(1, round(fs * 0.75)))
    kernel = np.ones(win, dtype=float) / float(win)
    baseline = np.convolve(x, kernel, mode="same")
    return x - baseline


def _save_line_plot(path: str, y: np.ndarray, title: str, peaks: np.ndarray | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig = plt.figure(figsize=(9.6, 3.6), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(y, linewidth=1.5, color="red")
    if peaks is not None and len(peaks) > 0:
        peaks = peaks[(peaks >= 0) & (peaks < len(y))]
        ax.scatter(peaks, y[peaks], s=28, color="black")
    ax.set_title(title)
    ax.set_xlabel("sample")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


@app.route("/pan_tompkins")
def pan_tompkins_analysis():
    """Pan-Tompkins QRS detection on the latest session data."""
    # Load latest session CSV
    session_path = _get_latest_session_path()
    if not session_path or not os.path.isfile(session_path):
        return jsonify(status="no_data", message="No session data found")

    t, red, ir = _load_csv_data(session_path)
    if len(ir) == 0:
        return jsonify(status="empty", message="No IR data in session")

    # Run Pan-Tompkins
    processor = PanTompkins(fs=SAMPLE_HZ)
    result = processor.process(ir)

    detrended = np.array(result["detrended"], dtype=float)
    filtered = np.array(result["filtered"], dtype=float)
    derivative = np.array(result["derivative"], dtype=float)
    squared = np.array(result["squared"], dtype=float)
    integrated = np.array(result["integrated"], dtype=float)
    qrs_peaks = np.array(result["qrs_peaks"], dtype=int)

    os.makedirs(PT_PLOTS_DIR, exist_ok=True)
    _save_line_plot(os.path.join(PT_PLOTS_DIR, "detrended.png"), detrended[:2000], "Detrend")
    _save_line_plot(os.path.join(PT_PLOTS_DIR, "filtered.png"), filtered[:2000], "Filtered")
    _save_line_plot(os.path.join(PT_PLOTS_DIR, "derivative.png"), derivative[:2000], "Derivative")
    _save_line_plot(os.path.join(PT_PLOTS_DIR, "squared.png"), squared[:2000], "Squared")
    _save_line_plot(os.path.join(PT_PLOTS_DIR, "integrated.png"), integrated[:2000], "Integrated")
    _save_line_plot(os.path.join(PT_PLOTS_DIR, "final.png"), integrated[:2000], "Final Analysis", peaks=qrs_peaks)

    return jsonify(
        status="ok",
        session_path=session_path,
        total_samples=len(ir),
        fs=SAMPLE_HZ,
        bpm=result["bpm"],
        qrs_peaks=result["qrs_peaks"],
        rr_intervals_ms=result["rr_intervals_ms"],
        plots={
            "detrended": "/pt_plots/detrended.png",
            "filtered": "/pt_plots/filtered.png",
            "derivative": "/pt_plots/derivative.png",
            "squared": "/pt_plots/squared.png",
            "integrated": "/pt_plots/integrated.png",
            "final": "/pt_plots/final.png",
        },
    )


if __name__ == "__main__":
    with _state_lock:
        _cleanup_outputs_locked(keep_session_path=_get_latest_session_path())
    _ensure_worker_started()
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
