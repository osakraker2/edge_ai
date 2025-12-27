import json
import os
import threading
import time
from dataclasses import dataclass

import numpy as np
import wfdb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "model.json")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
REPORT_PNG = os.path.join(PLOTS_DIR, "ai_saglik_raporu.png")


_train_lock = threading.Lock()
_train_state = {
    "status": "idle",
    "running": False,
    "message": "",
    "started_at": None,
    "finished_at": None,
}
_train_thread = None

_score_lock = threading.Lock()
_score_state = {
    "status": "idle",
    "running": False,
    "message": "",
    "started_at": None,
    "finished_at": None,
    "result": None,
}
_score_thread = None


def _list_local_records(db_root: str) -> list[str]:
    records = set()
    for root, _, files in os.walk(db_root):
        for fn in files:
            if fn.endswith(".hea"):
                records.add(os.path.splitext(fn)[0])
    return sorted(records)


def _load_annotations(record: str, db_root: str) -> tuple[np.ndarray, list[str], float]:
    hea = wfdb.rdheader(os.path.join(db_root, record))
    ann = wfdb.rdann(os.path.join(db_root, record), "atr")
    return ann.sample.astype(int), list(ann.symbol), float(hea.fs)


def _extract_rr_ms(samples: np.ndarray, symbols: list[str], fs: float) -> np.ndarray:
    normal_syms = {"N", "L", "R"}
    idx = [i for i, s in enumerate(symbols) if s in normal_syms]
    if len(idx) < 2:
        return np.array([], dtype=float)

    samp = samples[idx]
    rr_ms = np.diff(samp) * (1000.0 / fs)
    rr_ms = rr_ms[(rr_ms > 450) & (rr_ms < 1300)]
    return rr_ms.astype(float)


def _train_model() -> None:
    global _train_state
    os.makedirs(DATA_DIR, exist_ok=True)
    db_root = os.path.join(DATA_DIR, "mitdb")

    # Offline-first: prefer DATA_DIR/mitdb but also support DATA_DIR (wfdb may download flat).
    search_roots = []
    if os.path.isdir(db_root):
        search_roots.append(db_root)
    search_roots.append(DATA_DIR)

    records = []
    chosen_root = None
    for root in search_roots:
        recs = _list_local_records(root)
        if recs:
            records = recs
            chosen_root = root
            break

    if not records:
        # Download once into DATA_DIR, then operate offline from local files.
        wfdb.dl_database("mitdb", dl_dir=DATA_DIR)
        # Re-scan after download
        for root in search_roots:
            recs = _list_local_records(root)
            if recs:
                records = recs
                chosen_root = root
                break

    if not records or not chosen_root:
        raise RuntimeError("MITDB local records not found under data/ or data/mitdb")

    rr_all = []
    for rec in records:
        try:
            samples, symbols, fs = _load_annotations(rec, chosen_root)
            rr = _extract_rr_ms(samples, symbols, fs)
            if len(rr) > 0:
                rr_all.append(rr)
        except Exception:
            continue

    if not rr_all:
        raise RuntimeError("No RR intervals extracted from MITDB")

    rr = np.concatenate(rr_all)
    model = {
        "version": 1,
        "trained_at": time.time(),
        "n_rr": int(len(rr)),
        "mean": float(np.mean(rr)),
        "std": float(np.std(rr) + 1e-9),
        "q05": float(np.quantile(rr, 0.05)),
        "q95": float(np.quantile(rr, 0.95)),
    }

    with open(MODEL_PATH, "w") as f:
        json.dump(model, f)


def ensure_model_async() -> None:
    global _train_thread
    with _train_lock:
        if _train_state.get("running"):
            return
        _train_state["running"] = True
        _train_state["status"] = "running"
        _train_state["message"] = "training"
        _train_state["started_at"] = time.time()
        _train_state["finished_at"] = None

    def runner():
        try:
            _train_model()
            with _train_lock:
                _train_state["running"] = False
                _train_state["status"] = "done"
                _train_state["message"] = "ok"
                _train_state["finished_at"] = time.time()
        except Exception as e:
            with _train_lock:
                _train_state["running"] = False
                _train_state["status"] = "error"
                _train_state["message"] = str(e)
                _train_state["finished_at"] = time.time()

    _train_thread = threading.Thread(target=runner, daemon=True)
    _train_thread.start()


def get_train_status() -> dict:
    with _train_lock:
        st = dict(_train_state)
    st["model_exists"] = os.path.isfile(MODEL_PATH)
    return st


def _load_model() -> dict:
    if not os.path.isfile(MODEL_PATH):
        raise RuntimeError("Model not trained")
    with open(MODEL_PATH, "r") as f:
        return json.load(f)


def _score_rr(rr_ms: list[float]) -> dict:
    model = _load_model()
    rr = np.array([float(x) for x in rr_ms if x is not None], dtype=float)
    rr = rr[np.isfinite(rr)]

    if len(rr) < 3:
        return {
            "label": "UNKNOWN",
            "score": 0.0,
            "confidence": 0.0,
            "note": "Not enough RR intervals",
            "total_windows": 0,
            "risky_windows": 0,
            "risk_pct": 0.0,
        }

    mean_rr = float(np.mean(rr))
    q05 = float(model["q05"])
    q95 = float(model["q95"])
    mu = float(model["mean"])
    sigma = float(model["std"])

    out_frac = float(np.mean((rr < q05) | (rr > q95)))
    z = abs(mean_rr - mu) / sigma
    z_norm = min(1.0, z / 3.0)

    score = min(1.0, 0.5 * out_frac + 0.5 * z_norm)
    label = "ANOMALY" if score >= 0.5 else "NORMAL"
    confidence = float(min(1.0, max(0.0, 1.0 - score)))

    # Windowed risk series for report plotting (AI-style)
    win = 10
    step = 2
    probs = []
    if len(rr) >= win:
        for i in range(0, len(rr) - win + 1, step):
            rr_w = rr[i : i + win]
            out_w = float(np.mean((rr_w < q05) | (rr_w > q95)))
            z_w = abs(float(np.mean(rr_w)) - mu) / sigma
            z_w = min(1.0, z_w / 3.0)
            s_w = min(1.0, 0.5 * out_w + 0.5 * z_w)
            probs.append(float(s_w))

    risky_windows = int(sum(1 for p in probs if p > 0.5))
    total_windows = int(len(probs))
    risk_pct = float((risky_windows / total_windows) * 100.0) if total_windows > 0 else 0.0

    # Save report PNG similar to ai_final_test.py
    try:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.figure(figsize=(15, 6))
        plt.plot(probs, label="Anomali Olasılığı", color="blue", linewidth=1.5)
        plt.axhline(y=0.5, color="red", linestyle="--", label="Eşik Değeri (Risk Sınırı)")
        plt.fill_between(range(len(probs)), probs, 0.5, where=(np.array(probs) > 0.5), color="red", alpha=0.3, label="Riskli Bölge")
        plt.title("Yapay Zeka Kalp Ritmi Takip Analizi", fontsize=14)
        plt.xlabel("Zaman (Pencere İndeksi)")
        plt.ylabel("Risk Skoru (0.0 - 1.0)")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(REPORT_PNG)
        plt.close()
    except Exception:
        pass

    return {
        "label": label,
        "score": float(score),
        "confidence": confidence,
        "note": f"mean_rr={mean_rr:.1f}ms, outlier_frac={out_frac:.2f}",
        "total_windows": total_windows,
        "risky_windows": risky_windows,
        "risk_pct": risk_pct,
        "report_png": "ai_saglik_raporu.png" if os.path.isfile(REPORT_PNG) else None,
    }


def score_latest_session_async(rr_ms: list[float]) -> None:
    global _score_thread
    with _score_lock:
        if _score_state.get("running"):
            return
        _score_state["running"] = True
        _score_state["status"] = "running"
        _score_state["message"] = "scoring"
        _score_state["started_at"] = time.time()
        _score_state["finished_at"] = None
        _score_state["result"] = None

    def runner():
        try:
            res = _score_rr(rr_ms)
            with _score_lock:
                _score_state["running"] = False
                _score_state["status"] = "done"
                _score_state["message"] = "ok"
                _score_state["finished_at"] = time.time()
                _score_state["result"] = res
        except Exception as e:
            with _score_lock:
                _score_state["running"] = False
                _score_state["status"] = "error"
                _score_state["message"] = str(e)
                _score_state["finished_at"] = time.time()
                _score_state["result"] = None

    _score_thread = threading.Thread(target=runner, daemon=True)
    _score_thread.start()


def get_score_status() -> dict:
    with _score_lock:
        st = dict(_score_state)
    return st
