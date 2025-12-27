import numpy as np

from scipy.signal import butter, filtfilt, find_peaks, detrend


class PanTompkins:
    """
    Pan-Tompkins QRS detection algorithm adapted for PPG IR signal.
    Works with 100 Hz sampling rate.
    """

    def __init__(self, fs: float = 100.0):
        self.fs = fs
        self.reset()

    def reset(self):
        self._last = {
            "detrended": None,
            "filtered": None,
            "derivative": None,
            "squared": None,
            "integrated": None,
            "qrs_peaks": None,
            "rr_intervals_ms": None,
            "bpm": None,
        }

    def bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        nyq = 0.5 * self.fs
        b, a = butter(2, [0.7 / nyq, 4.0 / nyq], btype="band")
        return filtfilt(b, a, signal)

    def derivative(self, signal: np.ndarray) -> np.ndarray:
        return np.diff(signal, prepend=signal[0])

    def squaring(self, signal: np.ndarray) -> np.ndarray:
        """Squaring to make all points positive and emphasize larger deviations."""
        return signal * signal

    def moving_window_integration(self, signal: np.ndarray, window_ms: float = 150.0) -> np.ndarray:
        """Moving window integration."""
        win_len = int(self.fs * window_ms / 1000.0)
        if win_len < 1:
            win_len = 1
        integrated = np.convolve(signal, np.ones(win_len) / win_len, mode='same')
        return integrated

    def detect_peaks(self, integrated: np.ndarray) -> tuple[np.ndarray, float]:
        threshold = float(np.mean(integrated) * 0.7)
        min_distance = int(round(0.5 * self.fs))
        peaks, _ = find_peaks(integrated, height=threshold, distance=min_distance)
        return peaks.astype(int), threshold

    def calculate_rr_and_bpm(self, qrs_peaks: np.ndarray) -> tuple[list[float], float | None]:
        """Calculate RR intervals and BPM from QRS peaks."""
        if len(qrs_peaks) < 2:
            return [], None

        rr_samples = np.diff(qrs_peaks)
        rr_seconds = rr_samples / self.fs
        rr_ms = rr_seconds * 1000.0

        # Filter RR intervals (physiological limits: 450-1300 ms)
        rr_ms = rr_ms[(rr_ms > 450) & (rr_ms < 1300)]

        if len(rr_ms) == 0:
            return [], None

        mean_rr_ms = float(np.mean(rr_ms))
        bpm = 60000.0 / mean_rr_ms

        return rr_ms.tolist(), bpm

    def process(self, signal: np.ndarray) -> dict:
        """
        Main processing pipeline.
        Returns dict with all intermediate signals and results.
        """
        if len(signal) == 0:
            return {
                "filtered": [],
                "derivative": [],
                "squared": [],
                "integrated": [],
                "peaks": [],
                "qrs_peaks": [],
                "rr_intervals_ms": [],
                "bpm": None,
            }

        x = signal.astype(float)
        trim = int(round(5.0 * self.fs))
        if len(x) > trim + 10:
            x = x[trim:]

        signal_detrended = detrend(x)
        filtered = self.bandpass_filter(signal_detrended)
        derivative = self.derivative(filtered)
        squared = self.squaring(derivative)
        integrated = self.moving_window_integration(squared, window_ms=150.0)

        qrs_peaks, threshold = self.detect_peaks(integrated)
        rr_intervals_ms, bpm = self.calculate_rr_and_bpm(qrs_peaks)

        self._last = {
            "detrended": signal_detrended,
            "filtered": filtered,
            "derivative": derivative,
            "squared": squared,
            "integrated": integrated,
            "qrs_peaks": qrs_peaks,
            "rr_intervals_ms": rr_intervals_ms,
            "bpm": bpm,
            "threshold": threshold,
            "trim": trim,
        }

        return {
            "detrended": signal_detrended.tolist(),
            "filtered": filtered.tolist(),
            "derivative": derivative.tolist(),
            "squared": squared.tolist(),
            "integrated": integrated.tolist(),
            "qrs_peaks": qrs_peaks.tolist(),
            "rr_intervals_ms": rr_intervals_ms,
            "bpm": bpm,
            "threshold": threshold,
            "trim": trim,
        }
