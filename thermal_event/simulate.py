"""Make dataset sensors look like ours: MLX90640 thermal (32x24, 4 Hz) and GenX320-like events.

Every model sees data through these functions, so results carry over when the real device arrives.
"""
import cv2
import numpy as np

THERMAL_SIZE = (32, 24)   # MLX90640, width x height
THERMAL_HZ = 4
EVENT_WIDTH = 320         # GenX320 is 320x320; bin wider sensors down by a whole factor (640x480 -> 320x240)
EVENT_WINDOW = 0.040      # seconds of events per sample (PEDRo-style 40 ms)


def anchor_times(thermal_times, hz=THERMAL_HZ):
    """Indices of thermal frames kept when subsampling to `hz` (nearest frame to each tick)."""
    ticks = np.arange(thermal_times[0], thermal_times[-1], 1.0 / hz)
    idx = np.clip(np.searchsorted(thermal_times, ticks), 1, len(thermal_times) - 1)
    left = thermal_times[idx - 1]
    idx = np.where(np.abs(ticks - left) < np.abs(thermal_times[idx] - ticks), idx - 1, idx)
    return np.unique(idx)


def thermal_to_mlx(img, noise_std=0.01, rng=None):
    """Area-average a thermal frame down to 32x24 and add Gaussian noise.

    Output is float32 scaled to [0, 1] by the frame's dtype range (raw counts, not degrees).
    noise_std is a fraction of that range.
    """
    scale = 65535.0 if img.dtype == np.uint16 else 255.0
    small = cv2.resize(img.astype(np.float32) / scale, THERMAL_SIZE, interpolation=cv2.INTER_AREA)
    if noise_std:
        rng = rng or np.random.default_rng()
        small += rng.normal(0, noise_std, small.shape).astype(np.float32)
    return small


def bin_events(events, factor=None):
    """Coarser pixel grid; timestamps stay exact. Returns (x, y, width, height)."""
    factor = factor or max(1, events.width // EVENT_WIDTH)
    return events.x // factor, events.y // factor, events.width // factor, events.height // factor


def event_frame(events, t_end, window=EVENT_WINDOW, factor=None):
    """Two-channel count image (darker, brighter) of the events in [t_end - window, t_end)."""
    x, y, w, h = bin_events(events, factor)
    m = (events.t >= t_end - window) & (events.t < t_end)
    frame = np.zeros((2, h, w), np.float32)
    np.add.at(frame, (events.p[m].astype(int), y[m].astype(int), x[m].astype(int)), 1)
    return frame
