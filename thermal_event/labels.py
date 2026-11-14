"""Turn YOLO boxes on an RGB frame into per-zone labels (clear / person / furniture / other, near or far).

The cache stores the raw boxes too, so these rules can change later without the bag.
"""
import numpy as np

ZONES = ("left", "center", "right")
CLASSES = ("clear", "person", "furniture", "other")
FURNITURE = {"chair", "couch", "bed", "dining table", "bench", "toilet"}
PRIORITY = ("person", "furniture", "other")  # a zone takes the highest-priority class it overlaps
ZONE_OVERLAP = 0.1  # a box counts for a zone if it covers at least this fraction of the zone's width
NEAR_FRAC = 0.4     # a box is "near" if it is at least this fraction of the image height (no depth topic)
LABEL_CONF = 0.25   # YOLO's default confidence; boxes below it are stored but not used for labels


def group(name):
    return "person" if name == "person" else "furniture" if name in FURNITURE else "other"


def zone_labels(boxes, width, height):
    """boxes: list of (x1, y1, x2, y2, class name). Returns {zone: (label, near)}."""
    out = {}
    for z, zone in enumerate(ZONES):
        z0, z1 = z * width / 3, (z + 1) * width / 3
        hits = [(group(n), (y2 - y1) / height) for x1, y1, x2, y2, n in boxes
                if min(x2, z1) - max(x1, z0) >= ZONE_OVERLAP * (z1 - z0)]
        if not hits:
            out[zone] = ("clear", False)
            continue
        label = min((g for g, _ in hits), key=PRIORITY.index)
        out[zone] = (label, max(h for g, h in hits if g == label) >= NEAR_FRAC)
    return out


def nearest(times, targets):
    """Index of the nearest entry in sorted `times` for each target, and the signed offset."""
    idx = np.clip(np.searchsorted(times, targets), 1, len(times) - 1)
    idx = np.where(np.abs(targets - times[idx - 1]) < np.abs(times[idx] - targets), idx - 1, idx)
    return idx, times[idx] - targets
