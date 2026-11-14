"""Read thermal, RGB, event and IMU streams from an M2DGR rosbag (ROS1).

This is the only file that knows about the dataset. When data from the real device arrives, write a
loader with the same four functions and leave the rest of the code alone.

Timestamps are sensor header stamps in seconds (float64), not the time the bag recorded them.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore

# Topic names per dataset, from each dataset's documentation.
DATASETS = {
    "m2dgr": {
        "thermal": "/thermal_image_raw",                # PLUG 617, 640x512, 25 Hz
        "rgb": "/camera/color/image_raw/compressed",    # RealSense D435i, 640x480, 15 Hz
        "events": "/dvs/events",                        # DVXplorer, 640x480, packets at ~15 Hz
        "imu": "/handsfree/imu",                        # 150 Hz
    },
    "vivid": {  # ViViD++ handheld
        "thermal": "/thermal/image_raw",                # FLIR A65, 640x512, 20 Hz
        "rgb": "/camera/image_color",                   # Flea3, 1280x1024
        "events": "/dvs/events",                        # DAVIS 240C, 240x180
        "imu": "/imu/data",                             # Microstrain 3DM-GX5
    },
}
TOPICS = DATASETS["m2dgr"]

TYPESTORE = get_typestore(Stores.ROS1_NOETIC)

# dvs_msgs/Event as serialized by ROS1: uint16 x, uint16 y, time ts (sec, nsec), bool polarity.
# Parsed straight from bytes, because deserializing millions of events one by one is far too slow.
EVENT_DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("sec", "<u4"), ("nsec", "<u4"), ("p", "u1")])


@dataclass
class Events:
    t: np.ndarray  # float64 seconds
    x: np.ndarray  # uint16
    y: np.ndarray  # uint16
    p: np.ndarray  # uint8, 1 = brighter, 0 = darker
    width: int
    height: int


def _stamp(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def _connections(reader, topic):
    conns = [c for c in reader.connections if c.topic == topic]
    if not conns:
        raise KeyError(f"topic {topic} not in bag; run e0_inspect.py --summary-only to list topics")
    return conns


def summarize(path):
    """One row per topic: name, message type, count, rate over the bag's duration."""
    with Reader(path) as reader:
        dur = max((reader.end_time - reader.start_time) * 1e-9, 1e-9)
        rows = [(c.topic, c.msgtype, c.msgcount, c.msgcount / dur) for c in reader.connections]
        return sorted(rows), dur


def _decode_image(msg, msgtype):
    if msgtype.endswith("CompressedImage"):
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_UNCHANGED)
        return img[:, :, ::-1] if img.ndim == 3 else img  # BGR -> RGB
    enc = msg.encoding.lower()
    if enc.startswith("bayer_"):  # raw sensor pattern; OpenCV names the pattern one pixel shifted
        code = {"rggb8": cv2.COLOR_BayerBG2RGB, "bggr8": cv2.COLOR_BayerRG2RGB,
                "gbrg8": cv2.COLOR_BayerGR2RGB, "grbg8": cv2.COLOR_BayerGB2RGB}[enc[6:]]
        return cv2.cvtColor(np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width), code)
    dtype = np.uint16 if "16" in enc else np.uint8
    if enc[-2:-1] == "c":  # OpenCV-style names like 8UC1, 8UC3, 16UC1: last digit is the channel count
        chans = int(enc[-1])
    else:
        chans = 3 if enc in ("rgb8", "bgr8") else 1
    img = np.frombuffer(msg.data, dtype).reshape(msg.height, msg.width, chans).squeeze()
    if msg.is_bigendian and dtype == np.uint16:
        img = img.byteswap()
    if img.ndim == 3 and enc not in ("rgb8", "bgr8") and (img == img[:, :, :1]).all():
        return img[:, :, 0]  # grey stored as three identical channels (M2DGR thermal: 8UC3)
    return img[:, :, ::-1] if enc == "bgr8" else img


def iter_images(path, topic, keep=None):
    """Yield (index, time, image) for each frame on `topic`; image is None unless index is in `keep`.

    keep=None decodes every frame. Streaming keeps memory flat on long bags.
    """
    with Reader(path) as reader:
        for i, (conn, _, raw) in enumerate(reader.messages(connections=_connections(reader, topic))):
            msg = TYPESTORE.deserialize_ros1(raw, conn.msgtype)
            img = _decode_image(msg, conn.msgtype) if keep is None or i in keep else None
            yield i, _stamp(msg.header), img


def read_images(path, topic, keep=None):
    """Timestamps of every frame on `topic`, and {index: image} for the frames in `keep`."""
    times, frames = [], {}
    for i, t, img in iter_images(path, topic, keep):
        times.append(t)
        if img is not None:
            frames[i] = img
    return np.array(times), frames


def read_imu(path, topic=TOPICS["imu"]):
    """Returns (times, gyro, accel): gyro in rad/s and accel in m/s^2, each an (N, 3) array."""
    times, gyro, accel = [], [], []
    with Reader(path) as reader:
        for conn, _, raw in reader.messages(connections=_connections(reader, topic)):
            msg = TYPESTORE.deserialize_ros1(raw, conn.msgtype)
            times.append(_stamp(msg.header))
            w, a = msg.angular_velocity, msg.linear_acceleration
            gyro.append((w.x, w.y, w.z))
            accel.append((a.x, a.y, a.z))
    return np.array(times), np.array(gyro), np.array(accel)


def _parse_event_array(raw):
    # Header: seq (u4), stamp (u4 sec, u4 nsec), frame_id (u4 length + bytes); then height, width (u4)
    off = 12
    off += 4 + int(np.frombuffer(raw, "<u4", 1, off)[0])
    height, width, n = np.frombuffer(raw, "<u4", 3, off)
    ev = np.frombuffer(raw, EVENT_DTYPE, int(n), off + 12)
    return ev, int(width), int(height)


def iter_event_packets(path, topic=TOPICS["events"]):
    """Yield one Events per message, in bag order. Memory stays at one packet."""
    with Reader(path) as reader:
        for conn, _, raw in reader.messages(connections=_connections(reader, topic)):
            ev, width, height = _parse_event_array(bytes(raw))
            if len(ev):
                yield Events(ev["sec"] + ev["nsec"] * 1e-9, ev["x"], ev["y"], ev["p"], width, height)


def read_events(path, windows, topic=TOPICS["events"]):
    """Events that fall inside any of the (start, end) windows, in seconds.

    Only windows are kept, so a long bag with billions of events fits in memory.
    """
    windows = np.asarray(sorted(windows), float)
    keep, width, height = [], 0, 0
    with Reader(path) as reader:
        for conn, _, raw in reader.messages(connections=_connections(reader, topic)):
            ev, width, height = _parse_event_array(bytes(raw))
            if len(ev) == 0:
                continue
            t = ev["sec"] + ev["nsec"] * 1e-9
            # skip packets that overlap no window
            i = np.searchsorted(windows[:, 1], t[0])
            if i == len(windows) or windows[i, 0] > t[-1]:
                continue
            mask = np.zeros(len(t), bool)
            for start, end in windows[i:]:
                if start > t[-1]:
                    break
                mask |= (t >= start) & (t < end)
            if mask.any():
                keep.append((t[mask], ev[mask]))
    if not keep:
        return Events(np.empty(0), np.empty(0, np.uint16), np.empty(0, np.uint16), np.empty(0, np.uint8),
                      width, height)
    t = np.concatenate([k[0] for k in keep])
    ev = np.concatenate([k[1] for k in keep])
    return Events(t, ev["x"].copy(), ev["y"].copy(), ev["p"].copy(), width, height)
