"""E1: every way of turning events into a network input, in one place.

Two halves:

1. `window_inputs()` runs once per sample inside build_e1_cache.py, on the raw events from the bag
   (exact timestamps). It stores the inputs that need exact timing: voxel grid, time surface,
   fixed-number-of-events images, the spiking network's time steps and the graph network's events.
2. The `*_input()` functions run during training. They turn what is stored (in the new e1/ cache, or
   the old 40 ms count bins for count images) into float arrays for the network.

Every window ends at the same moment the old 40 ms bin before each sample ends, so all inputs for a
sample see events up to exactly the same time.

Events use the binned 320 x 240 grid (the GenX320-like size the old cache uses); polarity
0 = got darker, 1 = got brighter, as in bag_io.

Two kinds of settings:
- "ours": every input sees the same 240 ms (6 x 40 ms), so inputs differ only in form;
- "paper": the settings of the paper or official code each input comes from (RVT, E2VID / Li et al.,
  Zubic et al. / HOTS, E2VID's window size), checked against their code where it is public. Where a
  paper leaves a detail out, the docstring says what was assumed.
"""
import numpy as np
import torch

EVENT_SCALE = float(np.log1p(255.0))  # log scale shared by all count inputs: 255 events -> 1.0

WINDOW_S = 0.240                      # window for stacked, voxel, spiking and graph inputs (6 x 40 ms)
VOXEL_BINS = 6                        # voxel slices over WINDOW_S (40 ms apart, like the stacked images)
SNN_STEPS = 12                        # spiking network time steps over WINDOW_S (20 ms each)
TS_MAX_MS = 1000                      # time surface: age stored in whole ms, 1000 = nothing in the last 1 s
TAU_MS = 50.0                         # time surface decay constant ("ours"); 25 and 100 also tested
FIXED_N = (26_880, 50_000, 200_000, 800_000)  # events per fixed-number window; 26,880 = E2VID's rule
                                      # N = 0.35 x width x height (rpg_e2vid run_reconstruction.py)
# paper settings
RVT_WINDOW_S, RVT_BINS, RVT_CUTOFF = 0.050, 10, 10   # RVT: 50 ms, 10 bins, counts clipped at 10
VOXEL5_BINS = 5                       # E2VID / Li et al.: 5-bin voxel grid, normalised (see voxel_norm)
ZUBIC_TAU_MS, ZUBIC_SAMPLES = 5.0, 6  # Zubic et al.: HOTS time surface, tau 5 ms, 6 samples x 2 polarities
ZUBIC_MAX = 255                       # stored in 0.1 ms steps: 255 = nothing in the last 25.5 ms
# TEVIO-style adaptive time surface (TEVIO eq. 4). The paper gives the formula but no values, and its
# threshold uses depth we don't have; these are our choices:
TEVIO_TAU_UP, TEVIO_TAU_LOW = 100.0, 10.0   # upper / lower bound of the per-pixel decay (ms)
TEVIO_TAU_REF = 50.0                  # decay of the reference surface the threshold is applied to (ms)
TEVIO_DELTA = float(np.exp(-1.0))     # neighbour counts if its reference value > this (changed within 50 ms)
TEVIO_NEIGHBOURS = 5                  # 5 x 5 neighbourhood
TEVIO_SMOOTH = 3                      # 3 x 3 blur, then 3 x 3 median on the decay map
# EventMG-style network (Wu et al., NeurIPS 2025): N, delta_st and n_min are the paper's defaults
# (Table 6); c, k and the coordinate scaling are not given, so these are ours
EVENTMG_N = 20_000                    # most recent N events per window
EVENTMG_DST = 0.05                    # edge if sqrt(|ds|^2 + c^2 dt^2) <= delta_st
EVENTMG_C = 1.0                       # time scale c (ours)
EVENTMG_K = 16                        # at most k neighbours (ours)
EVENTMG_NMIN = 100                    # components smaller than this are ignored
FIXED_MAX_S = 1.0                     # a fixed-number window never reaches back further than this
GRAPH_STORE = 10_000                  # events stored per sample: AEGNN's n_samples for N-Cars (random,
                                      # without replacement); the generic graph network draws GRAPH_POINTS of them
GRAPH_POINTS = 2048                   # events the graph network sees per sample (redrawn every epoch)
GRAPH_T_PX = 100.0                    # graph distance: 10 ms apart counts as 1 pixel apart
LOOKBACK_S = max(FIXED_MAX_S, WINDOW_S, TS_MAX_MS / 1e3)


# ---------------------------------------------------------------- 1. building (raw events -> stored)

def window_inputs(t, x, y, p, end, h, w, rng):
    """Everything E1 stores for one sample. Events: end - LOOKBACK_S <= t < end, any order.

    t float64 seconds; x, y on the h x w grid; p 0/1. Returns a dict of arrays (see build_e1_cache)."""
    order = np.argsort(t, kind="stable")
    t, x, y, p = t[order], x[order], y[order], p[order].astype(np.int64)
    age = end - t                                    # seconds before the window end, > 0, ascending t
    hw = h * w
    flat = p * hw + y.astype(np.int64) * w + x       # index into a (2, h, w) image
    out = {}

    # fixed number of events: the most recent N, however long that takes (capped at FIXED_MAX_S)
    imgs, secs, got = [], [], []
    for n in FIXED_N:
        k = min(n, len(t))
        c = np.bincount(flat[len(t) - k:], minlength=2 * hw)
        imgs.append(np.minimum(c, 65535).astype(np.uint16).reshape(2, h, w))  # 1 s windows can pass 255
        secs.append(age[len(t) - k] if k else 0.0)
        got.append(k)
    out["fixed_n"] = np.stack(imgs)                  # (len(FIXED_N), 2, h, w) uint16 counts
    out["fixed_n_window_s"] = np.array(secs, np.float32)
    out["fixed_n_events"] = np.array(got, np.int64)  # < N when FIXED_MAX_S ran out first

    # time surface: per pixel and polarity, ms since its most recent event (exact, floored to 1 ms)
    out["ts_age_ms"] = last_age(flat, age, TS_MAX_MS / 1e3, 1e3, TS_MAX_MS, np.uint16, hw).reshape(2, h, w)

    # Zubic et al.'s time surface: the same, sampled at 6 equally spaced moments (the ends of the six
    # 40 ms bins; the paper does not say which moments), in 0.1 ms steps for its 5 ms decay
    zs = []
    for k in range(ZUBIC_SAMPLES):
        shift = (ZUBIC_SAMPLES - 1 - k) * WINDOW_S / ZUBIC_SAMPLES     # how long before the window end
        before = age > shift
        zs.append(last_age(flat[before], age[before] - shift, ZUBIC_MAX / 1e4, 1e4, ZUBIC_MAX, np.uint8, hw))
    out["ts_zubic"] = np.stack(zs).reshape(ZUBIC_SAMPLES, 2, h, w)

    # the shared WINDOW_S window
    m = age <= WINDOW_S
    tw, fw, pw = (WINDOW_S - age[m]) / WINDOW_S, flat[m], p[m]   # tw: 0 = window start, ->1 = end
    pix = fw % hw

    # voxel grids: 6 slices ("ours", same span as the stacked images) and 5 slices (E2VID / Li et al.)
    sign = 2.0 * pw - 1.0
    out["voxel"] = voxel(tw, pix, sign, VOXEL_BINS, hw).reshape(VOXEL_BINS, h, w).astype(np.float16)
    out["voxel5"] = voxel(tw, pix, sign, VOXEL5_BINS, hw).reshape(VOXEL5_BINS, h, w).astype(np.float16)

    # RVT's stacked histogram: last 50 ms in 10 bins of 5 ms, per polarity, counts clipped at 10
    r = age <= RVT_WINDOW_S
    rb = np.minimum(((RVT_WINDOW_S - age[r]) / RVT_WINDOW_S * RVT_BINS).astype(np.int64), RVT_BINS - 1)
    c = np.bincount(rb * 2 * hw + flat[r], minlength=RVT_BINS * 2 * hw)
    out["rvt"] = np.minimum(c, RVT_CUTOFF).astype(np.uint8).reshape(RVT_BINS * 2, h, w)

    # spiking network: counts in SNN_STEPS consecutive steps (20 ms each), per polarity
    step = np.minimum((tw * SNN_STEPS).astype(np.int64), SNN_STEPS - 1)
    c = np.bincount(step * 2 * hw + fw, minlength=SNN_STEPS * 2 * hw)
    out["snn"] = np.minimum(c, 255).astype(np.uint8).reshape(SNN_STEPS, 2, h, w)

    # graph network: a uniform random sample of the window's events, with exact times
    idx = np.flatnonzero(m)
    if len(idx) > GRAPH_STORE:
        idx = np.sort(rng.choice(idx, GRAPH_STORE, replace=False))
    g = np.zeros(GRAPH_STORE, [("x", "<u2"), ("y", "<u2"), ("age_us", "<u4"), ("p", "u1")])
    g["x"][:len(idx)], g["y"][:len(idx)] = x[idx], y[idx]
    g["age_us"][:len(idx)] = np.round(age[idx] * 1e6).astype(np.uint32)
    g["p"][:len(idx)] = p[idx]
    out["graph"], out["graph_n"] = g, len(idx)

    # EventMG-style input: the most recent EVENTMG_N events, exact times
    k = min(EVENTMG_N, len(t))
    e = np.zeros(EVENTMG_N, g.dtype)
    sl = slice(len(t) - k, len(t))
    e["x"][:k], e["y"][:k], e["p"][:k] = x[sl], y[sl], p[sl]
    e["age_us"][:k] = np.round(age[sl] * 1e6).astype(np.uint32)
    out["eventmg"], out["eventmg_n"] = e, k
    out["window_events"] = int(m.sum())
    return out


def last_age(flat, age, max_s, unit, none, dtype, hw):
    """Per (polarity, pixel): age of its most recent event, in `unit`s per second, if younger than
    max_s; `none` otherwise. flat/age must be in ascending time order."""
    m = age < max_s
    f_rev, a_rev = flat[m][::-1], age[m][::-1]       # newest first
    out = np.full(2 * hw, none, dtype)
    u, first = np.unique(f_rev, return_index=True)
    out[u] = np.minimum(np.floor(a_rev[first] * unit), none - 1).astype(dtype)
    return out


def voxel(tw, pix, sign, bins, hw):
    """Voxel grid (Zhu et al.; rpg_e2vid events_to_voxel_grid): time scaled to [0, bins-1], each event
    adds +1 / -1 split between its two nearest slices by its distance to each.

    tw: event time within the window, 0 = start, 1 = end. E2VID scales by the first and last event
    of the window; with 0.2-2.5 M events per window that is the window itself."""
    tn = tw * (bins - 1)
    lo = np.minimum(np.floor(tn).astype(np.int64), bins - 2)
    wr = tn - lo
    return (np.bincount(lo * hw + pix, sign * (1 - wr), bins * hw)
            + np.bincount((lo + 1) * hw + pix, sign * wr, bins * hw))


# ---------------------------------------------------------------- 2. training (stored -> network input)
# Selecting/sampling runs in numpy on the CPU (cheap); the element-wise maths runs in torch on the
# training device, so large inputs (12 spiking steps of 320 x 240) are not expanded to float on the CPU.

def count_window(hist, n_bins):
    """Count image from the old cache: darker / brighter events in the last n_bins x 40 ms. (n, 2, H, W) uint16.

    hist: (n, B, 2, H, W) uint8 40 ms bins, oldest first. Exact for any multiple of 40 ms."""
    return hist[:, -n_bins:].sum(1, dtype=np.uint16)


def stack_bins(hist, n_bins=6):
    """Stacked count images: the last n_bins 40 ms images side by side, oldest first. (n, 2*n_bins, H, W) uint8."""
    h = hist[:, -n_bins:]
    return np.ascontiguousarray(h.reshape(len(h), -1, *h.shape[-2:]))


def graph_points(graph, graph_n, rng, n_points=GRAPH_POINTS):
    """Draw n_points of the stored events per sample. (n, n_points, 4) float32: x, y (pixels), age (s), polarity.

    Redrawn every epoch during training. Samples with fewer stored events repeat some; samples with
    no events get all-zero points."""
    out = np.zeros((len(graph), n_points, 4), np.float32)
    for i in range(len(graph)):
        k = int(graph_n[i])
        if k == 0:
            continue
        g = graph[i][rng.choice(k, n_points, replace=k < n_points)]
        out[i] = np.stack([g["x"], g["y"], g["age_us"] / 1e6, g["p"]], 1)
    return out


def aegnn_points(graph, graph_n):
    """Input for AEGNN: all stored events (up to 10,000, AEGNN's N-Cars count), padded.
    (n, GRAPH_STORE, 5) float32: x, y (pixels), age (s), polarity (0/1, as AEGNN's Gen1 loader), valid."""
    out = np.zeros((len(graph), GRAPH_STORE, 5), np.float32)
    for i in range(len(graph)):
        k = int(graph_n[i])
        g = graph[i][:k]
        out[i, :k] = np.stack([g["x"], g["y"], g["age_us"] / 1e6, g["p"], np.ones(k)], 1)
    return out


def eventmg_graph(ev, n, w=320, h=240):
    """EventMG-style graph for one sample (numpy, run once per sample and kept).

    Nodes: the stored events. Coordinates for the graph: x / w, y / w (aspect kept), time scaled to 0-1
    over the window (0 = oldest); edge if distance <= EVENTMG_DST, at most EVENTMG_K neighbours
    (EventMG eq. 1). Components: connected components with >= EVENTMG_NMIN events (eq. 6), others -1.
    Returns features (N, 4) [x/w, y/h, t, polarity], neighbours (N, K) (padded with the node itself),
    component id (N,), valid (N,)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    N, K = EVENTMG_N, EVENTMG_K
    e = ev[:n]
    age = e["age_us"].astype(np.float64)
    t = (age.max() - age) / max(age.max() - age.min(), 1.0)
    coords = np.stack([e["x"] / w, e["y"] / w, EVENTMG_C * t], 1)
    feat = np.zeros((N, 4), np.float32)
    feat[:n] = np.stack([e["x"] / w, e["y"] / h, t, e["p"]], 1)
    nbr = np.tile(np.arange(N, dtype=np.int32)[:, None], (1, K))
    comp = np.full(N, -1, np.int32)
    if n > 1:
        d, j = cKDTree(coords).query(coords, k=K + 1, distance_upper_bound=EVENTMG_DST)
        d, j = d[:, 1:], j[:, 1:]                                  # drop the node itself
        ok = np.isfinite(d)
        nbr[:n] = np.where(ok, j, np.arange(n)[:, None])
        rows = np.repeat(np.arange(n), K)[ok.ravel()]
        _, lab = connected_components(coo_matrix((np.ones(len(rows)), (rows, j.ravel()[ok.ravel()])), shape=(n, n)),
                                      directed=False)
        size = np.bincount(lab)
        keep = size[lab] >= EVENTMG_NMIN
        _, comp_ids = np.unique(lab[keep], return_inverse=True)
        comp[:n][keep] = comp_ids
    valid = np.zeros(N, np.float32)
    valid[:n] = 1
    return feat, nbr, comp, valid


def flip_points(pts, width):
    """Mirror left/right for the graph input (training augmentation)."""
    pts = pts.copy()
    pts[..., 0] = width - 1 - pts[..., 0]
    return pts


def log_counts(c):
    """torch: counts -> log scale (255 events -> 1.0), so a busy pixel does not swamp a quiet one."""
    return torch.log1p(c.float()) / EVENT_SCALE


def signed_log(v):
    """torch: voxel values (brighter minus darker, can be negative) -> signed log scale."""
    v = v.float()
    return torch.sign(v) * torch.log1p(v.abs()) / EVENT_SCALE


def time_surface(age_ms, tau=TAU_MS):
    """torch: stored ages -> exp(-age / tau) (TEVIO eq. 2, fixed tau); 1 = changed just now, 0 = quiet.

    Ages are floored to whole ms when stored, so the middle of that ms (+0.5) is used."""
    a = age_ms.float()  # (uint16 tensors support few ops, so compare as float)
    return torch.where(a < TS_MAX_MS, torch.exp(-(a + 0.5) / tau), torch.zeros_like(a))


def time_surface_tevio(age_ms):
    """torch: TEVIO-style adaptive time surface (Gong et al., TEVIO, eqs. 2 and 4), one channel.

    Per pixel u, with ages of the most recent event (polarities combined, as in TEVIO's surface):
      reference surface T_ref = exp(-age / TEVIO_TAU_REF);
      qualified neighbours: pixels in the 5 x 5 neighbourhood with T_ref > TEVIO_DELTA;
      tau(u) = max(TEVIO_TAU_UP - mean age of the qualified neighbours, TEVIO_TAU_LOW)
               (TEVIO_TAU_UP when none qualify);
      tau map smoothed with a blur and a median filter ("refined using blur and median blur filters");
      output exp(-age(u) / tau(u)).
    TEVIO's threshold delta(u) depends on aligned depth and temperature; the paper gives no formula and
    we have no depth, so delta is a constant here. All numeric settings are ours (see constants)."""
    a = age_ms.float().amin(1, keepdim=True)                       # (n, 1, H, W), ms
    seen = a < TS_MAX_MS
    a = torch.where(seen, a + 0.5, torch.zeros_like(a))
    q = (seen & (torch.exp(-a / TEVIO_TAU_REF) > TEVIO_DELTA)).float()
    k = torch.ones(1, 1, TEVIO_NEIGHBOURS, TEVIO_NEIGHBOURS, device=a.device)
    k[..., TEVIO_NEIGHBOURS // 2, TEVIO_NEIGHBOURS // 2] = 0          # neighbours only
    pad = TEVIO_NEIGHBOURS // 2
    cnt = torch.nn.functional.conv2d(q, k, padding=pad)
    mean_age = torch.nn.functional.conv2d(a * q, k, padding=pad) / cnt.clamp(min=1)
    tau = torch.where(cnt > 0, (TEVIO_TAU_UP - mean_age).clamp(min=TEVIO_TAU_LOW),
                      torch.full_like(a, TEVIO_TAU_UP))
    s = TEVIO_SMOOTH
    tau = torch.nn.functional.avg_pool2d(tau, s, 1, s // 2, count_include_pad=False)
    tau = torch.nn.functional.pad(tau, (s // 2,) * 4, mode="replicate")
    tau = tau.unfold(2, s, 1).unfold(3, s, 1).flatten(-2).median(-1).values
    return torch.where(seen, torch.exp(-a / tau), torch.zeros_like(a))


def time_surface_zubic(age_01ms):
    """torch: Zubic et al.'s 12-channel time surface, exp(-age / 5 ms) at 6 moments x 2 polarities.
    (n, 6, 2, H, W) uint8 in 0.1 ms steps -> (n, 12, H, W)."""
    a = age_01ms.float() / 10 + 0.05
    v = torch.where(age_01ms < ZUBIC_MAX, torch.exp(-a / ZUBIC_TAU_MS), torch.zeros_like(a))
    return v.flatten(1, 2)


def voxel_norm(v):
    """torch: E2VID's input normalisation (rpg_e2vid EventPreprocessor): per sample, the non-zero
    values get mean 0 and std 1 (statistics over all slices and pixels); zeros stay zero."""
    v = v.float()
    nz = (v != 0).float()
    n = nz.flatten(1).sum(1).clamp(min=1).view(-1, 1, 1, 1)
    mean = v.flatten(1).sum(1).view(-1, 1, 1, 1) / n
    std = torch.sqrt(((v ** 2).flatten(1).sum(1).view(-1, 1, 1, 1) / n - mean ** 2).clamp(min=1e-12))
    return nz * (v - mean) / std


def rvt_input(c):
    """torch: RVT stacked histogram (counts 0-10, uint8). RVT's code stores raw clipped counts; the
    scaling inside their model was not checked, so counts are divided by the cutoff to give 0-1."""
    return c.float() / RVT_CUTOFF


def raw_counts(c):
    """torch: counts as they are, no scaling (the count-image papers' plain histogram)."""
    return c.float()
