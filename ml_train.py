"""ThermalGuard ML — synthetic dataset -> features -> Isolation Forest.

Generates normal and faulty episodes from the physics simulator, extracts
the spec's feature set (Section 9) per frame, trains an Isolation Forest on
NORMAL data only, and evaluates detection on held-out fault episodes.
Saves model + scaler to model.joblib for the live pipeline.

The model consumes temperature MATRICES (4x6 per wall), never rendered
images — rendering is for humans, matrices are for the ML.

Usage:
  python ml_train.py                # generate, train, evaluate, save
  python ml_train.py --episodes 40  # bigger dataset

Runtime scoring (import from live code):
  from ml_train import AnomalyScorer
  scorer = AnomalyScorer("model.joblib")
  score = scorer.score(mats, amb)   # 0..1, higher = more anomalous
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from simulator import ThermalSim, WALLS, ROWS, COLS

MODEL_PATH = Path(__file__).parent / "model.joblib"
WINDOW = 5  # frames of history for rise-rate features (10s at 2s cycle)


# ------------------------------------------------------------------ features
def neighbor_max_delta(m):
    """Largest |cell - 4-neighbour| across the wall."""
    best = 0.0
    for dr, dc in [(0, 1), (1, 0)]:
        a = m[max(0, -dr):ROWS - dr or ROWS, max(0, -dc):COLS - dc or COLS]
        b = m[dr:, dc:]
        d = np.abs(a - b)
        if d.size:
            best = max(best, float(np.nanmax(d)))
    return best


def frame_features(mats, amb, history):
    """Feature vector per spec Section 9 (absolute, ambient-adjusted,
    spatial, temporal). `history` = deque of previous all-wall stacks."""
    stack = np.stack([mats[w] for w in WALLS])          # (4, R, C)
    f = []
    # absolute
    f += [np.nanmax(stack), np.nanmin(stack), np.nanmean(stack),
          np.nanstd(stack), np.nanmax(stack) - np.nanmin(stack)]
    # ambient-adjusted
    f += [np.nanmax(stack) - amb, np.nanmean(stack) - amb,
          float(np.sum(stack > amb + 2.0))]             # hot-cell count
    # spatial (worst wall)
    f += [max(neighbor_max_delta(mats[w]) for w in WALLS)]
    # hotspot area: cells within 1.5C of max, above amb+2
    mx = np.nanmax(stack)
    f += [float(np.sum((stack > mx - 1.5) & (stack > amb + 2.0)))]
    # temporal
    if history:
        prev = history[0]                               # oldest in window
        f += [float(np.nanmax(stack - prev)),           # max rise over window
              float(np.nanmean(stack - prev))]
    else:
        f += [0.0, 0.0]
    return np.array(f, dtype=float)


FEATURE_NAMES = ["max", "min", "mean", "std", "range",
                 "max_d_amb", "mean_d_amb", "hot_cells",
                 "max_neigh_delta", "hotspot_area",
                 "max_rise_w", "mean_rise_w"]


# ------------------------------------------------------------------ episodes
def run_episode(kind, steps=120, inject_at=40, seed=None):
    """Yield (features, label) per frame. label 1 after fault injection."""
    from collections import deque
    from thermal3d import frame_to_walls

    sim = ThermalSim(seed=seed if seed is not None else np.random.randint(1e6))
    hist = deque(maxlen=WINDOW)
    rng = np.random.default_rng(seed)
    out = []
    for i in range(steps):
        if kind != "normal" and i == inject_at:
            w = rng.choice(WALLS)
            r, c = rng.integers(ROWS), rng.integers(COLS)
            if kind == "hotspot":
                sim.inject_hotspot(w, r, c, power=float(rng.uniform(0.3, 0.9)))
            elif kind == "runaway":
                sim.inject_hotspot(w, r, c, power=0.05, growth=1.06)
            elif kind == "cluster":
                for dr, dc in [(0, 0), (0, 1), (1, 0)]:
                    sim.inject_hotspot(w, min(r + dr, ROWS - 1),
                                       min(c + dc, COLS - 1),
                                       power=float(rng.uniform(0.3, 0.7)))
        frame = sim.step()
        mats, amb = frame_to_walls(frame)
        feats = frame_features(mats, amb, hist)
        label = int(kind != "normal" and i >= inject_at + 3)  # small onset lag
        out.append((feats, label))
        hist.append(np.stack([mats[w] for w in WALLS]))
    return out


def build_dataset(n_episodes):
    kinds = ["normal"] * n_episodes + \
            ["hotspot"] * (n_episodes // 2) + \
            ["runaway"] * (n_episodes // 4) + \
            ["cluster"] * (n_episodes // 4)
    X, y, ep_kind = [], [], []
    for i, k in enumerate(kinds):
        for feats, label in run_episode(k, seed=1000 + i):
            X.append(feats); y.append(label); ep_kind.append(k)
    return np.array(X), np.array(y), np.array(ep_kind)


# ------------------------------------------------------------------ scorer
class AnomalyScorer:
    """Runtime wrapper: raw IF score -> calibrated 0..1 anomaly score."""

    def __init__(self, path=MODEL_PATH):
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.lo, self.hi = bundle["score_range"]
        from collections import deque
        self.history = deque(maxlen=WINDOW)

    def score(self, mats, amb):
        feats = frame_features(mats, amb, self.history)
        self.history.append(np.stack([mats[w] for w in WALLS]))
        raw = -self.model.score_samples(self.scaler.transform([feats]))[0]
        return float(np.clip((raw - self.lo) / (self.hi - self.lo), 0, 1))


# ------------------------------------------------------------------ train
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()

    print(f"Generating dataset ({args.episodes} normal episodes + faults)...")
    X, y, kind = build_dataset(args.episodes)
    print(f"  {len(X)} frames, {y.sum()} fault-labelled")

    scaler = StandardScaler().fit(X[y == 0])
    Xn = scaler.transform(X)
    model = IsolationForest(n_estimators=200, contamination="auto",
                            random_state=42).fit(Xn[y == 0])

    raw = -model.score_samples(Xn)
    lo, hi = np.percentile(raw[y == 0], 1), np.percentile(raw[y == 1], 95)
    score = np.clip((raw - lo) / (hi - lo), 0, 1)

    print("\nScore distribution (0..1, higher = anomalous):")
    print(f"  normal frames: mean {score[y == 0].mean():.3f}  "
          f"p95 {np.percentile(score[y == 0], 95):.3f}")
    for k in ["hotspot", "runaway", "cluster"]:
        m = (kind == k) & (y == 1)
        if m.any():
            print(f"  {k:8s} faults: mean {score[m].mean():.3f}  "
                  f"p5 {np.percentile(score[m], 5):.3f}")

    thr = 0.65  # spec's WATCH threshold
    tpr = (score[y == 1] >= thr).mean()
    fpr = (score[y == 0] >= thr).mean()
    print(f"\nAt WATCH threshold {thr}: detection {tpr:.1%}, "
          f"false alarms {fpr:.2%}")

    joblib.dump({"model": model, "scaler": scaler,
                 "score_range": (lo, hi),
                 "feature_names": FEATURE_NAMES}, MODEL_PATH)
    print(f"saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
