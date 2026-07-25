"""Pattern comparison: delta-ambient across all walls and classes.

The question this answers: what SPATIAL PATTERN does each fault class
create, relative to normal operation? Ambient is subtracted, so NORMAL
renders near-zero everywhere and each fault's shape stands alone.
One shared scale (0..+8 C over ambient) across all 16 panels.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataset_viewer import run_case, CASES
from simulator import WALLS, ROWS, COLS

DMAX = 8.0  # delta-ambient scale ceiling

def main(frame=110, out="pattern_compare.png"):
    fig, axes = plt.subplots(len(CASES), len(WALLS),
                             figsize=(15, 10), dpi=170)
    for ri, kind in enumerate(CASES):
        mats, amb, score = run_case(kind, steps=frame + 1)[frame]
        for ci, w in enumerate(WALLS):
            ax = axes[ri][ci]
            d = np.nan_to_num(np.asarray(mats[w], float) - amb, nan=0.0)
            im = ax.imshow(d, cmap="inferno", vmin=0, vmax=DMAX)
            ax.set_xticks(np.arange(-.5, COLS), minor=True)
            ax.set_yticks(np.arange(-.5, ROWS), minor=True)
            ax.grid(which="minor", color="black", linewidth=1.2)
            ax.tick_params(which="both", length=0,
                           labelbottom=False, labelleft=False)
            for r in range(ROWS):
                for c in range(COLS):
                    v = d[r, c]
                    ax.text(c, r, f"{v:+.1f}", ha="center", va="center",
                            fontsize=8,
                            color="white" if v > DMAX * 0.45 else
                                  ("#cccccc" if v < 0.5 else "black"))
            if ri == 0:
                ax.set_title(f"Wall {w}", fontsize=12)
        axes[ri][0].set_ylabel(f"{kind.upper()}\nscore {score:.2f}",
                               fontsize=11, rotation=0, ha="right",
                               va="center", labelpad=50)
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label("Δ ambient (°C)")
    fig.suptitle("Spatial patterns per class — ambient removed, shared scale",
                 fontsize=14)
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)

if __name__ == "__main__":
    main()
