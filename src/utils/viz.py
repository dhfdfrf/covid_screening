import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def overlay_and_save(img: np.ndarray, mask: np.ndarray, out_path: Path, alpha: float = 0.35) -> None:
    """
    img: (H,W) float [0,1] or any range
    mask: (H,W) {0,1}
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = img.astype(np.float32)
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    m = (mask > 0).astype(np.float32)

    rgb = np.stack([x, x, x], axis=-1)
    red = np.zeros_like(rgb)
    red[..., 0] = 1.0
    out = rgb * (1 - alpha * m[..., None]) + red * (alpha * m[..., None])

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(out)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)