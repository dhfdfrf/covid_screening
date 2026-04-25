from pathlib import Path
import numpy as np

def save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)

def load_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}
