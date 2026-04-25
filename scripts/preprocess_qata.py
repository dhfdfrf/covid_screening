import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import hashlib

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


def resize_img(img: np.ndarray, size: int, is_mask: bool) -> np.ndarray:
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    return cv2.resize(img, (size, size), interpolation=interp)


def norm01(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    return x


def is_mask_file(path: Path) -> bool:
    p = str(path).lower()
    return ("ground-truth" in p) or ("mask_" in path.name.lower()) or ("mask" == path.parent.name.lower())


def is_image_file(path: Path) -> bool:
    p = str(path).lower()
    if path.suffix.lower() not in IMG_EXTS:
        return False
    if is_mask_file(path):
        return False
    # 只允许这些目录下的文件作为输入图像
    allowed_keywords = [
        "images",
        "image",
        "covid",
        "control_group",
        "train set",
        "test set",
        "val set",
        "validation set",
    ]
    return any(k in p for k in allowed_keywords)


def find_gt_for_image(img_path: Path) -> Path | None:
    """
    严格规则：
    1) 同一个 split 下，Ground-truths 目录中找同名 mask
    2) mask 文件名前缀是 mask_
    """
    stem = img_path.stem

    # 向上找 split 根目录（Train Set / Test Set / Val Set）
    parents = [img_path.parent, *img_path.parents]
    split_root = None
    for p in parents:
        if p.name.lower() in ["train set", "test set", "val set", "validation set"]:
            split_root = p
            break

    if split_root is None:
        return None

    gt_dir = split_root / "Ground-truths"
    if not gt_dir.exists():
        return None

    for suffix in [".png", ".jpg", ".jpeg"]:
        cand1 = gt_dir / f"mask_{stem}{suffix}"
        cand2 = gt_dir / f"{stem}{suffix}"
        if cand1.exists():
            return cand1
        if cand2.exists():
            return cand2

    return None


def make_split(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)

    n = len(df)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    split = np.array(["test"] * n, dtype=object)
    split[idx[:n_train]] = "train"
    split[idx[n_train:n_train + n_val]] = "val"
    df["split"] = split
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_npz = out_dir / "npz"
    out_npz.mkdir(parents=True, exist_ok=True)

    all_files = [p for p in raw_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    img_paths = [p for p in all_files if is_image_file(p)]

    if len(img_paths) == 0:
        raise RuntimeError(f"No valid image files found under: {raw_dir}")

    print(f"raw_dir = {raw_dir}")
    print(f"all image-like files = {len(all_files)}")
    print(f"valid input images = {len(img_paths)}")
    print("first 10 valid input images:")
    for p in img_paths[:10]:
        print("  ", p)

    records = []
    for img_path in tqdm(img_paths, desc="Scanning & preprocessing"):
        try:
            mask_path = find_gt_for_image(img_path)

            img = read_gray(img_path)
            img = resize_img(img, args.size, is_mask=False)
            img = norm01(img)

            if mask_path is not None and mask_path.exists():
                m = read_gray(mask_path)
                m = resize_img(m, args.size, is_mask=True)
                m = (m > 0).astype(np.uint8)
                has_mask = int(m.sum() > 0)
            else:
                m = np.zeros((args.size, args.size), dtype=np.uint8)
                has_mask = 0

            rel = img_path.relative_to(raw_dir)
            key = hashlib.md5(str(rel).encode("utf-8")).hexdigest()
            npz_path = out_npz / f"{key}.npz"

            np.savez_compressed(
                npz_path,
                img=img[None, ...].astype(np.float32),
                mask=m[None, ...].astype(np.uint8),
            )

            records.append(
                {
                    "id": key,
                    "img_path": str(img_path),
                    "mask_path": str(mask_path) if mask_path else "",
                    "npz_path": str(npz_path),
                    "has_mask": has_mask,
                    "sha256_img": sha256_file(img_path),
                }
            )

        except Exception as e:
            print(f"Skip: {img_path} -> {e}")
            continue

    print(f"processed records = {len(records)}")

    df = pd.DataFrame(records)
    if len(df) == 0:
        raise RuntimeError("No valid samples processed.")

    print("has_mask counts:")
    print(df["has_mask"].value_counts(dropna=False))

    # 只保留有感染 mask 的样本做分割
    df = df[df["has_mask"] == 1].reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError("No samples with valid masks were found after filtering has_mask == 1.")

    df = make_split(df, seed=args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "manifest.csv", index=False, encoding="utf-8")

    print(f"Saved manifest: {out_dir / 'manifest.csv'}")
    print(df["split"].value_counts())


if __name__ == "__main__":
    main()