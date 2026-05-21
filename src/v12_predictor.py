from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from PIL import Image

try:
    import pydicom
except ImportError:  # pragma: no cover - optional at runtime
    pydicom = None

from src.models.transunet2d_v11 import tta_inference
from src.models.transunet2d_v12 import build_transunet2d_v12
from src.models.transunet2d_v13 import build_transunet2d_v13
from src.models.transunet2d_v14 import build_transunet2d_v14
from src.models.transunet2d_v16 import build_transunet2d_v16
from src.models.transunet2d_v17 import build_transunet2d_v17
from src.models.transunet2d_v18 import build_transunet2d_v18
from src.models.transunet2d_v19 import build_transunet2d_v19
from src.models.transunet2d_v20 import build_transunet2d_v20
from src.models.uctransnet2d import build_uctransnet2d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_V12_CKPT = Path("outputs/transunet2d_v12_deep_boundary_ft_best.pt")
DEFAULT_V15_CONFIG = Path("configs/v15_ensemble_config.json")
DEFAULT_V12_CKPT = DEFAULT_V15_CONFIG
SUPPORTED_IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".dcm",
    ".dicom",
}

try:
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # pragma: no cover - pillow<9 compatibility
    RESAMPLE_BILINEAR = Image.BILINEAR


@dataclass(frozen=True)
class SegmentationResult:
    image_path: Path
    checkpoint_path: Path
    device: str
    input_size: tuple[int, int]
    threshold: float
    min_area: int
    tta_mode: str
    original_image_u8: np.ndarray
    probability_map: np.ndarray
    binary_mask: np.ndarray
    overlay_image_u8: np.ndarray
    infection_area_ratio: float

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{self.image_path.stem}_{stamp}"

        original_path = out_dir / f"{stem}_original.png"
        mask_path = out_dir / f"{stem}_mask.png"
        overlay_path = out_dir / f"{stem}_overlay.png"
        meta_path = out_dir / f"{stem}_meta.json"

        Image.fromarray(self.original_image_u8).save(original_path)
        Image.fromarray((self.binary_mask * 255).astype(np.uint8)).save(mask_path)
        Image.fromarray(self.overlay_image_u8).save(overlay_path)

        meta = {
            "image_path": str(self.image_path),
            "checkpoint_path": str(self.checkpoint_path),
            "device": self.device,
            "input_size": list(self.input_size),
            "threshold": self.threshold,
            "min_area": self.min_area,
            "tta_mode": self.tta_mode,
            "infection_area_ratio": self.infection_area_ratio,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "original": original_path,
            "mask": mask_path,
            "overlay": overlay_path,
            "meta": meta_path,
        }


def _normalize01(image: np.ndarray) -> np.ndarray:
    x = image.astype(np.float32)
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max - x_min < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def _load_dicom_grayscale(path: Path) -> np.ndarray:
    if pydicom is None:
        raise RuntimeError("pydicom is required to read DICOM files.")

    ds = pydicom.dcmread(str(path))
    image = ds.pixel_array.astype(np.float32)

    if getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
        image = image.max() - image

    return image


def load_grayscale_image(path: str | Path) -> np.ndarray:
    image_path = Path(path)
    suffix = image_path.suffix.lower()

    if suffix not in SUPPORTED_IMAGE_EXTS:
        raise ValueError(f"Unsupported image format: {suffix}")

    if suffix in {".dcm", ".dicom"}:
        image = _load_dicom_grayscale(image_path)
    else:
        image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)

    return _normalize01(image)


def build_overlay(image01: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    image_u8 = np.clip(image01 * 255.0, 0, 255).astype(np.uint8)
    rgb = np.stack([image_u8, image_u8, image_u8], axis=-1).astype(np.float32)
    red = np.zeros_like(rgb)
    red[..., 0] = 255.0
    mask_f = (mask > 0).astype(np.float32)[..., None]
    overlay = rgb * (1.0 - alpha * mask_f) + red * (alpha * mask_f)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask.astype(np.uint8)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    out = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            out[labels == label] = 1
    return out


def unwrap_logits(outputs):
    if isinstance(outputs, dict):
        return outputs["seg"]
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def predict_logits(model: torch.nn.Module, x: torch.Tensor, tta_mode: str) -> torch.Tensor:
    if tta_mode == "all":
        return tta_inference(model, x)

    seg = unwrap_logits(model(x))
    if tta_mode == "none":
        return seg

    logits = [seg]
    if tta_mode in ("h", "hv"):
        logits.append(torch.flip(unwrap_logits(model(torch.flip(x, [3]))), [3]))
    if tta_mode in ("v", "hv"):
        logits.append(torch.flip(unwrap_logits(model(torch.flip(x, [2]))), [2]))
    return torch.stack(logits, dim=0).mean(dim=0)


def build_model_by_name(model_name: str, checkpoint_path: Path | None = None) -> torch.nn.Module:
    name_hint = model_name.strip().lower()
    if checkpoint_path is not None:
        name_hint = f"{name_hint} {checkpoint_path.name.lower()}"

    if "uctransnet" in name_hint:
        return build_uctransnet2d(in_channels=1, out_channels=1)
    if "v20" in name_hint:
        return build_transunet2d_v20(in_channels=1, out_channels=1)
    if "v19" in name_hint:
        return build_transunet2d_v19(in_channels=1, out_channels=1)
    if "v18" in name_hint:
        return build_transunet2d_v18(in_channels=1, out_channels=1)
    if "v17" in name_hint:
        return build_transunet2d_v17(in_channels=1, out_channels=1)
    if "v16" in name_hint:
        return build_transunet2d_v16(in_channels=1, out_channels=1)
    if "v14" in name_hint:
        return build_transunet2d_v14(in_channels=1, out_channels=1)
    if "v13" in name_hint:
        return build_transunet2d_v13(in_channels=1, out_channels=1)
    return build_transunet2d_v12(in_channels=1, out_channels=1)


def build_model_for_checkpoint(checkpoint_path: Path, ckpt) -> torch.nn.Module:
    model_name = ""
    if isinstance(ckpt, dict):
        model_name = str(ckpt.get("model_name", "")).lower()
    return build_model_by_name(model_name, checkpoint_path)


def _resolve_project_path(value: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()

    root_candidate = (PROJECT_ROOT / path).resolve()
    if root_candidate.exists():
        return root_candidate

    if base_dir is not None:
        return (base_dir / path).resolve()
    return root_candidate


class V12Segmenter:
    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_V12_CKPT,
        device: str | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model: torch.nn.Module | None = None
        self.ensemble_members: list[dict[str, object]] = []
        self.default_threshold: float | None = None
        self.default_min_area: int | None = None
        self.default_tta_mode: str | None = None

        if self.checkpoint_path.suffix.lower() == ".json":
            self._load_ensemble_config(self.checkpoint_path)
        else:
            ckpt = torch.load(self.checkpoint_path, map_location="cpu")
            state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            self.model = build_model_for_checkpoint(self.checkpoint_path, ckpt).to(self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()

    def _load_ensemble_config(self, config_path: Path) -> None:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        members = config.get("members", [])
        if not isinstance(members, list) or not members:
            raise ValueError(f"Invalid ensemble config, missing members: {config_path}")

        raw_weights = [float(member.get("weight", 0.0)) for member in members]
        weight_sum = sum(raw_weights)
        if weight_sum <= 0:
            raise ValueError(f"Invalid ensemble config, weights must sum to > 0: {config_path}")

        self.default_threshold = float(config["threshold"]) if "threshold" in config else None
        self.default_min_area = int(config["min_area"]) if "min_area" in config else None
        self.default_tta_mode = str(config["tta_mode"]) if "tta_mode" in config else None

        for member, raw_weight in zip(members, raw_weights):
            model_name = str(member.get("model", "")).strip().lower()
            ckpt_path = _resolve_project_path(member["checkpoint"], base_dir=config_path.parent)
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            model = build_model_by_name(model_name, ckpt_path).to(self.device)
            model.load_state_dict(state_dict)
            model.eval()
            self.ensemble_members.append(
                {
                    "model": model,
                    "model_name": model_name,
                    "checkpoint_path": ckpt_path,
                    "weight": raw_weight / weight_sum,
                }
            )

    def _predict_probability(self, x: torch.Tensor, tta_mode: str) -> torch.Tensor:
        if self.ensemble_members:
            fused: torch.Tensor | None = None
            for member in self.ensemble_members:
                model = member["model"]
                weight = float(member["weight"])
                assert isinstance(model, torch.nn.Module)
                prob = torch.sigmoid(predict_logits(model, x, tta_mode=tta_mode)) * weight
                fused = prob if fused is None else fused + prob
            assert fused is not None
            return fused

        if self.model is None:
            raise RuntimeError("No model loaded.")
        return torch.sigmoid(predict_logits(self.model, x, tta_mode=tta_mode))

    @torch.no_grad()
    def segment_file(
        self,
        image_path: str | Path,
        image_size: int = 224,
        threshold: float = 0.5,
        min_area: int = 0,
        tta_mode: Literal["none", "h", "v", "hv", "all"] = "none",
    ) -> SegmentationResult:
        image_path = Path(image_path).resolve()
        image01 = load_grayscale_image(image_path)
        original_h, original_w = image01.shape

        resized = np.asarray(
            Image.fromarray(np.clip(image01 * 255.0, 0, 255).astype(np.uint8)).resize(
                (image_size, image_size),
                RESAMPLE_BILINEAR,
            ),
            dtype=np.float32,
        )
        resized = resized / 255.0

        x = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).to(self.device)
        prob = self._predict_probability(x, tta_mode=tta_mode)[0, 0].cpu().numpy()

        prob_full = cv2.resize(prob, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        mask = (prob_full > threshold).astype(np.uint8)
        mask = remove_small_components(mask, min_area=min_area)

        overlay = build_overlay(image01, mask)
        image_u8 = np.clip(image01 * 255.0, 0, 255).astype(np.uint8)

        return SegmentationResult(
            image_path=image_path,
            checkpoint_path=self.checkpoint_path,
            device=str(self.device),
            input_size=(image_size, image_size),
            threshold=threshold,
            min_area=min_area,
            tta_mode=tta_mode,
            original_image_u8=image_u8,
            probability_map=np.clip(prob_full, 0.0, 1.0).astype(np.float32),
            binary_mask=mask.astype(np.uint8),
            overlay_image_u8=overlay,
            infection_area_ratio=float(mask.mean()),
        )
