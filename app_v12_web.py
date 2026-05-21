from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import tempfile
import threading
from dataclasses import dataclass, field
from email.parser import BytesParser
from email.policy import default
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.v12_predictor import DEFAULT_V12_CKPT, SUPPORTED_IMAGE_EXTS, V12Segmenter


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
OUTPUT_DIR = ROOT / "outputs" / "web_infer"
DATA_ROOT = ROOT / "data" / "raw" / "qata"
MANIFEST_PATH = ROOT / "data" / "processed" / "qata" / "manifest.csv"
SUPPORTED_MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
STATIC_FILES = {
    "/static/styles.css": WEB_DIR / "styles.css",
    "/static/app.js": WEB_DIR / "app.js",
}


def _image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _resolve_checkpoint_path(raw_value: str) -> Path:
    value = raw_value.strip()
    if not value:
        return DEFAULT_V12_CKPT.resolve()

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    else:
        path = path.resolve()
    return path


def list_available_checkpoints() -> list[str]:
    outputs_dir = ROOT / "outputs"
    config_dir = ROOT / "configs"
    candidates: set[str] = set()

    for path in (DEFAULT_V12_CKPT.resolve(),):
        if path.is_file():
            candidates.add(str(path))

    for base_dir, patterns in (
        (config_dir, ("v15*.json",)),
        (outputs_dir, ("*v12*.pt", "*v13*.pt", "*v14*.pt", "*v16*.pt", "*v17*.pt", "*v18*.pt", "*v19*.pt", "*v20*.pt", "*uctransnet*.pt")),
    ):
        if not base_dir.exists():
            continue
        for pattern in patterns:
            for path in base_dir.glob(pattern):
                if path.is_file():
                    candidates.add(str(path.resolve()))
    return sorted(candidates)


def _read_infer_defaults(path: Path) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "threshold": 0.5,
        "min_area": 0,
        "tta_mode": "none",
    }
    if path.suffix.lower() != ".json" or not path.exists():
        return defaults

    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return defaults

    if "threshold" in config:
        defaults["threshold"] = float(config["threshold"])
    if "min_area" in config:
        defaults["min_area"] = int(config["min_area"])
    if "tta_mode" in config:
        defaults["tta_mode"] = str(config["tta_mode"])
    return defaults


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_mask_for_image_path(image_path: Path) -> Path | None:
    stem = image_path.stem
    candidate_dirs = []

    if image_path.parent.name.lower() in {"images", "image"}:
        candidate_dirs.append(image_path.parent.parent / "Ground-truths")
        candidate_dirs.append(image_path.parent.parent / "Masks")
        candidate_dirs.append(image_path.parent.parent / "masks")

    for parent in image_path.parents:
        if parent.name.lower() in {"train set", "test set", "val set", "validation set"}:
            candidate_dirs.append(parent / "Ground-truths")
            candidate_dirs.append(parent / "Masks")
            candidate_dirs.append(parent / "masks")
            break

    for candidate_dir in candidate_dirs:
        if not candidate_dir.exists():
            continue
        for suffix in SUPPORTED_MASK_EXTS:
            for name in (f"mask_{stem}{suffix}", f"{stem}{suffix}"):
                candidate = candidate_dir / name
                if candidate.exists():
                    return candidate
    return None


def _find_gt_from_manifest(upload_hash: str) -> tuple[Path, str] | None:
    if not MANIFEST_PATH.exists():
        return None

    with MANIFEST_PATH.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("sha256_img") != upload_hash:
                continue
            mask_path = Path(row.get("mask_path", ""))
            if mask_path.exists():
                return mask_path.resolve(), "manifest_sha256"
            image_path = Path(row.get("img_path", ""))
            mask_path = _find_mask_for_image_path(image_path)
            if mask_path is not None:
                return mask_path.resolve(), "manifest_image_path"
    return None


def _find_gt_from_raw_dataset(upload_filename: str, upload_hash: str) -> tuple[Path, str] | None:
    if not DATA_ROOT.exists():
        return None

    filename = Path(upload_filename).name
    stem = Path(upload_filename).stem

    for image_path in sorted(DATA_ROOT.rglob(filename)):
        if not image_path.is_file():
            continue
        try:
            if _sha256_file(image_path) != upload_hash:
                continue
        except OSError:
            continue
        mask_path = _find_mask_for_image_path(image_path)
        if mask_path is not None:
            return mask_path.resolve(), "raw_image_sha256"

    for suffix in sorted(SUPPORTED_MASK_EXTS):
        for pattern in (f"mask_{stem}{suffix}", f"{stem}{suffix}"):
            matches = sorted(DATA_ROOT.rglob(pattern))
            if matches:
                return matches[0].resolve(), "raw_mask_filename"
    return None


def _find_auto_gt_mask(upload_filename: str, upload_content: bytes) -> tuple[Path, str] | None:
    upload_hash = _sha256_bytes(upload_content)
    return (
        _find_gt_from_manifest(upload_hash)
        or _find_gt_from_raw_dataset(upload_filename, upload_hash)
    )


def _load_mask_image(path: Path, size: tuple[int, int]) -> np.ndarray:
    mask = Image.open(path).convert("L").resize(size, Image.Resampling.NEAREST)
    mask_np = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
    return mask_np


def _build_mask_overlay(original_u8: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    base = np.stack([original_u8, original_u8, original_u8], axis=-1).astype(np.float32)
    mask_f = (mask > 0).astype(np.float32)[..., None]
    tint = np.zeros_like(base)
    tint[..., 0] = color[0]
    tint[..., 1] = color[1]
    tint[..., 2] = color[2]
    overlay = base * (1.0 - 0.38 * mask_f) + tint * (0.38 * mask_f)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _build_error_overlay(original_u8: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    overlay = np.stack([original_u8, original_u8, original_u8], axis=-1).astype(np.float32)
    tp = (pred == 1) & (gt == 1)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)

    overlay[tp] = 0.55 * overlay[tp] + 0.45 * np.array([0, 255, 0], dtype=np.float32)
    overlay[fp] = 0.45 * overlay[fp] + 0.55 * np.array([255, 0, 0], dtype=np.float32)
    overlay[fn] = 0.45 * overlay[fn] + 0.55 * np.array([0, 80, 255], dtype=np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _compute_comparison(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    inter = tp
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    union = int(((pred == 1) | (gt == 1)).sum())
    eps = 1e-6

    return {
        "dice": float((2 * inter + eps) / (pred_sum + gt_sum + eps)),
        "iou": float((inter + eps) / (union + eps)),
        "precision": float((tp + eps) / (tp + fp + eps)),
        "recall": float((tp + eps) / (tp + fn + eps)),
        "pixel_diff_ratio": float((pred != gt).mean()),
        "pred_area_percent": float(pred.mean() * 100.0),
        "gt_area_percent": float(gt.mean() * 100.0),
        "area_gap_percent": float(pred.mean() * 100.0 - gt.mean() * 100.0),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _comparison_stem(saved_files: dict[str, Path]) -> str:
    meta_name = saved_files["meta"].stem
    suffix = "_meta"
    return meta_name[:-len(suffix)] if meta_name.endswith(suffix) else meta_name


def _save_comparison_artifacts(
    output_dir: Path,
    stem: str,
    gt_mask: np.ndarray,
    gt_overlay: np.ndarray,
    error_overlay: np.ndarray,
) -> dict[str, Path]:
    gt_mask_path = output_dir / f"{stem}_gt_mask.png"
    gt_overlay_path = output_dir / f"{stem}_gt_overlay.png"
    error_overlay_path = output_dir / f"{stem}_error_overlay.png"

    Image.fromarray((gt_mask * 255).astype(np.uint8)).save(gt_mask_path)
    Image.fromarray(gt_overlay).save(gt_overlay_path)
    Image.fromarray(error_overlay).save(error_overlay_path)

    return {
        "gt_mask": gt_mask_path,
        "gt_overlay": gt_overlay_path,
        "error_overlay": error_overlay_path,
    }


@dataclass
class AppState:
    output_dir: Path = OUTPUT_DIR
    segmenters: dict[Path, V12Segmenter] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def get_segmenter(self, checkpoint_path: Path) -> V12Segmenter:
        with self.lock:
            segmenter = self.segmenters.get(checkpoint_path)
            if segmenter is None:
                segmenter = V12Segmenter(checkpoint_path=checkpoint_path)
                self.segmenters[checkpoint_path] = segmenter
            return segmenter


@dataclass(frozen=True)
class FormPart:
    name: str
    filename: str | None
    value: str | None
    content: bytes


class V12WebHandler(BaseHTTPRequestHandler):
    server_version = "V12Web/1.0"

    @property
    def state(self) -> AppState:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_file(WEB_DIR / "index.html", "text/html; charset=utf-8")
            return

        if self.path == "/api/config":
            infer_defaults = _read_infer_defaults(DEFAULT_V12_CKPT.resolve())
            self._send_json(
                {
                    "default_checkpoint": str(DEFAULT_V12_CKPT.resolve()),
                    "available_checkpoints": list_available_checkpoints(),
                    "supported_extensions": sorted(SUPPORTED_IMAGE_EXTS),
                    "default_output_dir": str(self.state.output_dir.resolve()),
                    "default_threshold": infer_defaults["threshold"],
                    "default_min_area": infer_defaults["min_area"],
                    "default_tta_mode": infer_defaults["tta_mode"],
                }
            )
            return

        static_path = STATIC_FILES.get(self.path)
        if static_path is not None:
            content_type = "text/plain; charset=utf-8"
            if static_path.suffix == ".css":
                content_type = "text/css; charset=utf-8"
            elif static_path.suffix == ".js":
                content_type = "application/javascript; charset=utf-8"
            self._send_file(static_path, content_type)
            return

        self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path != "/api/segment":
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return

        try:
            self._handle_segment()
        except Exception as exc:  # pragma: no cover - exercised by manual use
            self._send_json(
                {"error": str(exc)},
                status=HTTPStatus.BAD_REQUEST,
            )

    def _handle_segment(self) -> None:
        form = self._parse_multipart_form()

        if "image_file" not in form:
            raise ValueError("Missing uploaded image file.")

        file_item = form["image_file"]
        if not file_item.filename:
            raise ValueError("No image file selected.")

        suffix = Path(file_item.filename).suffix.lower()
        if suffix not in SUPPORTED_IMAGE_EXTS:
            raise ValueError(f"Unsupported image format: {suffix or 'unknown'}")

        checkpoint_path = _resolve_checkpoint_path(self._form_text(form, "checkpoint_path", ""))
        image_size = int(self._form_text(form, "image_size", "224"))
        threshold = float(self._form_text(form, "threshold", "0.5"))
        min_area = int(self._form_text(form, "min_area", "0"))
        tta_mode = self._form_text(form, "tta_mode", "none")
        gt_file_item = form.get("gt_mask_file")

        if image_size <= 0:
            raise ValueError("image_size must be greater than 0.")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1.")
        if min_area < 0:
            raise ValueError("min_area must be 0 or larger.")
        if tta_mode not in {"none", "h", "v", "hv", "all"}:
            raise ValueError("tta_mode must be one of none/h/v/hv/all.")
        if gt_file_item is not None and gt_file_item.filename:
            gt_suffix = Path(gt_file_item.filename).suffix.lower()
            if gt_suffix not in SUPPORTED_MASK_EXTS:
                raise ValueError(f"Unsupported GT mask format: {gt_suffix or 'unknown'}")

        temp_path: Path | None = None
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                safe_name = self._safe_upload_name(file_item.filename)
                temp_path = Path(tmp_dir) / safe_name
                temp_path.write_bytes(file_item.content)

                segmenter = self.state.get_segmenter(checkpoint_path)
                result = segmenter.segment_file(
                    image_path=temp_path,
                    image_size=image_size,
                    threshold=threshold,
                    min_area=min_area,
                    tta_mode=tta_mode,
                )
                saved = result.save(self.state.output_dir)
                comparison = None
                gt_source = None

                if gt_file_item is not None and gt_file_item.filename:
                    gt_safe_name = self._safe_upload_name(gt_file_item.filename)
                    gt_temp_path = Path(tmp_dir) / f"gt_{gt_safe_name}"
                    gt_temp_path.write_bytes(gt_file_item.content)

                    gt_mask = _load_mask_image(
                        gt_temp_path,
                        (result.original_image_u8.shape[1], result.original_image_u8.shape[0]),
                    )
                    gt_source = {
                        "mode": "uploaded",
                        "path": gt_file_item.filename,
                        "match": "uploaded_file",
                    }
                else:
                    auto_gt = _find_auto_gt_mask(file_item.filename, file_item.content)
                    if auto_gt is not None:
                        auto_gt_path, match_reason = auto_gt
                        gt_mask = _load_mask_image(
                            auto_gt_path,
                            (result.original_image_u8.shape[1], result.original_image_u8.shape[0]),
                        )
                        gt_source = {
                            "mode": "auto",
                            "path": str(auto_gt_path),
                            "match": match_reason,
                        }

                if gt_source is not None:
                    gt_overlay = _build_mask_overlay(result.original_image_u8, gt_mask, (61, 167, 255))
                    error_overlay = _build_error_overlay(
                        result.original_image_u8,
                        result.binary_mask.astype(np.uint8),
                        gt_mask,
                    )
                    comparison_metrics = _compute_comparison(
                        result.binary_mask.astype(np.uint8),
                        gt_mask,
                    )
                    comparison_files = _save_comparison_artifacts(
                        self.state.output_dir,
                        _comparison_stem(saved),
                        gt_mask,
                        gt_overlay,
                        error_overlay,
                    )
                    comparison = {
                        "metrics": comparison_metrics,
                        "saved_files": {
                            key: str(path.resolve()) for key, path in comparison_files.items()
                        },
                        "images": {
                            "gt_mask": _image_to_data_url(
                                Image.fromarray((gt_mask * 255).astype(np.uint8))
                            ),
                            "gt_overlay": _image_to_data_url(Image.fromarray(gt_overlay)),
                            "error_overlay": _image_to_data_url(Image.fromarray(error_overlay)),
                        },
                        "source": gt_source,
                    }

                payload = {
                    "checkpoint_path": str(result.checkpoint_path),
                    "device": result.device,
                    "input_size": list(result.input_size),
                    "threshold": result.threshold,
                    "min_area": result.min_area,
                    "tta_mode": result.tta_mode,
                    "infection_area_ratio": result.infection_area_ratio,
                    "infection_area_percent": result.infection_area_ratio * 100.0,
                    "saved_files": {key: str(path.resolve()) for key, path in saved.items()},
                    "images": {
                        "original": _image_to_data_url(Image.fromarray(result.original_image_u8)),
                        "mask": _image_to_data_url(Image.fromarray((result.binary_mask * 255).astype("uint8"))),
                        "overlay": _image_to_data_url(Image.fromarray(result.overlay_image_u8)),
                    },
                    "comparison": comparison,
                }
                self._send_json(payload)
        finally:
            temp_path = None

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return

        content = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _parse_multipart_form(self) -> dict[str, FormPart]:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            raise ValueError("Content-Type must be multipart/form-data.")

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            raise ValueError("Empty request body.")

        body = self.rfile.read(content_length)
        parser = BytesParser(policy=default)
        message = parser.parsebytes(
            (
                f"Content-Type: {content_type}\r\n"
                "MIME-Version: 1.0\r\n"
                "\r\n"
            ).encode("utf-8")
            + body
        )

        form: dict[str, FormPart] = {}
        for part in message.iter_parts():
            if part.get_content_disposition() != "form-data":
                continue

            name = part.get_param("name", header="content-disposition")
            if not name:
                continue

            filename = part.get_filename()
            content = part.get_payload(decode=True) or b""
            value = None if filename else content.decode("utf-8", errors="replace")
            form[name] = FormPart(
                name=name,
                filename=filename,
                value=value,
                content=content,
            )
        return form

    def _form_text(self, form: dict[str, FormPart], name: str, default_value: str) -> str:
        part = form.get(name)
        if part is None or part.value is None:
            return default_value
        return part.value

    def _safe_upload_name(self, filename: str) -> str:
        source = Path(filename)
        stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in source.stem).strip("_")
        stem = stem or "upload"
        suffix = source.suffix.lower()
        return f"{stem}{suffix}"


def create_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), V12WebHandler)
    server.state = AppState()  # type: ignore[attr-defined]
    return server


def main() -> None:
    host = "127.0.0.1"
    port = 8000
    server = create_server(host=host, port=port)
    print(f"V12 web app running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
