from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from src.v12_predictor import DEFAULT_V12_CKPT, SUPPORTED_IMAGE_EXTS, V12Segmenter


def _read_default_options(path: Path) -> dict[str, str]:
    defaults = {"threshold": "0.50", "min_area": "0", "tta_mode": "none"}
    if path.suffix.lower() != ".json" or not path.exists():
        return defaults
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return defaults

    if "threshold" in config:
        defaults["threshold"] = f"{float(config['threshold']):.2f}"
    if "min_area" in config:
        defaults["min_area"] = str(int(config["min_area"]))
    if "tta_mode" in config:
        defaults["tta_mode"] = str(config["tta_mode"])
    return defaults


class V12SegmentationApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("COVID Screening - V12 Segmentation")
        self.root.geometry("1380x860")
        self.root.minsize(1180, 760)
        self.root.configure(bg="#f6efe3")

        self.image_path_var = tk.StringVar()
        self.ckpt_path_var = tk.StringVar(
            value=str(DEFAULT_V12_CKPT) if DEFAULT_V12_CKPT.exists() else ""
        )
        default_options = _read_default_options(DEFAULT_V12_CKPT)
        self.image_size_var = tk.StringVar(value="224")
        self.threshold_var = tk.StringVar(value=default_options["threshold"])
        self.min_area_var = tk.StringVar(value=default_options["min_area"])
        self.tta_mode_var = tk.StringVar(value=default_options["tta_mode"])
        self.status_var = tk.StringVar(value="Ready. Select a chest image and run segmentation.")
        self.summary_var = tk.StringVar(value="No result yet.")

        self.segmenter: V12Segmenter | None = None
        self.segmenter_ckpt: Path | None = None
        self.current_result = None

        self._build_style()
        self._build_layout()

    def _build_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Root.TFrame", background="#f6efe3")
        style.configure(
            "Panel.TLabelframe",
            background="#fff9ef",
            bordercolor="#cfb78d",
            relief="solid",
        )
        style.configure(
            "Panel.TLabelframe.Label",
            background="#fff9ef",
            foreground="#4b3417",
            font=("Segoe UI Semibold", 11),
        )
        style.configure(
            "Header.TLabel",
            background="#f6efe3",
            foreground="#4b3417",
            font=("Segoe UI Semibold", 20),
        )
        style.configure(
            "Body.TLabel",
            background="#f6efe3",
            foreground="#6d5738",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Preview.TLabel",
            background="#fff9ef",
            foreground="#7d6545",
            anchor="center",
            font=("Segoe UI", 11),
        )
        style.configure(
            "Action.TButton",
            font=("Segoe UI Semibold", 10),
            padding=(10, 8),
        )

    def _build_layout(self) -> None:
        container = ttk.Frame(self.root, style="Root.TFrame", padding=18)
        container.pack(fill="both", expand=True)

        ttk.Label(container, text="V12 Chest X-ray Segmentation", style="Header.TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            container,
            text="Load a chest image, run a v12/v13/v14 checkpoint, and preview the lesion mask and overlay.",
            style="Body.TLabel",
        ).pack(anchor="w", pady=(4, 14))

        controls = ttk.LabelFrame(container, text="Controls", style="Panel.TLabelframe", padding=14)
        controls.pack(fill="x")
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Image file", style="Body.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=6)
        ttk.Entry(controls, textvariable=self.image_path_var).grid(row=0, column=1, sticky="ew", pady=6)
        ttk.Button(controls, text="Browse", style="Action.TButton", command=self._select_image).grid(row=0, column=2, padx=(10, 0), pady=6)

        ttk.Label(controls, text="V15 config or v20 checkpoint", style="Body.TLabel").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=6)
        ttk.Entry(controls, textvariable=self.ckpt_path_var).grid(row=1, column=1, sticky="ew", pady=6)
        ttk.Button(controls, text="Browse", style="Action.TButton", command=self._select_checkpoint).grid(row=1, column=2, padx=(10, 0), pady=6)

        options = ttk.Frame(controls, style="Root.TFrame")
        options.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        for idx in range(9):
            options.columnconfigure(idx, weight=1 if idx in (1, 3, 5, 7) else 0)

        ttk.Label(options, text="Input size", style="Body.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Entry(options, width=8, textvariable=self.image_size_var).grid(row=0, column=1, sticky="w", padx=(8, 24))
        ttk.Label(options, text="Threshold", style="Body.TLabel").grid(row=0, column=2, sticky="w")
        ttk.Entry(options, width=8, textvariable=self.threshold_var).grid(row=0, column=3, sticky="w", padx=(8, 24))
        ttk.Label(options, text="Min area", style="Body.TLabel").grid(row=0, column=4, sticky="w")
        ttk.Entry(options, width=8, textvariable=self.min_area_var).grid(row=0, column=5, sticky="w", padx=(8, 24))
        ttk.Label(options, text="TTA", style="Body.TLabel").grid(row=0, column=6, sticky="w")
        ttk.Combobox(
            options,
            textvariable=self.tta_mode_var,
            values=("none", "h", "v", "hv", "all"),
            state="readonly",
            width=8,
        ).grid(row=0, column=7, sticky="w", padx=(8, 24))
        ttk.Button(options, text="Run Segmentation", style="Action.TButton", command=self._run_segmentation).grid(row=0, column=8, sticky="e")

        status_box = ttk.Frame(container, style="Root.TFrame")
        status_box.pack(fill="x", pady=(12, 14))
        ttk.Label(status_box, textvariable=self.status_var, style="Body.TLabel").pack(anchor="w")
        ttk.Label(status_box, textvariable=self.summary_var, style="Body.TLabel").pack(anchor="w", pady=(4, 0))

        preview_row = ttk.Frame(container, style="Root.TFrame")
        preview_row.pack(fill="both", expand=True)
        preview_row.columnconfigure(0, weight=1)
        preview_row.columnconfigure(1, weight=1)
        preview_row.columnconfigure(2, weight=1)

        self.original_label = self._build_preview_panel(preview_row, 0, "Original")
        self.mask_label = self._build_preview_panel(preview_row, 1, "Mask")
        self.overlay_label = self._build_preview_panel(preview_row, 2, "Overlay")

        actions = ttk.Frame(container, style="Root.TFrame")
        actions.pack(fill="x", pady=(14, 0))
        ttk.Button(actions, text="Save Current Result", style="Action.TButton", command=self._save_result).pack(side="right")

    def _build_preview_panel(self, parent: ttk.Frame, column: int, title: str) -> ttk.Label:
        panel = ttk.LabelFrame(parent, text=title, style="Panel.TLabelframe", padding=10)
        panel.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 8, 0))
        parent.columnconfigure(column, weight=1)

        label = ttk.Label(panel, text="No preview", style="Preview.TLabel")
        label.pack(fill="both", expand=True)
        return label

    def _select_image(self) -> None:
        filetypes = [
            ("Chest images", " ".join(f"*{ext}" for ext in sorted(SUPPORTED_IMAGE_EXTS))),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(title="Select chest image", filetypes=filetypes)
        if path:
            self.image_path_var.set(path)

    def _select_checkpoint(self) -> None:
        path = filedialog.askopenfilename(
            title="Select checkpoint or v15 config",
            filetypes=[("Model checkpoint or config", "*.pt *.pth *.json"), ("All files", "*.*")],
        )
        if path:
            self.ckpt_path_var.set(path)

    def _ensure_segmenter(self) -> V12Segmenter:
        ckpt_path = Path(self.ckpt_path_var.get().strip()).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        if self.segmenter is None or self.segmenter_ckpt != ckpt_path:
            self.status_var.set("Loading checkpoint...")
            self.root.update_idletasks()
            self.segmenter = V12Segmenter(checkpoint_path=ckpt_path)
            self.segmenter_ckpt = ckpt_path

        return self.segmenter

    def _run_segmentation(self) -> None:
        image_path = self.image_path_var.get().strip()
        if not image_path:
            messagebox.showerror("Missing image", "Select a chest image first.")
            return

        try:
            image_size = int(self.image_size_var.get().strip())
            threshold = float(self.threshold_var.get().strip())
            min_area = int(self.min_area_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid options", "Input size, threshold, and min area must be numeric.")
            return

        if image_size <= 0:
            messagebox.showerror("Invalid input size", "Input size must be greater than 0.")
            return
        if not 0.0 <= threshold <= 1.0:
            messagebox.showerror("Invalid threshold", "Threshold must be between 0 and 1.")
            return
        if min_area < 0:
            messagebox.showerror("Invalid min area", "Min area must be 0 or larger.")
            return

        try:
            segmenter = self._ensure_segmenter()
            self.status_var.set("Running segmentation...")
            self.root.update_idletasks()
            result = segmenter.segment_file(
                image_path=image_path,
                image_size=image_size,
                threshold=threshold,
                min_area=min_area,
                tta_mode=self.tta_mode_var.get().strip(),
            )
        except Exception as exc:
            self.status_var.set("Segmentation failed.")
            messagebox.showerror("Segmentation failed", str(exc))
            return

        self.current_result = result
        self._update_preview(self.original_label, Image.fromarray(result.original_image_u8))
        self._update_preview(self.mask_label, Image.fromarray((result.binary_mask * 255).astype("uint8")))
        self._update_preview(self.overlay_label, Image.fromarray(result.overlay_image_u8))

        self.status_var.set(
            f"Done on {result.device}. Checkpoint: {result.checkpoint_path.name}"
        )
        self.summary_var.set(
            f"Infection area: {result.infection_area_ratio * 100:.2f}% | "
            f"Input: {result.input_size[0]}x{result.input_size[1]} | "
            f"TTA: {result.tta_mode} | Thr: {result.threshold:.2f} | Min area: {result.min_area}"
        )

    def _update_preview(self, label: ttk.Label, image: Image.Image) -> None:
        preview = image.copy()
        preview.thumbnail((390, 390))
        photo = ImageTk.PhotoImage(preview)
        label.configure(image=photo, text="")
        label.image = photo

    def _save_result(self) -> None:
        if self.current_result is None:
            messagebox.showinfo("No result", "Run segmentation first.")
            return

        target_dir = filedialog.askdirectory(
            title="Select output folder",
            initialdir=str(Path("outputs/gui_infer").resolve()),
            mustexist=False,
        )
        if not target_dir:
            return

        try:
            saved = self.current_result.save(target_dir)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return

        self.status_var.set(f"Saved result to {Path(target_dir).resolve()}")
        self.summary_var.set(
            f"Mask: {saved['mask'].name} | Overlay: {saved['overlay'].name} | Meta: {saved['meta'].name}"
        )


def main() -> None:
    root = tk.Tk()
    app = V12SegmentationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
