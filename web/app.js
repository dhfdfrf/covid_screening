const form = document.getElementById("segment-form");
const fileInput = document.getElementById("image-file");
const gtMaskInput = document.getElementById("gt-mask-file");
const fileName = document.getElementById("file-name");
const gtFileName = document.getElementById("gt-file-name");
const imageDropzone = document.getElementById("image-dropzone");
const gtDropzone = document.getElementById("gt-dropzone");
const runButton = document.getElementById("run-button");
const statusText = document.getElementById("status-text");
const statusHint = document.getElementById("status-hint");
const checkpointInput = document.getElementById("checkpoint-path");
const checkpointList = document.getElementById("checkpoint-list");
const defaultCheckpoint = document.getElementById("default-checkpoint");
const thresholdInput = form.querySelector('[name="threshold"]');
const minAreaInput = form.querySelector('[name="min_area"]');
const ttaSelect = form.querySelector('[name="tta_mode"]');
const savedFiles = document.getElementById("saved-files");
const comparePanel = document.getElementById("compare-panel");

const previewOriginal = document.getElementById("preview-original");
const previewMask = document.getElementById("preview-mask");
const previewOverlay = document.getElementById("preview-overlay");
const previewGtMask = document.getElementById("preview-gt-mask");
const previewGtOverlay = document.getElementById("preview-gt-overlay");
const previewErrorOverlay = document.getElementById("preview-error-overlay");

const downloadOriginal = document.getElementById("download-original");
const downloadMask = document.getElementById("download-mask");
const downloadOverlay = document.getElementById("download-overlay");
const downloadGtMask = document.getElementById("download-gt-mask");
const downloadGtOverlay = document.getElementById("download-gt-overlay");
const downloadErrorOverlay = document.getElementById("download-error-overlay");

const metricArea = document.getElementById("metric-area");
const metricDevice = document.getElementById("metric-device");
const metricTta = document.getElementById("metric-tta");
const metricSave = document.getElementById("metric-save");
const metricDice = document.getElementById("metric-dice");
const metricIou = document.getElementById("metric-iou");
const metricPrecision = document.getElementById("metric-precision");
const metricRecall = document.getElementById("metric-recall");
const metricPixelDiff = document.getElementById("metric-pixel-diff");
const metricAreaGap = document.getElementById("metric-area-gap");
const gtSourceText = document.getElementById("gt-source-text");

function setStatus(text, hint = "") {
  statusText.textContent = text;
  statusHint.textContent = hint;
}

function setDownload(anchor, dataUrl, filename) {
  anchor.href = dataUrl || "#";
  anchor.download = filename;
}

function setPreview(target, src) {
  target.src = src || "";
}

function updateSelectedFile(input, label, emptyText) {
  const file = input.files && input.files[0];
  label.textContent = file ? file.name : emptyText;
}

function applyDropzone(dropzone, input, label, emptyText) {
  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.add("is-dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.remove("is-dragover");
    });
  });

  dropzone.addEventListener("drop", (event) => {
    const files = event.dataTransfer.files;
    if (files && files.length > 0) {
      input.files = files;
      updateSelectedFile(input, label, emptyText);
    }
  });
}

function resetComparison() {
  comparePanel.classList.add("is-hidden");
  metricDice.textContent = "-";
  metricIou.textContent = "-";
  metricPrecision.textContent = "-";
  metricRecall.textContent = "-";
  metricPixelDiff.textContent = "-";
  metricAreaGap.textContent = "-";
  gtSourceText.textContent = "检测到真值 mask 后显示这一部分。";
  setPreview(previewGtMask, "");
  setPreview(previewGtOverlay, "");
  setPreview(previewErrorOverlay, "");
  setDownload(downloadGtMask, "", "gt_mask.png");
  setDownload(downloadGtOverlay, "", "gt_overlay.png");
  setDownload(downloadErrorOverlay, "", "error_overlay.png");
}

async function loadConfig() {
  try {
    const response = await fetch("/api/config");
    if (!response.ok) {
      throw new Error("Failed to load config");
    }
    const config = await response.json();
    defaultCheckpoint.textContent = config.default_checkpoint || "Not found";

    if (!checkpointInput.value && config.default_checkpoint) {
      checkpointInput.value = config.default_checkpoint;
    }
    if (thresholdInput && config.default_threshold !== undefined) {
      thresholdInput.value = Number(config.default_threshold).toFixed(2);
    }
    if (minAreaInput && config.default_min_area !== undefined) {
      minAreaInput.value = String(config.default_min_area);
    }
    if (ttaSelect && config.default_tta_mode) {
      ttaSelect.value = config.default_tta_mode;
    }

    checkpointList.innerHTML = "";
    (config.available_checkpoints || []).forEach((path) => {
      const option = document.createElement("option");
      option.value = path;
      checkpointList.appendChild(option);
    });
  } catch (error) {
    defaultCheckpoint.textContent = "Unavailable";
    setStatus("Config load failed.", String(error));
  }
}

fileInput.addEventListener("change", () => {
  updateSelectedFile(fileInput, fileName, "未选择文件");
});
gtMaskInput.addEventListener("change", () => {
  updateSelectedFile(gtMaskInput, gtFileName, "未选择真值文件");
});

applyDropzone(imageDropzone, fileInput, fileName, "未选择文件");
applyDropzone(gtDropzone, gtMaskInput, gtFileName, "未选择真值文件");

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!fileInput.files || fileInput.files.length === 0) {
    setStatus("请选择胸片文件。", "支持 PNG / JPG / TIFF / DICOM。");
    return;
  }

  const uploadName = fileInput.files[0].name;
  const gtUploadName = gtMaskInput.files && gtMaskInput.files[0] ? gtMaskInput.files[0].name : "gt_mask.png";
  const formData = new FormData(form);

  runButton.disabled = true;
  resetComparison();
  setStatus("正在运行分割...", "模型首次加载会更慢。");

  try {
    const response = await fetch("/api/segment", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Segmentation failed");
    }

    setPreview(previewOriginal, payload.images.original);
    setPreview(previewMask, payload.images.mask);
    setPreview(previewOverlay, payload.images.overlay);

    setDownload(downloadOriginal, payload.images.original, `${uploadName}_original.png`);
    setDownload(downloadMask, payload.images.mask, `${uploadName}_pred_mask.png`);
    setDownload(downloadOverlay, payload.images.overlay, `${uploadName}_pred_overlay.png`);

    metricArea.textContent = `${payload.infection_area_percent.toFixed(2)}%`;
    metricDevice.textContent = payload.device;
    metricTta.textContent = payload.tta_mode;
    metricSave.textContent = payload.saved_files.overlay;

    let savedMarkup = `
      <strong>已保存到本地</strong><br>
      original: ${payload.saved_files.original}<br>
      pred mask: ${payload.saved_files.mask}<br>
      pred overlay: ${payload.saved_files.overlay}<br>
      meta: ${payload.saved_files.meta}
    `;

    if (payload.comparison) {
      const metrics = payload.comparison.metrics;
      comparePanel.classList.remove("is-hidden");
      metricDice.textContent = metrics.dice.toFixed(4);
      metricIou.textContent = metrics.iou.toFixed(4);
      metricPrecision.textContent = metrics.precision.toFixed(4);
      metricRecall.textContent = metrics.recall.toFixed(4);
      metricPixelDiff.textContent = `${(metrics.pixel_diff_ratio * 100).toFixed(2)}%`;
      metricAreaGap.textContent = `${metrics.area_gap_percent >= 0 ? "+" : ""}${metrics.area_gap_percent.toFixed(2)}%`;

      setPreview(previewGtMask, payload.comparison.images.gt_mask);
      setPreview(previewGtOverlay, payload.comparison.images.gt_overlay);
      setPreview(previewErrorOverlay, payload.comparison.images.error_overlay);

      setDownload(downloadGtMask, payload.comparison.images.gt_mask, `${gtUploadName}_gt_mask.png`);
      setDownload(downloadGtOverlay, payload.comparison.images.gt_overlay, `${gtUploadName}_gt_overlay.png`);
      setDownload(downloadErrorOverlay, payload.comparison.images.error_overlay, `${uploadName}_error_overlay.png`);

      if (payload.comparison.source) {
        const source = payload.comparison.source;
        gtSourceText.textContent = source.mode === "auto"
          ? `GT 自动匹配：${source.path} (${source.match})`
          : `GT 来自上传文件：${source.path}`;
      }

      savedMarkup += `
        <br>gt mask: ${payload.comparison.saved_files.gt_mask}
        <br>gt overlay: ${payload.comparison.saved_files.gt_overlay}
        <br>error overlay: ${payload.comparison.saved_files.error_overlay}
      `;
      if (payload.comparison.source) {
        savedMarkup += `<br>gt source: ${payload.comparison.source.path}`;
      }
    }

    savedFiles.innerHTML = savedMarkup;
    setStatus("分割完成。", `Checkpoint: ${payload.checkpoint_path}`);
  } catch (error) {
    resetComparison();
    setStatus("分割失败。", String(error));
  } finally {
    runButton.disabled = false;
  }
});

resetComparison();
loadConfig();
