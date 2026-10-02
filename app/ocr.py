"""Image-to-reading pipeline, with recognition-derived confidence and abstention.

The pretrained OCR model detects text, not meter-specific tampering. Multiple
numeric regions require human selection; never assume a serial number is a reading.
"""
from io import BytesIO
import re
import threading
import time

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from rapidocr_onnxruntime import RapidOCR

Image.MAX_IMAGE_PIXELS = 16_000_000


class MeterOCR:
    def __init__(self):
        # The default enlarges the SHORT edge to 736px, making wide meter
        # displays several thousand pixels across. Bound the LONG edge instead.
        self.engine = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1,
                               det_limit_side_len=384, det_limit_type="max",
                               rec_batch_num=1, cls_batch_num=1)
        self.lock = threading.Lock()

    def read(self, image_bytes, previous=None):
        started = time.monotonic()
        try:
            image = Image.open(BytesIO(image_bytes))
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Please upload a JPEG, PNG or WebP image.")
            if image.width * image.height > 3_000_000:
                raise ValueError("Image exceeds the demo's 3-megapixel processing limit. Use the browser upload, which resizes photos automatically.")
            image = ImageOps.exif_transpose(image).convert("RGB")
            if min(image.size) < 60:
                raise ValueError("Image is too small; photograph the meter display more closely.")
            image.thumbnail((1600, 1600))
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError("The uploaded file is not a supported, readable image.") from exc
        # Re-encode without EXIF/GPS metadata before storage.
        clean = BytesIO()
        image.save(clean, format="JPEG", quality=85)
        pixels = np.asarray(image)
        gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        with self.lock:
            result, _ = self.engine(cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
        candidates = []
        for box, text, confidence in result or []:
            compact = re.sub(r"\s+", "", text).replace(",", ".")
            if re.fullmatch(r"\d{2,8}(?:\.\d{1,3})?", compact):
                candidates.append({"text": compact, "value": float(compact), "confidence": round(float(confidence), 4), "box": [[round(float(x), 1), round(float(y), 1)] for x, y in box]})
        candidates.sort(key=lambda c: c["confidence"], reverse=True)
        flags = []
        if not candidates:
            flags.append("unreadable_display")
        if len(candidates) > 1:
            flags.append("multiple_numeric_regions_crop_display")
        chosen = candidates[0] if len(candidates) == 1 else None
        if chosen and chosen["confidence"] < .85:
            flags.append("low_ocr_confidence")
        if blur < 30:
            flags.append("blurred_image")
        if chosen and previous is not None and chosen["value"] < previous:
            flags.append("decreasing_reading_check_reset_or_rollover")
        return {"reading_value": chosen["value"] if chosen else None, "reading_text": chosen["text"] if chosen else None,
                "confidence": chosen["confidence"] if chosen else None, "candidates": candidates,
                "requires_review": bool(flags), "flags": flags, "method": "PP-OCRv4 ONNX",
                "tampering": "not_determined", "previous_reading": previous,
                "image_size": list(image.size), "processing_ms": round((time.monotonic() - started) * 1000)}, clean.getvalue()
