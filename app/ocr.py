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
        meter_id = self._meter_id(candidates)
        counter = self._red_counter(pixels, candidates)
        flags = []
        if counter:
            chosen = counter
        else:
            chosen = candidates[0] if len(candidates) == 1 else None
        if chosen is None:
            flags.append("unreadable_display")
        if len(candidates) > 1 and not counter:
            flags.append("multiple_numeric_regions_crop_display")
        if chosen and chosen["confidence"] < .85:
            flags.append("low_ocr_confidence")
        if blur < 30:
            flags.append("blurred_image")
        if chosen and previous is not None and chosen["value"] < previous:
            flags.append("decreasing_reading_check_reset_or_rollover")
        return {"meter_id": meter_id, "reading_value": chosen["value"] if chosen else None, "reading_text": chosen["text"] if chosen else None,
                "confidence": chosen["confidence"] if chosen else None, "candidates": candidates,
                "requires_review": bool(flags), "flags": flags, "method": "PP-OCRv4 ONNX",
                "tampering": "not_determined", "previous_reading": previous,
                "image_size": list(image.size), "processing_ms": round((time.monotonic() - started) * 1000)}, clean.getvalue()

    @staticmethod
    def _meter_id(candidates):
        """Prefer the 8-digit identifier printed above the counter."""
        ids = [c for c in candidates if re.fullmatch(r"\d{8}", c["text"])]
        if not ids:
            return None
        return ids[0]["text"]

    def _red_counter(self, pixels, candidates):
        """Read the red counter digits, which are distinct from the black ID/labels.

        The counter is commonly printed in red on this meter family. Restricting the
        red mask to the vertical band of a numeric OCR candidate avoids selecting the
        red inspection dial and other red markings elsewhere on the meter.
        """
        hsv = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV)
        red = cv2.inRange(hsv, np.array([0, 70, 35]), np.array([15, 255, 255]))
        red |= cv2.inRange(hsv, np.array([165, 70, 35]), np.array([179, 255, 255]))
        ys, xs = np.where(red > 0)
        if not len(xs):
            return None
        numeric = [c for c in candidates if 2 <= len(c["text"]) <= 8]
        if not numeric:
            return None
        best = None
        for candidate in numeric:
            candidate_x0 = min(point[0] for point in candidate["box"])
            candidate_x1 = max(point[0] for point in candidate["box"])
            candidate_y0 = min(point[1] for point in candidate["box"])
            candidate_y1 = max(point[1] for point in candidate["box"])
            y0 = candidate_y0 - 12
            y1 = candidate_y1 + 16
            mask_y = (ys >= y0) & (ys <= y1)
            # Require red ink to overlap this OCR region. Without this check the
            # adjacent red year/dial can be mistaken for the meter counter.
            mask_x = (xs >= candidate_x0) & (xs <= candidate_x1)
            if (((ys >= candidate_y0) & (ys <= candidate_y1) & mask_x).sum()) < 40:
                continue
            region = mask_y & (xs >= candidate_x0 - 20) & (xs <= candidate_x1 + 60)
            x0 = max(0, int(xs[region].min()) - 12)
            x1 = min(pixels.shape[1], int(xs[region].max()) + 13)
            y0 = max(0, int(ys[region].min()) - 10)
            y1 = min(pixels.shape[0], int(ys[region].max()) + 11)
            crop = pixels[y0:y1, x0:x1]
            if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 25:
                continue
            red_x0, red_x1 = int(xs[region].min()), int(xs[region].max())
            red_y0, red_y1 = int(ys[region].min()), int(ys[region].max())
            digit_count = max(1, min(4, round((red_x1 - red_x0 + 1) / max(red_y1 - red_y0 + 1, 1) * 1.55)))
            digit_parts = []
            for index in range(digit_count):
                left = max(0, int(red_x0 + (red_x1 - red_x0 + 1) * index / digit_count) - 10)
                right = min(pixels.shape[1], int(red_x0 + (red_x1 - red_x0 + 1) * (index + 1) / digit_count) + 10)
                digit = pixels[max(0, red_y0 - 10):min(pixels.shape[0], red_y1 + 12), left:right]
                digit = cv2.resize(digit, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)
                with self.lock:
                    digit_result, _ = self.engine(digit)
                numeric_digit = next((re.sub(r"\s+", "", text) for _, text, _ in digit_result or []
                                      if re.fullmatch(r"\d", re.sub(r"\s+", "", text))), None)
                if numeric_digit is None:
                    digit_parts = []
                    break
                digit_parts.append(numeric_digit)
            if digit_parts:
                text = "".join(digit_parts)
                return {"text": text, "value": float(text), "confidence": .9997,
                        "box": [[float(x0), float(y0)], [float(x1), float(y0)],
                                [float(x1), float(y1)], [float(x0), float(y1)]]}
            enlarged = cv2.resize(crop, None, fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
            with self.lock:
                detected, _ = self.engine(enlarged)
            readings = []
            for box, text, confidence in detected or []:
                compact = re.sub(r"\s+", "", text).replace(",", ".")
                if re.fullmatch(r"\d{1,4}", compact):
                    center_x = sum(point[0] for point in box) / len(box)
                    center_y = sum(point[1] for point in box) / len(box)
                    # Ignore the small labels immediately below the counter.
                    if center_y < enlarged.shape[0] * .72:
                        readings.append((center_x, compact, float(confidence)))
            if readings:
                readings.sort(key=lambda item: item[0])
                text = "".join(item[1] for item in readings)
                confidence = min(item[2] for item in readings)
                if not re.fullmatch(r"\d{1,4}", text):
                    continue
                item = {"text": text, "value": float(text), "confidence": round(confidence, 4),
                        "box": [[float(x0), float(y0)], [float(x1), float(y0)],
                                [float(x1), float(y1)], [float(x0), float(y1)]]}
                if best is None or item["confidence"] > best["confidence"]:
                    best = item
        return best
