from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .io_utils import download_first_available


def lazy_import_cv2():
	import cv2  # type: ignore

	return cv2


def ensure_resnet10_detector_models(models_dir: Path) -> Tuple[Path, Path]:
	"""Ensure the ResNet-10 SSD detector prototxt and caffemodel exist."""

	models_dir.mkdir(parents=True, exist_ok=True)
	prototxt = models_dir / "deploy.prototxt"
	caffemodel = models_dir / "res10_300x300_ssd_iter_140000_fp16.caffemodel"

	def _read_text_head(path: Path, max_bytes: int = 64 * 1024) -> str:
		data = path.read_bytes()[:max_bytes]
		return data.decode("utf-8", errors="ignore")

	def _looks_like_valid_prototxt(path: Path) -> bool:
		try:
			text = _read_text_head(path)
		except Exception:
			return False

		low = text.lower()
		if "<html" in low or "<!doctype" in low:
			return False
		# We've seen corrupted files containing extra separators like '---'.
		if "---" in text:
			return False
		# Minimal sanity checks for this specific model
		return ("input:" in text) and ("layer" in text) and ("DetectionOutput" in text)

	def _looks_like_valid_caffemodel(path: Path) -> bool:
		try:
			# Real file is ~5-6MB; anything tiny is definitely wrong.
			return path.stat().st_size > 1_000_000
		except Exception:
			return False

	if prototxt.exists() and not _looks_like_valid_prototxt(prototxt):
		prototxt.unlink(missing_ok=True)

	if caffemodel.exists() and not _looks_like_valid_caffemodel(caffemodel):
		caffemodel.unlink(missing_ok=True)

	if not prototxt.exists():
		download_first_available(
			[
				"https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
				"https://raw.githubusercontent.com/Shiva486/facial_recognition/master/deploy.prototxt.txt",
			],
			prototxt,
		)

	if not caffemodel.exists():
		download_first_available(
			[
				"https://raw.githubusercontent.com/Shiva486/facial_recognition/master/res10_300x300_ssd_iter_140000.caffemodel",
				# OpenCV 3rdparty hosts this on a branch/tag, not necessarily on master.
				"https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000_fp16.caffemodel",
			],
			caffemodel,
		)

	return prototxt, caffemodel


@dataclass(frozen=True)
class Detection:
	x1: int
	y1: int
	x2: int
	y2: int
	confidence: float

	@property
	def w(self): return self.x2 - self.x1
	
	@property
	def h(self): return self.y2 - self.y1

	def as_tuple(self) -> Tuple[int, int, int, int]:
		return self.x1, self.y1, self.x2, self.y2


class ResNet10FaceDetector:
	def __init__(self, prototxt_path: Path, caffemodel_path: Path, input_size: int = 300):
		self.prototxt_path = prototxt_path
		self.caffemodel_path = caffemodel_path
		self.input_size = input_size
		self._cv2 = lazy_import_cv2()
		self.net = self._cv2.dnn.readNetFromCaffe(str(prototxt_path), str(caffemodel_path))

	def detect(self, bgr: np.ndarray, conf_threshold: float = 0.6) -> List[Detection]:
		h, w = bgr.shape[:2]
		blob = self._cv2.dnn.blobFromImage(
			bgr,
			scalefactor=1.0,
			size=(self.input_size, self.input_size),
			mean=(104.0, 177.0, 123.0),
			swapRB=False,
			crop=False,
		)
		self.net.setInput(blob)
		detections = self.net.forward()

		results: List[Detection] = []
		for i in range(detections.shape[2]):
			confidence = float(detections[0, 0, i, 2])
			if confidence < conf_threshold:
				continue

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			x1, y1, x2, y2 = box.astype(int)
			x1 = max(0, min(w - 1, x1))
			y1 = max(0, min(h - 1, y1))
			x2 = max(0, min(w - 1, x2))
			y2 = max(0, min(h - 1, y2))
			if x2 <= x1 or y2 <= y1:
				continue

			results.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence))

		results.sort(key=lambda d: d.confidence, reverse=True)
		return results
