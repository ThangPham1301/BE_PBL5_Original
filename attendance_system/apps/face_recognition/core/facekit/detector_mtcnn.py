from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


def lazy_import_mtcnn():
	from facenet_pytorch import MTCNN  # type: ignore

	return MTCNN


def lazy_import_torch():
	import torch  # type: ignore

	return torch


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


class MTCNNFaceDetector:
	def __init__(self, device: Optional[str] = None, min_face_size: int = 40):
		torch = lazy_import_torch()
		MTCNN = lazy_import_mtcnn()

		if device is None:
			device = "cuda:0" if torch.cuda.is_available() else "cpu"

		self.detector = MTCNN(
			keep_all=True,
			post_process=False,
			device=device,
			min_face_size=min_face_size,
		)

	def detect(self, bgr: np.ndarray, conf_threshold: float = 0.6) -> List[Detection]:
		if bgr is None or bgr.size == 0:
			return []

		h, w = bgr.shape[:2]
		rgb = bgr[:, :, ::-1]

		boxes, probs = self.detector.detect(rgb)
		if boxes is None or probs is None:
			return []

		results: List[Detection] = []
		for box, prob in zip(boxes, probs):
			confidence = float(prob) if prob is not None else 0.0
			if confidence < conf_threshold:
				continue

			x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
			x1 = max(0, min(w - 1, x1))
			y1 = max(0, min(h - 1, y1))
			x2 = max(0, min(w - 1, x2))
			y2 = max(0, min(h - 1, y2))

			if x2 <= x1 or y2 <= y1:
				continue

			results.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence))

		results.sort(key=lambda d: d.confidence, reverse=True)
		return results
