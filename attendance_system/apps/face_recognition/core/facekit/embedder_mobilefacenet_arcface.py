from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np


def lazy_import_cv2():
	import cv2  # type: ignore

	return cv2


def lazy_import_ort():
	import onnxruntime as ort  # type: ignore

	return ort


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
	norm = float(np.linalg.norm(vec))
	return vec / max(norm, eps)


class MobileFaceNetArcFaceEmbedder:
	"""ONNX MobileFaceNet embedder (ArcFace-style preprocessing + L2 normalize)."""

	def __init__(self, onnx_path: Path, providers: Optional[List[str]] = None):
		self.onnx_path = onnx_path
		self._cv2 = lazy_import_cv2()
		ort = lazy_import_ort()

		if providers is None:
			providers = ["CPUExecutionProvider"]

		self.session = ort.InferenceSession(str(onnx_path), providers=providers)
		self.input_name = self.session.get_inputs()[0].name
		self.output_name = self.session.get_outputs()[0].name

		in_shape = self.session.get_inputs()[0].shape
		self.input_h = int(in_shape[2]) if isinstance(in_shape[2], int) else 112
		self.input_w = int(in_shape[3]) if isinstance(in_shape[3], int) else 112

	def embed_face_bgr(self, face_bgr: np.ndarray) -> np.ndarray:
		face = self._cv2.resize(face_bgr, (self.input_w, self.input_h), interpolation=self._cv2.INTER_LINEAR)
		face = face.astype(np.float32)

		# Common ArcFace preprocessing: (img - 127.5) / 128
		face = (face - 127.5) / 128.0
		face = np.transpose(face, (2, 0, 1))  # HWC -> CHW
		face = np.expand_dims(face, axis=0)

		emb = self.session.run([self.output_name], {self.input_name: face})[0]
		emb = np.asarray(emb, dtype=np.float32).reshape(-1)
		return l2_normalize(emb)
