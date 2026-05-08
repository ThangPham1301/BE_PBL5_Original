from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np


def lazy_import_cv2():
	import cv2  # type: ignore

	return cv2


def lazy_import_torch():
	import torch  # type: ignore
	import torch.nn as nn  # type: ignore

	return torch, nn


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
	norm = float(np.linalg.norm(vec))
	return vec / max(norm, eps)


torch, nn = lazy_import_torch()


class ConvBN(nn.Module):
	def __init__(self, inp, oup, k=3, s=1, p=1, groups=1):
		super().__init__()
		self.block = nn.Sequential(
			nn.Conv2d(inp, oup, k, s, p, groups=groups, bias=False),
			nn.BatchNorm2d(oup),
			nn.PReLU(oup),
		)

	def forward(self, x): return self.block(x)


class Bottleneck(nn.Module):
	def __init__(self, inp, oup, stride, expand_ratio):
		super().__init__()
		hidden = inp * expand_ratio
		self.use_res = stride == 1 and inp == oup
		self.conv = nn.Sequential(
			nn.Conv2d(inp, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.PReLU(hidden),
			nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False), nn.BatchNorm2d(hidden), nn.PReLU(hidden),
			nn.Conv2d(hidden, oup, 1, bias=False), nn.BatchNorm2d(oup),
		)

	def forward(self, x): return x + self.conv(x) if self.use_res else self.conv(x)


class MobileFaceNet(nn.Module):
	CFG = [(2, 64, 5, 2), (4, 128, 1, 2), (2, 128, 6, 1), (4, 128, 1, 2), (2, 128, 2, 1)]

	def __init__(self, emb=512):
		super().__init__()
		self.stem = nn.Sequential(ConvBN(3, 64, s=2), ConvBN(64, 64, groups=64))
		layers, inp = [], 64
		for t, c, n, s in self.CFG:
			for i in range(n):
				layers.append(Bottleneck(inp, c, s if i == 0 else 1, t))
				inp = c
		self.blocks = nn.Sequential(*layers)
		self.conv_last = ConvBN(128, 512, k=1, s=1, p=0)
		self.gdc = nn.Sequential(nn.Conv2d(512, 512, 7, groups=512, bias=False), nn.BatchNorm2d(512))
		self.output = nn.Sequential(nn.Flatten(), nn.Linear(512, emb, bias=False), nn.BatchNorm1d(emb))

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
				nn.init.ones_(m.weight)
				nn.init.zeros_(m.bias)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
			elif isinstance(m, nn.PReLU):
				nn.init.constant_(m.weight, 0.25)

	def forward(self, x):
		return self.output(self.gdc(self.conv_last(self.blocks(self.stem(x)))))


class MobileFaceNetTorchEmbedder:
	"""PyTorch MobileFaceNet embedder loaded from MFNet.pth checkpoint."""

	def __init__(self, checkpoint_path: Path, emb_size: int = 512, device: Optional[str] = None):
		self._cv2 = lazy_import_cv2()
		self.checkpoint_path = checkpoint_path

		if not checkpoint_path.exists():
			raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

		if device is None:
			device = "cuda:0" if torch.cuda.is_available() else "cpu"

		self.device = torch.device(device)
		self.model = MobileFaceNet(emb=emb_size).to(self.device)
		self._load_weights(checkpoint_path)
		self.model.eval()

	def _normalize_state_dict_keys(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
		normalized: Dict[str, torch.Tensor] = {}
		for key, value in state_dict.items():
			new_key = key
			for prefix in ("module.", "model.", "backbone.", "net."):
				if new_key.startswith(prefix):
					new_key = new_key[len(prefix):]
			normalized[new_key] = value
		return normalized

	def _extract_best_state_dict(self, checkpoint: object) -> Dict[str, torch.Tensor]:
		candidates = []
		if isinstance(checkpoint, dict):
			candidates.append(checkpoint)
			for key in ("state_dict", "model_state_dict", "backbone_state_dict"):
				nested = checkpoint.get(key)
				if isinstance(nested, dict):
					candidates.append(nested)
		else:
			raise RuntimeError("Unsupported checkpoint format. Expected dict-like object.")

		model_state = self.model.state_dict()
		model_keys = set(model_state.keys())
		best_state: Optional[Dict[str, torch.Tensor]] = None
		best_score = 0

		for candidate in candidates:
			normalized = self._normalize_state_dict_keys(candidate)
			filtered: Dict[str, torch.Tensor] = {}
			for key, value in normalized.items():
				if key in model_state and hasattr(value, "shape") and tuple(value.shape) == tuple(model_state[key].shape):
					filtered[key] = value

			score = len(filtered)
			if score > best_score:
				best_score = score
				best_state = filtered

		if best_state is None or best_score == 0:
			raise RuntimeError("No compatible MobileFaceNet weights found in checkpoint.")

		coverage = best_score / max(1, len(model_keys))
		if coverage < 0.85:
			raise RuntimeError(
				f"Incompatible checkpoint: only matched {best_score}/{len(model_keys)} tensors ({coverage:.1%})."
			)

		return best_state

	def _load_weights(self, checkpoint_path: Path) -> None:
		checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
		state_dict = self._extract_best_state_dict(checkpoint)
		missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

		missing_non_tracked = [key for key in missing if not key.endswith("num_batches_tracked")]
		if missing_non_tracked:
			raise RuntimeError(f"Missing weights for MobileFaceNet: {missing_non_tracked}")
		if unexpected:
			raise RuntimeError(f"Unexpected weights for MobileFaceNet: {unexpected}")

	def embed_face_bgr(self, face_bgr: np.ndarray) -> np.ndarray:
		face = self._cv2.resize(face_bgr, (112, 112), interpolation=self._cv2.INTER_LINEAR)
		face = self._cv2.cvtColor(face, self._cv2.COLOR_BGR2RGB)
		face = face.astype(np.float32)
		face = (face - 127.5) / 128.0
		face = np.transpose(face, (2, 0, 1)).copy()  # HWC -> CHW

		tensor = torch.from_numpy(face).unsqueeze(0).to(self.device)
		with torch.no_grad():
			embedding = self.model(tensor).detach().cpu().numpy().astype(np.float32).reshape(-1)

		return l2_normalize(embedding)
