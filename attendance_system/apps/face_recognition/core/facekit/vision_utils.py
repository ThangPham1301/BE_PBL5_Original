from __future__ import annotations

from typing import Tuple

import numpy as np


def square_crop(bgr: np.ndarray, box: Tuple[int, int, int, int], scale: float = 1.25) -> np.ndarray:
	"""Square crop around a box (x1,y1,x2,y2), scaled and clipped."""
	h, w = bgr.shape[:2]
	x1, y1, x2, y2 = box
	bw = x2 - x1
	bh = y2 - y1
	cx = x1 + bw / 2.0
	cy = y1 + bh / 2.0
	side = max(bw, bh) * scale

	nx1 = int(round(cx - side / 2.0))
	ny1 = int(round(cy - side / 2.0))
	nx2 = int(round(cx + side / 2.0))
	ny2 = int(round(cy + side / 2.0))

	nx1 = max(0, nx1)
	ny1 = max(0, ny1)
	nx2 = min(w, nx2)
	ny2 = min(h, ny2)

	if nx2 <= nx1 or ny2 <= ny1:
		return bgr[y1:y2, x1:x2]

	return bgr[ny1:ny2, nx1:nx2]


def draw_label(cv2, frame: np.ndarray, x1: int, y1: int, text: str) -> None:
	(tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
	y_text = max(0, y1 - 8)
	cv2.rectangle(frame, (x1, y_text - th - baseline), (x1 + tw + 6, y_text + baseline), (0, 0, 0), -1)
	cv2.putText(frame, text, (x1 + 3, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
