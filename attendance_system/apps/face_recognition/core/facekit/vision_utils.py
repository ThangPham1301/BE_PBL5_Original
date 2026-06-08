from __future__ import annotations

import numpy as np


def draw_label(cv2, frame: np.ndarray, x1: int, y1: int, text: str) -> None:
	(tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
	y_text = max(0, y1 - 8)
	cv2.rectangle(frame, (x1, y_text - th - baseline), (x1 + tw + 6, y_text + baseline), (0, 0, 0), -1)
	cv2.putText(frame, text, (x1 + 3, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
