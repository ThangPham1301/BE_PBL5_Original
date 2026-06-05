from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def lazy_import_cv2():
    import cv2  # type: ignore

    return cv2


def lazy_import_onnxruntime():
    import onnxruntime as ort  # type: ignore

    return ort


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def w(self):
        return self.x2 - self.x1

    @property
    def h(self):
        return self.y2 - self.y1

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


class SCRFDFaceDetector:
    """SCRFD ONNX face detector with the same output interface as ResNet10FaceDetector."""

    def __init__(self, model_path: Path, input_size: int = 640):
        self.model_path = Path(model_path)
        self.input_size = int(input_size)
        self.feat_strides = [8, 16, 32]
        self.cv2 = lazy_import_cv2()
        ort = lazy_import_onnxruntime()

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=['CPUExecutionProvider'],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.input_width, self.input_height = self._resolve_input_size()

    def _resolve_input_size(self) -> Tuple[int, int]:
        shape = self.session.get_inputs()[0].shape
        try:
            h = int(shape[2])
            w = int(shape[3])
            if h > 0 and w > 0:
                return w, h
        except Exception:
            pass

        return self.input_size, self.input_size

    def detect(self, bgr: np.ndarray, conf_threshold: float = 0.6) -> List[Detection]:
        if bgr is None or bgr.size == 0:
            return []

        blob, det_scale, original_size = self._preprocess(bgr)
        net_outs = self.session.run(self.output_names, {self.input_name: blob})
        boxes, scores = self._postprocess(net_outs, det_scale, original_size, conf_threshold)

        if len(boxes) == 0:
            return []

        keep = self._nms(boxes, scores, nms_threshold=0.4)
        h, w = original_size
        results: List[Detection] = []
        for index in keep:
            x1, y1, x2, y2 = boxes[index].astype(int)
            x1 = max(0, min(w - 1, int(x1)))
            y1 = max(0, min(h - 1, int(y1)))
            x2 = max(0, min(w - 1, int(x2)))
            y2 = max(0, min(h - 1, int(y2)))
            if x2 <= x1 or y2 <= y1:
                continue

            results.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(scores[index]),
                )
            )

        results.sort(key=lambda d: d.confidence, reverse=True)
        return results

    def _preprocess(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        model_ratio = self.input_height / self.input_width
        image_ratio = h / w

        if image_ratio > model_ratio:
            resized_h = self.input_height
            resized_w = int(resized_h / image_ratio)
        else:
            resized_w = self.input_width
            resized_h = int(resized_w * image_ratio)

        det_scale = resized_h / h
        resized = self.cv2.resize(bgr, (resized_w, resized_h), interpolation=self.cv2.INTER_LINEAR)
        canvas = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
        canvas[:resized_h, :resized_w, :] = resized

        blob = self.cv2.dnn.blobFromImage(
            canvas,
            scalefactor=1.0 / 128.0,
            size=(self.input_width, self.input_height),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False,
        )
        return blob.astype(np.float32), det_scale, (h, w)

    def _postprocess(self, net_outs, det_scale, original_size, conf_threshold):
        score_outputs, bbox_outputs = self._split_outputs(net_outs)
        all_boxes = []
        all_scores = []

        for stride in self.feat_strides:
            if stride not in score_outputs or stride not in bbox_outputs:
                continue

            scores = np.asarray(score_outputs[stride]).reshape(-1)
            bbox_preds = np.asarray(bbox_outputs[stride]).reshape(-1, 4) * stride

            height = self.input_height // stride
            width = self.input_width // stride
            anchor_count = max(1, int(scores.size / max(1, height * width)))
            anchor_centers = self._anchor_centers(height, width, stride, anchor_count)

            limit = min(anchor_centers.shape[0], bbox_preds.shape[0], scores.shape[0])
            if limit <= 0:
                continue

            scores = scores[:limit]
            bbox_preds = bbox_preds[:limit]
            anchor_centers = anchor_centers[:limit]
            positive = np.where(scores >= conf_threshold)[0]
            if positive.size == 0:
                continue

            boxes = self._distance_to_bbox(anchor_centers, bbox_preds)
            boxes = boxes[positive] / det_scale
            all_boxes.append(boxes)
            all_scores.append(scores[positive])

        if not all_boxes:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        boxes = np.vstack(all_boxes).astype(np.float32)
        scores = np.concatenate(all_scores).astype(np.float32)
        h, w = original_size
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w - 1)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h - 1)
        return boxes, scores

    def _split_outputs(self, net_outs) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        score_outputs: Dict[int, np.ndarray] = {}
        bbox_outputs: Dict[int, np.ndarray] = {}

        for name, output in zip(self.output_names, net_outs):
            lower_name = name.lower()
            for stride in self.feat_strides:
                stride_text = str(stride)
                if stride_text not in lower_name:
                    continue
                if any(token in lower_name for token in ('score', 'cls', 'conf')):
                    score_outputs[stride] = output
                elif any(token in lower_name for token in ('bbox', 'box')):
                    bbox_outputs[stride] = output

        if len(score_outputs) == 3 and len(bbox_outputs) == 3:
            return score_outputs, bbox_outputs

        # Common SCRFD output order: score_8, score_16, score_32, bbox_8, bbox_16, bbox_32, ...
        if len(net_outs) >= 6:
            score_outputs = {
                stride: net_outs[index]
                for index, stride in enumerate(self.feat_strides)
            }
            bbox_outputs = {
                stride: net_outs[index + len(self.feat_strides)]
                for index, stride in enumerate(self.feat_strides)
            }

        return score_outputs, bbox_outputs

    @staticmethod
    def _anchor_centers(height: int, width: int, stride: int, anchor_count: int) -> np.ndarray:
        anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        anchor_centers = (anchor_centers * stride).reshape((-1, 2))
        if anchor_count > 1:
            anchor_centers = np.stack([anchor_centers] * anchor_count, axis=1).reshape((-1, 2))
        return anchor_centers

    @staticmethod
    def _distance_to_bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, nms_threshold: float) -> List[int]:
        order = scores.argsort()[::-1]
        keep: List[int] = []

        while order.size > 0:
            index = int(order[0])
            keep.append(index)

            xx1 = np.maximum(boxes[index, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[index, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[index, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[index, 3], boxes[order[1:], 3])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            area_current = (boxes[index, 2] - boxes[index, 0] + 1) * (boxes[index, 3] - boxes[index, 1] + 1)
            area_others = (boxes[order[1:], 2] - boxes[order[1:], 0] + 1) * (
                boxes[order[1:], 3] - boxes[order[1:], 1] + 1
            )
            overlap = inter / (area_current + area_others - inter)
            remaining = np.where(overlap <= nms_threshold)[0]
            order = order[remaining + 1]

        return keep
