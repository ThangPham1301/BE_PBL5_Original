from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def lazy_import_cv2():
    import cv2  # type: ignore

    return cv2


def lazy_import_torch():
    import torch  # type: ignore

    return torch


def lazy_import_mtcnn():
    from facenet_pytorch import MTCNN  # type: ignore

    return MTCNN


@dataclass(frozen=True)
class MTCNNFace:
    box: Tuple[int, int, int, int]
    confidence: float
    landmarks: np.ndarray
    aligned_bgr: np.ndarray

    @property
    def w(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def h(self) -> int:
        return self.box[3] - self.box[1]


@dataclass(frozen=True)
class FaceValidation:
    is_clear: bool
    message: str
    metrics: Dict[str, float]
    face: Optional[MTCNNFace] = None


class MTCNNArcFaceAligner:
    """MTCNN 5-point landmark detector with ArcFace-style similarity alignment."""

    ARCFACE_112_LANDMARKS = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )

    def __init__(self, image_size: int = 112, output_width: int = 96, margin: int = 10, min_face_size: int = 40):
        self.cv2 = lazy_import_cv2()
        torch = lazy_import_torch()
        MTCNN = lazy_import_mtcnn()
        self.image_size = int(image_size)
        self.output_width = int(output_width)
        self.margin = int(margin)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(
            image_size=self.image_size,
            margin=self.margin,
            min_face_size=min_face_size,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=False,
            device=self.device,
        )

    def detect(self, bgr: np.ndarray, conf_threshold: float = 0.9) -> List[MTCNNFace]:
        if bgr is None or bgr.size == 0:
            return []

        rgb = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
        boxes, probs, points = self.mtcnn.detect(rgb, landmarks=True)
        if boxes is None or probs is None or points is None:
            return []

        h, w = bgr.shape[:2]
        faces: List[MTCNNFace] = []
        for box, prob, landmarks in zip(boxes, probs, points):
            if prob is None or float(prob) < conf_threshold:
                continue

            x1, y1, x2, y2 = box.astype(int).tolist()
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            landmarks = np.asarray(landmarks, dtype=np.float32)
            aligned = self.align_bgr(bgr, landmarks)
            if aligned.size == 0:
                continue

            faces.append(
                MTCNNFace(
                    box=(x1, y1, x2, y2),
                    confidence=float(prob),
                    landmarks=landmarks,
                    aligned_bgr=aligned,
                )
            )

        faces.sort(key=lambda face: face.confidence * face.w * face.h, reverse=True)
        return faces

    def align_bgr(self, bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        dst = self.ARCFACE_112_LANDMARKS.copy()
        dst[:, 0] *= float(self.output_width) / 112.0
        dst[:, 1] *= float(self.image_size) / 112.0

        matrix, _ = self.cv2.estimateAffinePartial2D(
            landmarks.astype(np.float32),
            dst,
            method=self.cv2.LMEDS,
        )
        if matrix is None:
            return np.empty((0, 0, 3), dtype=np.uint8)

        return self.cv2.warpAffine(
            bgr,
            matrix,
            (self.output_width, self.image_size),
            flags=self.cv2.INTER_LINEAR,
            borderValue=0.0,
        )

    def validate(
        self,
        bgr: np.ndarray,
        expected_pose: Optional[str] = None,
        conf_threshold: float = 0.9,
        min_face_area_ratio: float = 0.08,
        min_blur_score: float = 60.0,
    ) -> FaceValidation:
        faces = self.detect(bgr, conf_threshold=conf_threshold)
        if not faces:
            return FaceValidation(False, "Chưa thấy khuôn mặt rõ trong khung.", {})

        if len(faces) > 1:
            return FaceValidation(False, "Chỉ để một khuôn mặt trong khung hình.", {"face_count": float(len(faces))})

        face = faces[0]
        h, w = bgr.shape[:2]
        frame_area = float(h * w)
        face_area_ratio = float(face.w * face.h) / frame_area if frame_area > 0 else 0.0

        gray = self.cv2.cvtColor(face.aligned_bgr, self.cv2.COLOR_BGR2GRAY)
        blur_score = float(self.cv2.Laplacian(gray, self.cv2.CV_64F).var())

        center_x = (face.box[0] + face.box[2]) / 2.0 / max(1, w)
        center_y = (face.box[1] + face.box[3]) / 2.0 / max(1, h)
        yaw, pitch = self._estimate_pose(face.landmarks)
        metrics = {
            "confidence": round(face.confidence, 4),
            "face_area_ratio": round(face_area_ratio, 4),
            "blur_score": round(blur_score, 2),
            "center_x": round(center_x, 4),
            "center_y": round(center_y, 4),
            "yaw": round(yaw, 4),
            "pitch": round(pitch, 4),
        }

        if face_area_ratio < min_face_area_ratio:
            return FaceValidation(False, "Khuôn mặt còn nhỏ, đưa camera lại gần hơn.", metrics, face)

        if not (0.28 <= center_x <= 0.72 and 0.24 <= center_y <= 0.78):
            return FaceValidation(False, "Đưa khuôn mặt vào giữa vòng tròn.", metrics, face)

        if blur_score < min_blur_score:
            return FaceValidation(False, "Ảnh bị mờ, giữ yên và tăng ánh sáng.", metrics, face)

        pose_ok, pose_message = self._validate_pose(expected_pose, yaw, pitch)
        if not pose_ok:
            return FaceValidation(False, pose_message, metrics, face)

        return FaceValidation(True, "Khuôn mặt hợp lệ, có thể lưu ảnh này.", metrics, face)

    @staticmethod
    def _estimate_pose(landmarks: np.ndarray) -> Tuple[float, float]:
        left_eye, right_eye, nose, mouth_left, mouth_right = landmarks
        eye_mid = (left_eye + right_eye) / 2.0
        mouth_mid = (mouth_left + mouth_right) / 2.0
        eye_dist = float(np.linalg.norm(right_eye - left_eye))
        face_vertical = max(1.0, float(mouth_mid[1] - eye_mid[1]))

        yaw = float((nose[0] - eye_mid[0]) / max(1.0, eye_dist))
        pitch = float((nose[1] - eye_mid[1]) / face_vertical)
        return yaw, pitch

    @staticmethod
    def _validate_pose(expected_pose: Optional[str], yaw: float, pitch: float) -> Tuple[bool, str]:
        if expected_pose == "front":
            if abs(yaw) > 0.28:
                return False, "Nhìn thẳng vào camera để chụp góc chính diện."
            return True, ""

        if expected_pose == "left" and abs(yaw) < 0.06:
            return False, "Quay nhẹ mặt sang trái rồi chụp lại."

        if expected_pose == "right" and abs(yaw) < 0.06:
            return False, "Quay nhẹ mặt sang phải rồi chụp lại."

        if expected_pose == "up" and pitch > 0.55:
            return False, "Ngửa mặt lên nhẹ rồi chụp lại."

        if expected_pose == "down" and pitch < 0.45:
            return False, "Cúi mặt xuống nhẹ rồi chụp lại."

        return True, ""
