from __future__ import annotations

import argparse
import os
import time
from typing import Dict, Optional, Tuple

import cv2
import django


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.dev')
os.environ.setdefault('FACE_EMBEDDER_MODEL', 'lfw-bm2-backbone.pth')
django.setup()

from django.conf import settings  # noqa: E402

from apps.employees.models import Employee  # noqa: E402
from apps.face_recognition.services import FaceRecognitionService  # noqa: E402
from apps.face_recognition.core.facekit.vision_utils import square_crop  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description='Open laptop camera and recognize registered employees.')
    parser.add_argument('--camera', type=int, default=0, help='Camera index. Default: 0')
    parser.add_argument('--detect-threshold', type=float, default=0.45, help='Face detection confidence threshold.')
    parser.add_argument(
        '--match-threshold',
        type=float,
        default=float(getattr(settings, 'FACE_MATCH_THRESHOLD', 0.35)),
        help='Face match confidence threshold.',
    )
    parser.add_argument('--width', type=int, default=1280, help='Requested camera width.')
    parser.add_argument('--height', type=int, default=720, help='Requested camera height.')
    return parser.parse_args()


def employee_label(employee: Optional[Employee], confidence: float) -> str:
    if employee is None:
        return f'Unknown {confidence:.2f}'

    name = employee.user.get_full_name() or employee.user.username
    return f'{employee.employee_id} - {name} {confidence:.2f}'


def resolve_employee(employee_id: int, cache: Dict[int, Optional[Employee]]) -> Optional[Employee]:
    if employee_id not in cache:
        try:
            cache[employee_id] = Employee.objects.select_related('user').get(pk=employee_id, is_active=True)
        except Employee.DoesNotExist:
            cache[employee_id] = None
    return cache[employee_id]


def recognize_face(service: FaceRecognitionService, face_bgr, match_threshold: float) -> Tuple[Optional[int], float]:
    embedding = service.embedder.embed_face_bgr(face_bgr)
    return service.match_embedding(embedding, match_threshold=match_threshold)


def draw_label(frame, text: str, x1: int, y1: int, color):
    y_text = max(22, y1 - 8)
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    cv2.rectangle(frame, (x1, y_text - text_h - baseline - 6), (x1 + text_w + 8, y_text + 4), color, -1)
    cv2.putText(frame, text, (x1 + 4, y_text - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)


def main():
    args = parse_args()
    service = FaceRecognitionService()

    if service.detector is None or service.embedder is None:
        raise RuntimeError('Face detector/embedder unavailable. Check model files and dependencies.')

    print(f'Detector: {type(service.detector).__name__}')
    print(f'Embedder: {type(service.embedder).__name__}')
    print(f'Embedder model: {getattr(service.embedder, "model_path", "ONNX fallback")}')
    print(f'Match threshold: {args.match_threshold}')
    print('Press q or ESC to quit.')

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera index {args.camera}')

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    employee_cache: Dict[int, Optional[Employee]] = {}
    last_error = ''
    fps = 0.0
    last_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                last_error = 'Cannot read camera frame'
                continue

            detections = service.detector.detect(frame, conf_threshold=args.detect_threshold)

            for face in detections:
                x1, y1, x2, y2 = face.as_tuple()
                face_crop = square_crop(frame, (x1, y1, x2, y2), scale=1.25)

                employee = None
                confidence = 0.0
                color = (0, 0, 255)

                if face_crop.size > 0:
                    try:
                        employee_id, confidence = recognize_face(service, face_crop, args.match_threshold)
                        if employee_id is not None:
                            employee = resolve_employee(employee_id, employee_cache)
                            color = (0, 180, 80)
                        last_error = ''
                    except Exception as exc:
                        last_error = f'Recognition error: {exc}'

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                draw_label(frame, employee_label(employee, confidence), x1, y1, color)

            now = time.time()
            elapsed = max(0.001, now - last_time)
            fps = (fps * 0.85) + ((1.0 / elapsed) * 0.15)
            last_time = now

            cv2.putText(frame, f'FPS: {fps:.1f}', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
            cv2.putText(
                frame,
                f'Faces: {len(detections)}',
                (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
            )
            if last_error:
                cv2.putText(frame, last_error[:90], (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 255), 2)

            cv2.imshow('PBL5 Face Recognition', frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
