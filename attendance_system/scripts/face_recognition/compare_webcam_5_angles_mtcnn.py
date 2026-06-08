import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN

from apps.face_recognition.core.facekit.embedder_mobilefacenet_pytorch import MobileFaceNetFactory

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT


def choose_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


POSES = [
    ("FRONT", "Look straight at the camera"),
    ("IMAGE_LEFT", "Turn toward the LEFT side of the image"),
    ("IMAGE_RIGHT", "Turn toward the RIGHT side of the image"),
    ("UP", "Raise your chin"),
    ("DOWN", "Lower your chin"),
]

ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

BM6_CHECKPOINT = PROJECT_ROOT / "bm6.pth"


def load_mobilefacenet_model(device, checkpoint_path=None):
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else BM6_CHECKPOINT
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"MobileFaceNet checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model = MobileFaceNetFactory(128).model
    model.load_state_dict(checkpoint, strict=True)
    model.to(device).eval()
    print(f"Loaded MobileFaceNet checkpoint: {checkpoint_path}")
    return model


def model_input_size():
    return (112, 96)


@torch.no_grad()
def get_model_embedding(model, face_bgr, device, input_size):
    height, width = input_size
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (width, height)).astype(np.float32)
    rgb = (rgb - 127.5) / 128.0 if input_size == (112, 96) else (rgb / 255.0 - 0.5) / 0.5
    tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0).to(device=device, dtype=torch.float32)
    embedding = model(tensor)[0]
    return F.normalize(embedding, dim=0).cpu().numpy().astype(np.float32)


def load_mtcnn(device, min_face_size):
    return MTCNN(
        keep_all=True,
        device=device,
        min_face_size=min_face_size,
        post_process=False,
    )


def detect_faces_mtcnn(detector, frame_bgr, conf_threshold):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes, probs, landmarks = detector.detect(rgb, landmarks=True)
    if boxes is None:
        return []

    h, w = frame_bgr.shape[:2]
    detections = []
    for box, prob, points in zip(boxes, probs, landmarks):
        if prob is None or float(prob) < conf_threshold:
            continue

        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(
            {
                "box": (x1, y1, x2, y2),
                "confidence": float(prob),
                "landmarks": np.asarray(points, dtype=np.float32),
                "area": (x2 - x1) * (y2 - y1),
            }
        )

    detections.sort(key=lambda item: item["area"], reverse=True)
    return detections


def crop_face(frame_bgr, detection, margin_ratio):
    x1, y1, x2, y2 = detection["box"]
    h, w = frame_bgr.shape[:2]
    box = expand_box(x1, y1, x2, y2, w, h, margin_ratio=margin_ratio)
    x1, y1, x2, y2 = box
    face = frame_bgr[y1:y2, x1:x2]
    return (face, box) if face.size else (None, None)


def align_face(frame_bgr, landmarks, image_size=112, output_width=None):
    output_width = output_width or image_size
    template = ARCFACE_TEMPLATE_112.copy()
    template[:, 0] *= float(output_width) / 112.0
    template[:, 1] *= float(image_size) / 112.0
    transform, _ = cv2.estimateAffinePartial2D(
        np.asarray(landmarks, dtype=np.float32),
        template,
        method=cv2.LMEDS,
    )
    if transform is None:
        return None
    return cv2.warpAffine(
        frame_bgr,
        transform,
        (output_width, image_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def extract_face(frame_bgr, detection, args):
    if args.face_processing == "aligned":
        face = align_face(frame_bgr, detection["landmarks"], args.image_size)
        return face, detection["box"]
    return crop_face(frame_bgr, detection, args.margin)


def estimate_pose(landmarks, yaw_threshold=0.16, up_threshold=0.43, down_threshold=0.57):
    left_eye, right_eye, nose, left_mouth, right_mouth = landmarks
    eye_mid = (left_eye + right_eye) / 2.0
    mouth_mid = (left_mouth + right_mouth) / 2.0
    eye_distance = max(float(np.linalg.norm(right_eye - left_eye)), 1.0)
    face_height = max(float(mouth_mid[1] - eye_mid[1]), 1.0)

    center_x = float((eye_mid[0] + mouth_mid[0]) / 2.0)
    yaw = float((nose[0] - center_x) / eye_distance)
    pitch = float((nose[1] - eye_mid[1]) / face_height)

    if yaw < -yaw_threshold:
        pose = "IMAGE_LEFT"
    elif yaw > yaw_threshold:
        pose = "IMAGE_RIGHT"
    elif pitch < up_threshold:
        pose = "UP"
    elif pitch > down_threshold:
        pose = "DOWN"
    else:
        pose = "FRONT"
    return pose, yaw, pitch


def draw_detection(image, detection, color, label):
    x1, y1, x2, y2 = detection["box"]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    for point in detection["landmarks"]:
        cv2.circle(image, tuple(point.astype(int)), 2, color, -1)


def save_reference(face, save_dir, index, pose, processing):
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{index:02d}_{pose.lower()}_{processing}_{int(time.time())}"
    image_path = save_dir / f"{stem}.jpg"
    cv2.imwrite(str(image_path), face)
    return image_path


def capture_five_angles(args, detector):
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    references = []
    print("Capture 5 guided angles with MTCNN landmarks.")
    print("C=capture current angle | R=reset | Q=quit")

    try:
        while len(references) < len(POSES):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Cannot read camera frame.")

            display = frame.copy()
            detections = detect_faces_mtcnn(detector, frame, args.det_conf)
            expected_pose, instruction = POSES[len(references)]
            current_pose = "NO_FACE"
            yaw = pitch = 0.0

            if detections:
                current_pose, yaw, pitch = estimate_pose(
                    detections[0]["landmarks"],
                    yaw_threshold=args.yaw_threshold,
                    up_threshold=args.up_threshold,
                    down_threshold=args.down_threshold,
                )
                valid = current_pose == expected_pose or args.no_pose_check
                color = (0, 200, 0) if valid else (0, 200, 255)
                draw_detection(display, detections[0], color, f"{current_pose} y={yaw:+.2f} p={pitch:.2f}")
                for det in detections[1:]:
                    draw_detection(display, det, (120, 120, 120), f"{det['confidence']:.2f}")

            cv2.putText(
                display,
                f"Step {len(references) + 1}/5: {expected_pose}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
            )
            cv2.putText(display, instruction, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
            cv2.putText(display, "C=capture  R=reset  Q=quit", (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
            cv2.imshow("Capture 5 Face Angles - MTCNN", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise RuntimeError("Reference capture cancelled.")
            if key == ord("r"):
                references.clear()
                print("Reset all reference angles.")
                continue
            if key != ord("c"):
                continue
            if not detections:
                print("No face detected. Try again.")
                continue
            if len(detections) > 1:
                print("More than one face detected. Keep only your face in view.")
                continue
            if not args.no_pose_check and current_pose != expected_pose:
                print(f"Expected {expected_pose}, but MTCNN landmarks indicate {current_pose}. Adjust your pose.")
                continue

            face, _ = extract_face(frame, detections[0], args)
            if face is None:
                print("Face extraction/alignment failed. Try again.")
                continue

            path = save_reference(
                face,
                Path(args.save_dir),
                len(references) + 1,
                expected_pose,
                args.face_processing,
            )
            references.append(face)
            print(f"Captured {expected_pose}: {path}")
    finally:
        cap.release()
        cv2.destroyWindow("Capture 5 Face Angles - MTCNN")

    return references


def make_reference_embeddings(args, model, device, reference_faces):
    return np.stack(
        [get_model_embedding(model, face, device, args.model_input_size) for face in reference_faces],
        axis=0,
    )


def compare_realtime(args, model, detector, device, reference_embeddings, model_name):
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    print(f"\nRealtime comparison started | model: {model_name}")
    print("Q=quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Cannot read camera frame.")

            display = frame.copy()
            detections = detect_faces_mtcnn(detector, frame, args.det_conf)
            for detection in detections:
                face, box = extract_face(frame, detection, args)
                if face is None:
                    continue

                embedding = get_model_embedding(model, face, device, args.model_input_size)
                scores = np.asarray([cosine_similarity(ref, embedding) for ref in reference_embeddings])
                best_angle_index = int(np.argmax(scores))
                best_score = float(scores[best_angle_index])
                mean_score = float(np.mean(scores))
                same = best_score >= args.threshold
                color = (0, 200, 0) if same else (0, 0, 220)
                x1, y1, x2, y2 = box

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    display,
                    f"{'MATCH' if same else 'UNKNOWN'} best={best_score:.3f} mean={mean_score:.3f}",
                    (x1, max(20, y1 - 28)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    color,
                    2,
                )
                cv2.putText(
                    display,
                    f"Best ref: {POSES[best_angle_index][0]}",
                    (x1, max(20, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    color,
                    2,
                )

            cv2.putText(
                display,
                f"Model: {model_name} | refs: 5 | threshold: {args.threshold:.2f} | Q=quit",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Realtime Compare Against 5 References", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyWindow("Realtime Compare Against 5 References")


def compare_gallery(args, model, detector, device, reference_embeddings, model_name):
    gallery = cv2.imread(str(args.gallery))
    if gallery is None:
        raise FileNotFoundError(f"Cannot read gallery image: {args.gallery}")

    detections = detect_faces_mtcnn(detector, gallery, args.det_conf)
    if not detections:
        raise RuntimeError(f"No faces detected in gallery image: {args.gallery}")

    annotated = gallery.copy()
    rows = []
    for index, detection in enumerate(detections, start=1):
        face, box = extract_face(gallery, detection, args)
        if face is None:
            continue

        embedding = get_model_embedding(model, face, device, args.model_input_size)
        scores = np.asarray([cosine_similarity(ref, embedding) for ref in reference_embeddings])
        best_angle_index = int(np.argmax(scores))
        best_score = float(scores[best_angle_index])
        mean_score = float(np.mean(scores))
        same = best_score >= args.threshold
        color = (0, 200, 0) if same else (0, 0, 220)
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{index}: {'MATCH' if same else 'UNKNOWN'} {best_score:.3f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2,
        )
        rows.append((index, best_score, mean_score, same, POSES[best_angle_index][0]))

    rows.sort(key=lambda item: item[1], reverse=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), annotated)

    print(f"\nModel: {model_name} | processing: {args.face_processing}")
    print(f"Detected faces: {len(rows)} | threshold: {args.threshold:.3f}")
    print(f"Annotated output: {output}\n")
    print("Rank | Face | Best   | Mean   | Best angle  | Decision")
    print("-----+------+--------+--------+-------------+---------")
    for rank, (index, best, mean, same, angle) in enumerate(rows, start=1):
        print(f"{rank:>4} | {index:>4} | {best:>6.3f} | {mean:>6.3f} | {angle:<11} | {'MATCH' if same else 'UNKNOWN'}")

    if args.show:
        window_name = f"MTCNN Gallery Comparison - {model_name}"
        cv2.imshow(window_name, annotated)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)


def build_parser():
    default_save_dir = WORKSPACE_ROOT / "images" / "five_angle_references"
    default_output = WORKSPACE_ROOT / "images" / f"mtcnn_gallery_compare_{int(time.time())}.jpg"

    parser = argparse.ArgumentParser(
        description="Capture 5 guided face angles with MTCNN, then compare them against a gallery using MobileFaceNet."
    )
    parser.add_argument("--gallery", type=Path, default=None, help="Path to gallery/collage image.")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="After capturing 5 references, compare live webcam faces against them.",
    )
    parser.add_argument("--model", type=str, default=str(BM6_CHECKPOINT), help="MobileFaceNet checkpoint path.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument(
        "--face-processing",
        choices=["aligned", "crop"],
        default="aligned",
        help="Align with MTCNN landmarks or use an expanded detector crop.",
    )
    parser.add_argument("--threshold", type=float, default=0.45, help="Match when the best of 5 cosine scores reaches this value.")
    parser.add_argument("--det-conf", type=float, default=0.90, help="MTCNN confidence threshold.")
    parser.add_argument("--min-face-size", type=int, default=40)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--yaw-threshold", type=float, default=0.16)
    parser.add_argument("--up-threshold", type=float, default=0.43)
    parser.add_argument("--down-threshold", type=float, default=0.57)
    parser.add_argument("--no-pose-check", action="store_true", help="Allow manual capture even if pose estimation disagrees.")
    parser.add_argument("--save-dir", type=Path, default=default_save_dir)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main():
    args = build_parser().parse_args()
    if not args.realtime and args.gallery is None:
        raise SystemExit("Provide --gallery for image comparison, or use --realtime.")

    device = choose_device(args.device)
    detector = load_mtcnn(device, args.min_face_size)
    reference_faces = capture_five_angles(args, detector)

    args.model_input_size = model_input_size()
    model = load_mobilefacenet_model(device, args.model)
    references = make_reference_embeddings(args, model, device, reference_faces)
    if args.realtime:
        compare_realtime(args, model, detector, device, references, "bm6")
    else:
        compare_gallery(args, model, detector, device, references, "bm6")


if __name__ == "__main__":
    main()
