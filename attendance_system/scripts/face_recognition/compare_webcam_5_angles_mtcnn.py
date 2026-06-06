import argparse
import copy
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import MTCNN

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

FRIEND_CHECKPOINT = PROJECT_ROOT / "last_checkpoint.pth"


class FriendConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super().__init__()
        self.linear = linear
        self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp if dw else 1, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x if self.linear else self.prelu(x)


class FriendBottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super().__init__()
        self.connect = stride == 1 and inp == oup
        self.conv = nn.Sequential(
            FriendConvBlock(inp, inp * expansion, 1, 1, 0),
            FriendConvBlock(inp * expansion, inp * expansion, 3, stride, 1, dw=True),
            FriendConvBlock(inp * expansion, oup, 1, 1, 0, linear=True),
        )

    def forward(self, x):
        return x + self.conv(x) if self.connect else self.conv(x)


class FriendMobileFaceNet(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()
        self.conv1 = FriendConvBlock(3, 64, 3, 2, 1)
        self.dw_conv1 = FriendConvBlock(64, 64, 3, 1, 1, dw=True)
        inplanes = 64
        layers = []
        for expansion, channels, count, stride in [
            [2, 64, 5, 2],
            [4, 128, 1, 2],
            [2, 128, 6, 1],
            [4, 128, 1, 2],
            [2, 128, 2, 1],
        ]:
            for index in range(count):
                layers.append(FriendBottleneck(inplanes, channels, stride if index == 0 else 1, expansion))
                inplanes = channels
        self.blocks = nn.Sequential(*layers)
        self.conv2 = FriendConvBlock(128, 512, 1, 1, 0)
        self.linear7 = FriendConvBlock(512, 512, (7, 6), 1, 0, dw=True, linear=True)
        self.linear1 = FriendConvBlock(512, embedding_size, 1, 1, 0, linear=True)

    def forward(self, x):
        x = self.linear1(self.linear7(self.conv2(self.blocks(self.dw_conv1(self.conv1(x))))))
        return F.normalize(x.view(x.size(0), -1), dim=1)


def load_friend_model(device):
    if not FRIEND_CHECKPOINT.exists():
        raise FileNotFoundError(f"Friend checkpoint not found: {FRIEND_CHECKPOINT}")
    checkpoint = torch.load(str(FRIEND_CHECKPOINT), map_location="cpu", weights_only=False)
    model = FriendMobileFaceNet(embedding_size=128)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    print(f"Loaded friend pipeline checkpoint: {FRIEND_CHECKPOINT} | epoch={checkpoint.get('epoch')}")
    return model


def model_input_size(model_version):
    return (112, 96) if model_version == "friend" else (112, 112)


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
    if args.all_models:
        output = output.with_name(f"{output.stem}_{model_name}{output.suffix}")
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
    parser.add_argument(
        "--model-version",
        choices=["new", "bm2", "bm3", "lm", "lm2", "old", "pretrained", "friend"],
        default="new",
    )
    parser.add_argument("--model", type=str, default=None, help="MobileFaceNet checkpoint path. Overrides --model-version.")
    parser.add_argument("--embedding-size", type=int, default=512)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument(
        "--face-processing",
        choices=["aligned", "crop"],
        default="aligned",
        help="Align with MTCNN landmarks or use an expanded detector crop.",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run the same captured references through all configured checkpoints, including the friend's 128-d model.",
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
    if args.realtime and args.all_models:
        raise SystemExit("--realtime uses one model at a time. Choose --model-version instead of --all-models.")

    device = choose_device(args.device)
    detector = load_mtcnn(device, args.min_face_size)
    reference_faces = capture_five_angles(args, detector)

    model_versions = ["new", "bm2", "bm3", "lm", "lm2", "old", "pretrained", "friend"] if args.all_models else [args.model_version]
    for model_version in model_versions:
        model_args = copy.copy(args)
        model_args.model_version = model_version
        model_args.model_input_size = model_input_size(model_version)
        if args.all_models:
            model_args.model = None
        model = load_friend_model(device) if model_version == "friend" else load_recognition_model(
            resolve_model_path(model_args), args.embedding_size, device
        )
        references = make_reference_embeddings(model_args, model, device, reference_faces)
        if args.realtime:
            compare_realtime(model_args, model, detector, device, references, model_version)
        else:
            compare_gallery(model_args, model, detector, device, references, model_version)


if __name__ == "__main__":
    main()
