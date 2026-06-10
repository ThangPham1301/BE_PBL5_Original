from pathlib import Path
import time
from uuid import uuid4
import numpy as np
import cv2
from django.conf import settings
from .models import FaceEmbedding
from apps.employees.models import Employee
from .core.facekit.embedder_mobilefacenet_pytorch import MobileFaceNetPyTorchEmbedder
from .core.facekit.detector_mtcnn import MTCNNArcFaceAligner

# Define paths relative to this file
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent

class FaceRecognitionService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FaceRecognitionService, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        print("Initializing FaceRecognitionService...")
        self.registration_detector = None
        self.embedder = None
        self.last_error = None

        # MTCNN is the only detector used by the project. It provides landmarks
        # and aligns faces to the 112x96 MobileFaceNet input geometry.
        try:
            self.registration_detector = MTCNNArcFaceAligner(
                image_size=getattr(settings, 'FACE_MTCNN_IMAGE_SIZE', 112),
                output_width=getattr(settings, 'FACE_MTCNN_IMAGE_WIDTH', 96),
                margin=getattr(settings, 'FACE_MTCNN_MARGIN', 10),
                min_face_size=getattr(settings, 'FACE_MTCNN_MIN_FACE_SIZE', 40),
            )
            print("Loaded MTCNN registration detector with ArcFace landmark alignment")
        except Exception as e:
            print(f"Error loading MTCNN registration detector: {e}")

        # MobileFaceNet is the only embedder. The default checkpoint is bm6.pth.
        try:
            pytorch_path = Path(getattr(settings, 'FACE_EMBEDDER_MODEL', 'bm6.pth'))
            if not pytorch_path.is_absolute():
                pytorch_path = PROJECT_DIR / pytorch_path

            if pytorch_path.exists():
                self.embedder = MobileFaceNetPyTorchEmbedder(
                    pytorch_path,
                    embedding_size=getattr(settings, 'FACE_EMBEDDING_SIZE', 128),
                )
                print(f"Loaded PyTorch embedding model: {pytorch_path}")
            else:
                print(f"Warning: PyTorch embedding model not found at {pytorch_path}")
        except Exception as e:
            print(f"Error loading PyTorch embedder: {e}")

        self.cached_embeddings = None
        self.cached_labels = None
        self.last_cache_update = 0
        self.CACHE_TTL = getattr(settings, 'FACE_CACHE_TTL', 30)

    def _refresh_cache_if_needed(self):
        if self.cached_embeddings is None or (time.time() - self.last_cache_update > self.CACHE_TTL):
            self._load_embeddings()

    def invalidate_cache(self):
        self.cached_embeddings = None
        self.cached_labels = None
        self.last_cache_update = 0

    @staticmethod
    def _normalize_embedding(embedding):
        arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        return arr.astype(np.float32)

    @staticmethod
    def _decode_image_bytes(image_bytes):
        nparr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Ảnh không hợp lệ.")

        return frame

    def _decode_face_crop(self, image_bytes, expected_pose=None, require_pose=False):
        """Decode raw image bytes and return an MTCNN/ArcFace aligned face."""
        if self.registration_detector is None:
            self.initialize()

        if self.registration_detector is None:
            raise ValueError("Bộ phát hiện khuôn mặt MTCNN chưa sẵn sàng.")

        frame = self._decode_image_bytes(image_bytes)
        validation = self.registration_detector.validate(
            frame,
            expected_pose=expected_pose if require_pose else None,
            conf_threshold=getattr(settings, 'FACE_REGISTER_DETECT_THRESHOLD', 0.9),
            min_face_area_ratio=getattr(settings, 'FACE_REGISTER_MIN_AREA_RATIO', 0.08),
            min_blur_score=getattr(settings, 'FACE_REGISTER_MIN_BLUR_SCORE', 60.0),
        )
        if not validation.is_clear or validation.face is None:
            raise ValueError(validation.message)

        face_crop = validation.face.aligned_bgr

        return face_crop

    def validate_registration_image(self, image_bytes, expected_pose=None):
        if self.registration_detector is None:
            self.initialize()

        if self.registration_detector is None:
            return {
                'is_clear': False,
                'message': 'Bộ detect MTCNN chưa sẵn sàng',
                'metrics': {},
                'face_box': None,
                'landmarks': [],
                'frame_size': None,
            }

        frame = self._decode_image_bytes(image_bytes)
        frame_h, frame_w = frame.shape[:2]
        validation = self.registration_detector.validate(
            frame,
            expected_pose=expected_pose,
            conf_threshold=getattr(settings, 'FACE_REGISTER_DETECT_THRESHOLD', 0.9),
            min_face_area_ratio=getattr(settings, 'FACE_REGISTER_MIN_AREA_RATIO', 0.08),
            min_blur_score=getattr(settings, 'FACE_REGISTER_MIN_BLUR_SCORE', 60.0),
        )
        face_box = None
        landmarks = []
        if validation.face is not None:
            x1, y1, x2, y2 = validation.face.box
            face_box = {
                'x': round(x1 / max(1, frame_w), 4),
                'y': round(y1 / max(1, frame_h), 4),
                'width': round((x2 - x1) / max(1, frame_w), 4),
                'height': round((y2 - y1) / max(1, frame_h), 4),
            }
            landmarks = [
                {
                    'x': round(float(point[0]) / max(1, frame_w), 4),
                    'y': round(float(point[1]) / max(1, frame_h), 4),
                }
                for point in validation.face.landmarks
            ]

        return {
            'is_clear': validation.is_clear,
            'message': validation.message,
            'metrics': validation.metrics,
            'face_box': face_box,
            'landmarks': landmarks,
            'frame_size': {
                'width': int(frame_w),
                'height': int(frame_h),
            },
        }

    def upload_face_image_to_cloudinary(self, image_bytes, employee_id, expected_pose=None):
        """Detect, crop and upload face image to Cloudinary, returning the secure URL."""
        try:
            import cloudinary
            import cloudinary.uploader
        except Exception as exc:
            raise ValueError(f"Gói Cloudinary chưa sẵn sàng: {exc}")

        if not (
            settings.CLOUDINARY_CLOUD_NAME
            and settings.CLOUDINARY_API_KEY
            and settings.CLOUDINARY_API_SECRET
        ):
            raise ValueError("Cloudinary chưa được cấu hình đầy đủ")

        face_crop = self._decode_face_crop(image_bytes, expected_pose=expected_pose, require_pose=False)
        ok, encoded = cv2.imencode('.jpg', face_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise ValueError("Không thể mã hóa ảnh khuôn mặt")

        cloudinary.config(
            cloud_name=settings.CLOUDINARY_CLOUD_NAME,
            api_key=settings.CLOUDINARY_API_KEY,
            api_secret=settings.CLOUDINARY_API_SECRET,
            secure=True,
        )

        public_id = f"emp_{employee_id}_{int(time.time() * 1000)}_{uuid4().hex[:8]}"
        upload_result = cloudinary.uploader.upload(
            encoded.tobytes(),
            folder=settings.FACE_CLOUDINARY_FOLDER,
            public_id=public_id,
            resource_type='image',
            overwrite=False,
        )

        url = upload_result.get('secure_url') or upload_result.get('url')
        if not url:
            raise ValueError("Upload Cloudinary thành công nhưng không nhận được URL")

        return url

    def extract_embedding_from_bytes(self, image_bytes, expected_pose=None, require_pose=False):
        """Return a face embedding vector from raw image bytes."""
        if self.embedder is None:
            self.initialize()

        if self.embedder is None:
            raise ValueError("Bộ trích xuất đặc trưng khuôn mặt chưa sẵn sàng.")

        face_crop = self._decode_face_crop(
            image_bytes,
            expected_pose=expected_pose,
            require_pose=require_pose,
        )
        return self.embedder.embed_face_bgr(face_crop)

    def _load_embeddings(self):
        embedding_size = int(getattr(self.embedder, 'embedding_size', getattr(settings, 'FACE_EMBEDDING_SIZE', 128)))
        embeddings = (
            FaceEmbedding.objects
            .select_related('employee')
            .filter(embedding__isnull=False, employee__is_active=True)
        )

        emb_list = []
        lbl_list = []

        for face in embeddings:
            try:
                if not face.embedding:
                    continue

                emb_array = np.frombuffer(face.embedding, dtype=np.float32)
                if emb_array.shape != (embedding_size,):
                    print(f"[FACE DB] Skip embedding {face.id}: invalid shape {emb_array.shape}")
                    continue

                emb_list.append(self._normalize_embedding(emb_array))
                lbl_list.append(face.employee_id)
            except Exception as exc:
                print(f"[FACE DB] Error loading embedding {face.id}: {exc}")

        if emb_list:
            self.cached_embeddings = np.vstack(emb_list).astype(np.float32)
            self.cached_labels = np.asarray(lbl_list, dtype=np.int64)
        else:
            self.cached_embeddings = np.zeros((0, embedding_size), dtype=np.float32)
            self.cached_labels = np.asarray([], dtype=np.int64)

        self.last_cache_update = time.time()
        print(f"[FACE DB] Loaded {self.cached_embeddings.shape[0]} embeddings from PostgreSQL.")

    def match_embedding(self, embedding, match_threshold=None):
        self._refresh_cache_if_needed()

        if self.cached_embeddings.shape[0] == 0:
            return None, 0.0

        embedding = self._normalize_embedding(embedding)
        scores = self.cached_embeddings @ embedding
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        threshold = getattr(settings, 'FACE_MATCH_THRESHOLD', 0.65) if match_threshold is None else match_threshold

        print(
            f"[FACE DB] Best match score={best_score:.4f} "
            f"threshold={threshold} cache_size={self.cached_embeddings.shape[0]}"
        )

        if best_score > threshold:
            return int(self.cached_labels[best_idx]), best_score

        return None, best_score

    def process_image(self, image_bytes, detect_threshold=None):
        """
        Process uploaded image bytes (full frame or crop).
        Returns: Tuple(has_face, employee_id, confidence_score/distance)
        """
        self.last_error = None
        if self.registration_detector is None or self.embedder is None:
            self.initialize()
            if self.registration_detector is None or self.embedder is None:
                print("Face models unavailable.")
                self.last_error = "models_unavailable"
                return False, None, 0.0
            
        # Decode image
        nparr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            print(f"[FACE CHECK-IN] Invalid image bytes. bytes={len(image_bytes) if image_bytes else 0}")
            self.last_error = "invalid_image"
            return False, None, 0.0

        # 1. Detect and align with MTCNN, matching the registration pipeline.
        threshold = 0.5 if detect_threshold is None else detect_threshold
        face_crop = None

        mtcnn_threshold = getattr(settings, 'FACE_RECOGNITION_MTCNN_THRESHOLD', 0.85)
        faces = self.registration_detector.detect(frame, conf_threshold=mtcnn_threshold)
        if not faces and detect_threshold is not None:
            try:
                h, w = frame.shape[:2]
                upscaled = cv2.resize(frame, (int(w * 1.5), int(h * 1.5)), interpolation=cv2.INTER_CUBIC)
                faces = self.registration_detector.detect(upscaled, conf_threshold=max(0.2, threshold - 0.1))
                if faces:
                    frame = upscaled
            except Exception:
                faces = []

        if not faces:
            print(f"[FACE CHECK-IN] No face detected by MTCNN. frame_shape={frame.shape}, threshold={mtcnn_threshold}")
            self.last_error = "no_face_detected"
            return False, None, 0.0

        face_crop = faces[0].aligned_bgr

        if face_crop.size == 0:
            print("[FACE CHECK-IN] Invalid face crop.")
            self.last_error = "invalid_face_crop"
            return False, None, 0.0

        # 2. Embed
        embedding = self.embedder.embed_face_bgr(face_crop)
        
        # 3. Match against embeddings stored in PostgreSQL.
        employee_id, best_score = self.match_embedding(embedding)
        return True, employee_id, float(best_score)

    def enroll_face(self, image_bytes, employee_id):
        """
        Embed and save face for employee.
        """
        embedding = self.extract_embedding_from_bytes(image_bytes)
        
        try:
            emp = Employee.objects.get(pk=employee_id)
        except Employee.DoesNotExist:
            raise ValueError(f"Không tìm thấy nhân viên {employee_id}.")

        employee_name = emp.user.get_full_name() or emp.user.username
        embedding = self._normalize_embedding(embedding)

        FaceEmbedding.objects.create(
            employee=emp,
            embedding=embedding.tobytes(),
            employee_code_snapshot=emp.employee_id,
            employee_name_snapshot=employee_name,
        )
        self.invalidate_cache()

        return True
