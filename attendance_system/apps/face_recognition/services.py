from pathlib import Path
import time
from uuid import uuid4
import numpy as np
import cv2
from django.conf import settings
from .models import FaceEmbedding
from apps.employees.models import Employee
from .core.facekit.embedder_mobilefacenet_arcface import MobileFaceNetArcFaceEmbedder
from .core.facekit.embedder_mobilefacenet_pytorch import MobileFaceNetPyTorchEmbedder
from .core.facekit.detector_resnet10 import ResNet10FaceDetector
from .core.facekit.detector_scrfd import SCRFDFaceDetector
from .core.facekit.vision_utils import square_crop

# Define paths relative to this file
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'core' / 'models'
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
        self.detector = None
        self.embedder = None
        self.last_error = None
        
        # 1. Load Detector. Prefer SCRFD; keep ResNet10 as a fallback if the
        # SCRFD ONNX file has not been added to core/models yet.
        try:
            scrfd_model = Path(getattr(settings, 'FACE_DETECTOR_MODEL', 'scrfd_2.5g_bnkps.onnx'))
            if not scrfd_model.is_absolute():
                scrfd_model = MODELS_DIR / scrfd_model

            if scrfd_model.exists():
                self.detector = SCRFDFaceDetector(
                    scrfd_model,
                    input_size=getattr(settings, 'FACE_SCRFD_INPUT_SIZE', 640),
                )
                print(f"Loaded SCRFD face detector: {scrfd_model}")
            else:
                print(f"Warning: SCRFD detector model not found at {scrfd_model}")
        except Exception as e:
            print(f"Error loading SCRFD detector: {e}")

        if self.detector is None:
            try:
                prototxt = MODELS_DIR / 'deploy.prototxt'
                caffemodel = MODELS_DIR / 'res10_300x300_ssd_iter_140000_fp16.caffemodel'

                if prototxt.exists() and caffemodel.exists():
                    self.detector = ResNet10FaceDetector(prototxt, caffemodel)
                    print(f"Loaded fallback ResNet10 face detector: {caffemodel}")
                else:
                    print(f"Warning: Fallback detector models not found at {MODELS_DIR}")
            except Exception as e:
                print(f"Error loading fallback detector: {e}")

        # 2. Load Embedder. Prefer the configured embedding-only PyTorch model.
        try:
            pytorch_path = Path(getattr(settings, 'FACE_EMBEDDER_MODEL', 'lfw-bm2-backbone.pth'))
            if not pytorch_path.is_absolute():
                pytorch_path = PROJECT_DIR / pytorch_path

            if pytorch_path.exists():
                self.embedder = MobileFaceNetPyTorchEmbedder(pytorch_path)
                print(f"Loaded PyTorch embedding model: {pytorch_path}")
            else:
                print(f"Warning: PyTorch embedding model not found at {pytorch_path}")
        except Exception as e:
            print(f"Error loading PyTorch embedder: {e}")

        if self.embedder is None:
            try:
                onnx_path = MODELS_DIR / 'mobilefacenet_arcface.onnx'
                if onnx_path.exists():
                    self.embedder = MobileFaceNetArcFaceEmbedder(onnx_path)
                    print(f"Loaded ONNX fallback embedding model: {onnx_path}")
                else:
                    print(f"Warning: Embedder model not found at {MODELS_DIR}")
            except Exception as e:
                print(f"Error loading ONNX embedder: {e}")
            
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

    def _decode_face_crop(self, image_bytes):
        """Decode raw image bytes and return the detected face crop."""
        if self.detector is None:
            self.initialize()

        if self.detector is None:
            raise ValueError("Face detector unavailable")

        nparr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Invalid image")

        # Registration should be more tolerant than recognition matching.
        reg_threshold = getattr(settings, 'FACE_REGISTER_DETECT_THRESHOLD', 0.45)
        detections = self.detector.detect(frame, conf_threshold=reg_threshold)

        # Fallback: upscale once for far-face / low-detail frames.
        if not detections:
            try:
                h, w = frame.shape[:2]
                upscaled = cv2.resize(frame, (int(w * 1.5), int(h * 1.5)), interpolation=cv2.INTER_CUBIC)
                detections = self.detector.detect(upscaled, conf_threshold=max(0.35, reg_threshold - 0.1))
                if detections:
                    frame = upscaled
            except Exception:
                detections = []

        if not detections:
            raise ValueError("No face detected")

        best_face = max(detections, key=lambda d: d.w * d.h)
        x1, y1, x2, y2 = best_face.as_tuple()
        face_crop = square_crop(frame, (x1, y1, x2, y2), scale=1.25)

        if face_crop.size == 0:
            raise ValueError("Invalid face crop")

        return face_crop

    def upload_face_image_to_cloudinary(self, image_bytes, employee_id):
        """Detect, crop and upload face image to Cloudinary, returning the secure URL."""
        try:
            import cloudinary
            import cloudinary.uploader
        except Exception as exc:
            raise ValueError(f"Cloudinary package not available: {exc}")

        if not (
            settings.CLOUDINARY_CLOUD_NAME
            and settings.CLOUDINARY_API_KEY
            and settings.CLOUDINARY_API_SECRET
        ):
            raise ValueError("Cloudinary chưa được cấu hình đầy đủ")

        face_crop = self._decode_face_crop(image_bytes)
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

    def extract_embedding_from_bytes(self, image_bytes):
        """Return a face embedding vector from raw image bytes."""
        if self.embedder is None:
            self.initialize()

        if self.embedder is None:
            raise ValueError("Face embedder unavailable")

        face_crop = self._decode_face_crop(image_bytes)
        return self.embedder.embed_face_bgr(face_crop)

    def _load_embeddings(self):
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
                if emb_array.shape != (512,):
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
            self.cached_embeddings = np.zeros((0, 512), dtype=np.float32)
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
        threshold = getattr(settings, 'FACE_MATCH_THRESHOLD', 0.4) if match_threshold is None else match_threshold

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
        if self.detector is None or self.embedder is None:
            self.initialize()
            if self.detector is None or self.embedder is None:
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

        # 1. Detect faces. Check-in images can come from external cameras, so allow
        # a lower threshold and one upscale fallback when requested by the caller.
        threshold = 0.5 if detect_threshold is None else detect_threshold
        detections = self.detector.detect(frame, conf_threshold=threshold)

        if not detections and detect_threshold is not None:
            try:
                h, w = frame.shape[:2]
                upscaled = cv2.resize(frame, (int(w * 1.5), int(h * 1.5)), interpolation=cv2.INTER_CUBIC)
                detections = self.detector.detect(upscaled, conf_threshold=max(0.2, threshold - 0.1))
                if detections:
                    frame = upscaled
            except Exception:
                detections = []
        
        if not detections:
            print(f"[FACE CHECK-IN] No face detected. frame_shape={frame.shape}, threshold={threshold}")
            self.last_error = "no_face_detected"
            return False, None, 0.0
            
        # Get largest face and crop it
        best_face = max(detections, key=lambda d: d.w * d.h)
        x1, y1, x2, y2 = best_face.as_tuple()
        
        # Use square_crop as in original logic (better for MobileFaceNet)
        face_crop = square_crop(frame, (x1, y1, x2, y2), scale=1.25)
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
            raise ValueError(f"Employee {employee_id} not found")

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
