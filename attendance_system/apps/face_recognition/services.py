from pathlib import Path
import time
from uuid import uuid4
import numpy as np
import cv2
from django.conf import settings
from .models import FaceEmbedding
from apps.employees.models import Employee
from .core.facekit.embedder_mobilefacenet_pytorch import MobileFaceNetTorchEmbedder
from .core.facekit.detector_mtcnn import MTCNNFaceDetector
from .core.facekit.vision_utils import square_crop

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
        
        # 1. Load Detector
        try:
            self.detector = MTCNNFaceDetector()
        except Exception as e:
            print(f"Error loading detector: {e}")

        # 2. Load Embedder
        try:
            model_path = Path(
                getattr(
                    settings,
                    'FACE_MODEL_PATH',
                    r'D:\PythonWorkspace\PBL5-Model\Recognition\checkpoint\MFNet.pth',
                )
            )
            if not model_path.is_absolute():
                model_path = Path(settings.BASE_DIR) / model_path

            if model_path.exists():
                self.embedder = MobileFaceNetTorchEmbedder(model_path)
            else:
                print(f"Warning: Embedder model not found at {model_path}")
        except Exception as e:
            print(f"Error loading embedder: {e}")
            
        # Cache for embeddings
        self.cached_embeddings = None # shape (N, 128)
        self.cached_labels = None     # shape (N,)
        self.last_cache_update = 0
        self.CACHE_TTL = 30  # seconds

    def _refresh_cache_if_needed(self):
        import time
        if self.cached_embeddings is None or (time.time() - self.last_cache_update > self.CACHE_TTL):
            self._load_embeddings()

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
        """Load all embeddings from DB into numpy arrays."""
        embeddings = FaceEmbedding.objects.select_related('employee').all()
        
        emb_list = []
        lbl_list = []
        
        print(f"[DEBUG] Found {embeddings.count()} embeddings in DB.")

        for face in embeddings:
            try:
                emb_bytes = face.embedding
                if not emb_bytes:
                    continue
                    
                emb_array = np.frombuffer(emb_bytes, dtype=np.float32)
                
                if emb_array.shape == (512,):
                    emb_list.append(emb_array)
                    lbl_list.append(face.employee.id)
            except Exception as e:
                print(f"Error loading embedding {face.id}: {e}")
                
        if emb_list:
            self.cached_embeddings = np.vstack(emb_list)
            self.cached_labels = np.array(lbl_list)
        else:
            self.cached_embeddings = np.zeros((0, 512), dtype=np.float32)
            self.cached_labels = np.array([])
            
        import time
        self.last_cache_update = time.time()

    def process_image(self, image_bytes):
        """
        Process uploaded image bytes (full frame or crop).
        Returns: Tuple(has_face, employee_id, confidence_score/distance)
        """
        if self.detector is None or self.embedder is None:
            self.initialize()
            if self.detector is None or self.embedder is None:
                 print("Face models unavailable.")
                 return False, None, 0.0
            
        # Decode image
        nparr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return False, None, 0.0

        # 1. Detect faces
        detect_threshold = getattr(settings, 'FACE_DETECT_THRESHOLD', 0.6)
        detections = self.detector.detect(frame, conf_threshold=detect_threshold)
        
        if not detections:
            return False, None, 0.0
            
        # Get largest face and crop it
        best_face = max(detections, key=lambda d: d.w * d.h)
        x1, y1, x2, y2 = best_face.as_tuple()
        
        # Use square_crop as in original logic (better for MobileFaceNet)
        face_crop = square_crop(frame, (x1, y1, x2, y2), scale=1.25)
        if face_crop.size == 0:
            return False, None, 0.0

        # 2. Embed
        embedding = self.embedder.embed_face_bgr(face_crop)
        
        # 3. Match
        self._refresh_cache_if_needed()
        
        if self.cached_embeddings.shape[0] == 0:
             # No faces in DB yet
            return True, None, 0.0
            
        # Compute Cosine Similarity (assuming normalized vectors)
        # scores = (N,) array of dot products
        scores = self.cached_embeddings @ embedding
        
        # Find best match
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        
        # Default Threshold: 0.35 in original code. 
        MATCH_THRESHOLD = getattr(settings, 'FACE_MATCH_THRESHOLD', 0.4)
        
        print(f"[DEBUG] Best Score: {best_score:.4f} (Threshold: {MATCH_THRESHOLD}) | Cache Size: {len(self.cached_embeddings)}")

        if best_score > MATCH_THRESHOLD:
            employee_id = self.cached_labels[best_idx]
            return True, int(employee_id), float(best_score)
            
        return True, None, float(best_score)

    def enroll_face(self, image_bytes, employee_id):
        """
        Embed and save face for employee.
        """
        embedding = self.extract_embedding_from_bytes(image_bytes)
        
        try:
            emp = Employee.objects.get(pk=employee_id)
        except Employee.DoesNotExist:
            raise ValueError(f"Employee {employee_id} not found")
        
        FaceEmbedding.objects.create(
            employee=emp,
            embedding=embedding.tobytes()
        )
        
        # Invalidate cache
        self.cached_embeddings = None
        
        return True
