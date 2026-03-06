from pathlib import Path
import numpy as np
import cv2
from django.conf import settings
from .models import FaceEmbedding
from apps.employees.models import Employee
from .core.facekit.embedder_mobilefacenet_arcface import MobileFaceNetArcFaceEmbedder
from .core.facekit.detector_resnet10 import ResNet10FaceDetector
from .core.facekit.vision_utils import square_crop

# Define paths relative to this file
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'core' / 'models'

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
            prototxt = MODELS_DIR / 'deploy.prototxt'
            caffemodel = MODELS_DIR / 'res10_300x300_ssd_iter_140000_fp16.caffemodel'
            
            if prototxt.exists() and caffemodel.exists():
                self.detector = ResNet10FaceDetector(prototxt, caffemodel)
            else:
                print(f"Warning: Detector models not found at {MODELS_DIR}")
        except Exception as e:
            print(f"Error loading detector: {e}")

        # 2. Load Embedder
        try:
            onnx_path = MODELS_DIR / 'mobilefacenet_arcface.onnx'
            if onnx_path.exists():
                self.embedder = MobileFaceNetArcFaceEmbedder(onnx_path)
            else:
                print(f"Warning: Embedder model not found at {MODELS_DIR}")
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
        detections = self.detector.detect(frame, conf_threshold=0.5)
        
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
        if self.detector is None or self.embedder is None:
            self.initialize()
            
        nparr = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Invalid image")
            
        detections = self.detector.detect(frame)
        if not detections:
             raise ValueError("No face detected")
             
        # Use square_crop as in original logic (better for MobileFaceNet)
        best_face = max(detections, key=lambda d: d.w * d.h)
        x1, y1, x2, y2 = best_face.as_tuple()
        
        face_crop = square_crop(frame, (x1, y1, x2, y2), scale=1.25)
        
        embedding = self.embedder.embed_face_bgr(face_crop)
        
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
