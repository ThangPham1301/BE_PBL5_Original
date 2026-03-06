import cv2
import time
import requests
import numpy as np
from pathlib import Path
from django.core.management.base import BaseCommand
from django.conf import settings
# Use absolute imports based on app structure
from apps.face_recognition.core.facekit.detector_resnet10 import ResNet10FaceDetector
from apps.face_recognition.core.facekit.vision_utils import square_crop, draw_label

class Command(BaseCommand):
    help = 'Run the Raspberry Pi IoT Client Simulator for Face Recognition'

    def add_arguments(self, parser):
        parser.add_argument('--server', type=str, default='http://127.0.0.1:8000', help='Server URL (e.g. http://127.0.0.1:8000)')
        parser.add_argument('--camera', type=int, default=0, help='Camera Device Index (default: 0)')

    def handle(self, *args, **options):
        server_url = options['server'].rstrip('/')
        camera_idx = options['camera']
        
        self.stdout.write(self.style.SUCCESS(f"Starting Pi Client Simulator..."))
        self.stdout.write(f"Server: {server_url}")
        self.stdout.write(f"Camera: {camera_idx}")
        
        # 1. Setup paths to models
        import apps.face_recognition
        app_path = Path(apps.face_recognition.__file__).parent
        models_dir = app_path / 'core' / 'models'
        
        prototxt = models_dir / 'deploy.prototxt'
        caffemodel = models_dir / 'res10_300x300_ssd_iter_140000_fp16.caffemodel'
        
        if not prototxt.exists() or not caffemodel.exists():
            self.stdout.write(self.style.ERROR(f"Models mismatch! Checked {models_dir}"))
            self.stdout.write(f"Prototxt: {prototxt} ({prototxt.exists()})")
            return

        # 2. Init Local Detector
        self.stdout.write("Initializing local detector...")
        try:
            detector = ResNet10FaceDetector(prototxt, caffemodel)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Failed to load detector: {e}"))
            return
        
        # 3. Camera Loop
        cap = cv2.VideoCapture(camera_idx)
        if not cap.isOpened():
            self.stdout.write(self.style.ERROR(f"Cannot open webcam index {camera_idx}"))
            return
            
        self.stdout.write(self.style.SUCCESS("Camera started!"))
        self.stdout.write("---------------------------------------------------")
        self.stdout.write("Press 'q' to QUIT.")
        self.stdout.write("---------------------------------------------------")
        
        last_sent_time = 0
        INTERVAL = 2.0  # Seconds between recognition requests
        
        while True:
            ret, frame = cap.read()
            if not ret:
                self.stdout.write("Failed to grab frame.")
                break
                
            # Detect locally
            detections = detector.detect(frame, conf_threshold=0.6)
            
            # Draw boxes
            for det in detections:
                x1, y1, x2, y2 = det.as_tuple()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Check interval to send to server
                if time.time() - last_sent_time > INTERVAL:
                    # Crop with margin
                    face_crop = square_crop(frame, (x1, y1, x2, y2), scale=1.25)
                    if face_crop.size > 0:
                        # Encode
                        _, img_encoded = cv2.imencode('.jpg', face_crop)
                        
                        try:
                            # Send to API
                            files = {'file': ('face.jpg', img_encoded.tobytes(), 'image/jpeg')}
                            api_url = f"{server_url}/api/face/recognize/"
                            
                            print("Sending...", end="", flush=True)
                            res = requests.post(api_url, files=files, timeout=3)
                            
                            if res.status_code == 200:
                                data = res.json()
                                if data.get('identified'):
                                    name = data.get('name', 'Unknown')
                                    conf = data.get('confidence', 0.0)
                                    msg = data.get('attendance_message', '')
                                    print(f" -> {name} ({conf:.2f}) | {msg}")
                                    
                                    draw_label(cv2, frame, x1, y1, f"{name} {conf:.2f}")
                                else:
                                    print(" -> Unknown")
                                    draw_label(cv2, frame, x1, y1, "Unknown")
                            else:
                                print(f" -> Error {res.status_code}")
                                
                        except Exception as e:
                            print(f" -> Connection Error: {e}")
                        
                        last_sent_time = time.time()
            
            cv2.imshow("Pi Client Simulator", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
