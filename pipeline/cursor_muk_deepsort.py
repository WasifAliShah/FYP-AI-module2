
# '''
# yolo face detection integrated with insightface + bicubic upscaling + reid
# '''
# import time
# from collections import deque
# import numpy as np
# import cv2
# import torch
# import torchvision.transforms as T
# from torchvision.models import resnet50
# from ultralytics import YOLO
# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, VectorParams, NamedVector
# from dotenv import load_dotenv
# import os
# import qdrant_collections  # for collection constants
# from PIL import Image
# import uuid

# # Load environment variables
# load_dotenv()
# # 1. Initialize the client
# # Try to be resilient: support QDRANT_URL (http) or host+port for gRPC
# from urllib.parse import urlparse

# client = None
# qdrant_url = os.environ.get("QDRANT_URL")
# qdrant_host = os.environ.get("QDRANT_HOST")
# qdrant_port = os.environ.get("QDRANT_PORT") or os.environ.get("QDRANT_GRPC_PORT")

# try:
#     if qdrant_url:
#         parsed = urlparse(qdrant_url)
#         # If the URL port is the gRPC port (6334) prefer gRPC transport to avoid sending HTTP to gRPC
#         if parsed.port == 6334:
#             host = parsed.hostname or "localhost"
#             port = parsed.port
#             # prefer_grpc=True forces the client to use gRPC transport
#             client = QdrantClient(host=host, port=port, prefer_grpc=True)
#             # perform health check before printing success
#             client.get_collections()
#             print(f"Connected to Qdrant (gRPC) at {host}:{port}")
#         else:
#             # Default: use HTTP/REST URL
#             client = QdrantClient(url=qdrant_url)
#             client.get_collections()
#             print(f"Connected to Qdrant (HTTP) at {qdrant_url}")
#     elif qdrant_host and qdrant_port:
#         # assume this is gRPC configuration
#         client = QdrantClient(host=qdrant_host, port=int(qdrant_port), prefer_grpc=True)
#         client.get_collections()
#         print(f"Connected to Qdrant (gRPC) at {qdrant_host}:{qdrant_port}")
#     else:
#         # Fallback to defaults (client will try localhost:6333 HTTP)
#         client = QdrantClient()
#         client.get_collections()
#         print("Connected to Qdrant with default settings (HTTP)")

#     # Quick health check
#     client.get_collections()
# except Exception as e:
#     print("Failed to connect to Qdrant:", e)

# try:
#     # create_qdrant_schema is defined in qdrant_collections.py; call it qualified
#     qdrant_collections.create_qdrant_schema(client)
#     print("Qdrant schema created successfully.")
# except Exception as e:
#     print("Failed to create Qdrant schema:", e)

# # InsightFace
# try:
#     from insightface.app import FaceAnalysis
#     INSIGHTFACE_AVAILABLE = True
# except:
#     print("InsightFace not available!")
#     INSIGHTFACE_AVAILABLE = False

# # Optional: Real-ESRGAN for face super-resolution (if installed)
# USE_REAL_ESRGAN = False
# sr = None
# try:
#     from realesrgan import RealESRGAN
#     USE_REAL_ESRGAN = True
# except Exception:
#     USE_REAL_ESRGAN = False

# # TorchReID for ReID (preferred over ResNet50)
# USE_TORCHREID = False
# try:
#     import torchreid
#     USE_TORCHREID = True
#     print("TorchReID module found")
# except Exception as e:
#     print(f"TorchReID not available: {e}")
#     USE_TORCHREID = False

# # DeepSORT for multi-object tracking (replacement for ByteTrack)
# USE_DEEPSORT = False
# byte_tracker = None
# byte_tracker_objects = None
# try:
#     from deep_sort_realtime.deepsort_tracker import DeepSort
#     USE_DEEPSORT = True
#     print("DeepSORT module found")
# except Exception as e:
#     print(f"DeepSORT not available: {e}")
#     USE_DEEPSORT = False

# # Preserve existing checks that use USE_BYTETRACK
# USE_BYTETRACK = USE_DEEPSORT

# class _DeepSortByteTrackCompat:
#     """Compatibility wrapper to mimic ByteTrack's update API using DeepSORT.
#     - update(detections, img_info, img_size) -> returns iterable of tracks
#     Each returned track has attributes: track_id, tlbr (x1,y1,x2,y2)
#     """
#     def __init__(self, **kwargs):
#         # Use 'mobilenet' embedder (built-in, works on CPU without external dependencies)
#         self.require_confirmation = kwargs.get('require_confirmation', True)  # For persons: require confirmed tracks
#         n_init_val = 3 if self.require_confirmation else 1  # Higher n_init for persons to avoid ID churn
        
#         self.ds = DeepSort(
#             embedder='mobilenet',
#             embedder_gpu=False,
#             max_age=kwargs.get('track_buffer', 30),
#             n_init=n_init_val,
#             max_iou_distance=0.7,  # Standard gating (0.7 is default, good for persons)
#             max_cosine_distance=0.2
#         )

#     def update(self, detections, img_info=None, img_size=None, frame=None):
#         # detections: Nx5 [x1,y1,x2,y2,score]
#         if frame is None:
#             return []
#         det_list = []
#         for det in detections:
#             if len(det) < 5:
#                 continue
#             x1, y1, x2, y2, conf = det[:5]
#             det_list.append(([float(x1), float(y1), float(x2), float(y2)], float(conf), None))
        
#         try:
#             tracks = self.ds.update_tracks(det_list, frame=frame)
#         except Exception as e:
#             print(f"[DeepSORT Error] update_tracks failed: {e}")
#             return []
        
#         out = []
#                 'track_id': t.track_id,
#                 'tlbr': tlbr
#             }))
#         return out


# # ----------------------
# # CONFIG (CPU OPTIMIZED)
# # ----------------------
# VIDEO_PATH = "combined.mp4"
# # Override from environment if provided
# VIDEO_PATH = os.environ.get("VIDEO_PATH", VIDEO_PATH)
# try:
#     VIDEO_ID = int(os.environ.get("VIDEO_ID", "1"))
# except Exception:
#     VIDEO_ID = 1
# REF_FACE_PATHS = ["sabbas.jpg"]

# YOLO_PERSON_MODEL = "yolov8m.pt"        # your person model
# YOLO_FACE_MODEL = "yolov8m-face.pt"     # recommended: yolov8n-face or yolov8m-face
# YOLO_OBJECT_MODEL = "yolov8m.pt"       # general object detection (laptops, phones, bags, etc.)
# DETECT_EVERY_N_FRAMES = 13  # Person detection frequency
# DETECT_OBJECTS_EVERY_N_FRAMES = 5  # Object detection frequency (faster for small moving objects)

# # Display scaling - adjust if video appears zoomed in/out
# # 1.0 = original size, 0.5 = half size, 2.0 = double size
# DISPLAY_SCALE = 0.6  # Display at 60% of original size to fit on screen better

# # Object classes to detect (COCO class IDs)
# # 0: person, 24: handbag, 26: backpack, 28: suitcase, 63: laptop, 67: cell phone, etc.
# OBJECT_CLASSES = [24, 26, 28, 63, 67]  # handbag, backpack, suitcase, laptop, cell phone
# # Set to None to detect all 80 COCO classes
# # OBJECT_CLASSES = None

# # thresholds (tune as needed)
# # Face recognition thresholds - LOWER scores = better matches (cosine distance)
# # Typical good matches: < 0.3 (excellent), 0.3-0.5 (good), 0.5-0.65 (acceptable for small faces)
# # Scores > 0.70 are typically NOT matches (different people)
# FACE_STRICT = 0.50  # For large, clear faces (>= 100px)
# FACE_LOOSE  = 0.60  # For medium faces (60-100px)
# FACE_SMALL_MAX = 0.70  # Maximum threshold for very small faces - allows recognition of distant people
# REID_THRESHOLD_CPU = 0.45  # Increased to reduce ReID false positives (was 0.45)

# AGGREGATION_FRAMES = 10
# TRACKLET_MAX_AGE = 60  # Increased to keep tracks longer (2x for better continuity)
# IOU_THRESHOLD = 0.40

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using device:", DEVICE)
# print("Real-ESRGAN available:", USE_REAL_ESRGAN)

# # CLIP for text-image embeddings (load after DEVICE is defined)
# USE_CLIP = False
# clip_model = None
# clip_preprocess = None
# try:
#     import clip
#     clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
#     clip_model.eval()
#     USE_CLIP = True
#     print("CLIP model loaded (ViT-B/32)")
# except Exception as e:
#     print(f"CLIP not available: {e}")
#     USE_CLIP = False

# # ----------------------
# # LOAD MODELS
# # ----------------------
# yolo_person = YOLO(YOLO_PERSON_MODEL)
# yolo_face = YOLO(YOLO_FACE_MODEL)   # face detector
# yolo_objects = YOLO(YOLO_OBJECT_MODEL)  # general object detector

# # Initialize RealESRGAN if user has it and want to use GPU if available
# if USE_REAL_ESRGAN:
#     try:
#         sr = RealESRGAN(device=DEVICE)
#         sr.load_weights('RealESRGAN_x4plus.pth', download=False)  # ensure weights exist
#         print("Real-ESRGAN initialized on", DEVICE)
#     except Exception as e:
#         print("Real-ESRGAN init failed:", e)
#         sr = None
#         USE_REAL_ESRGAN = False

# # InsightFace (bigger input size for better accuracy on small faces)
# fa = None
# if INSIGHTFACE_AVAILABLE:
#     fa = FaceAnalysis(allowed_modules=['detection', 'landmark', 'recognition'])
#     print("Preparing InsightFace...")
#     # -1 = CPU. If you have GPU and insightface compiled with GPU support, set ctx_id=0
#     fa.prepare(ctx_id=-1, det_size=(1024, 1024))
#     print("InsightFace ready.")

# # ReID encoder: Try TorchReID first, fallback to ResNet50
# reid_model = None
# reid_tf = None
# USE_TORCHREID_ACTIVE = False

# if USE_TORCHREID:
#     try:
#         # Try to use OSNet x1_0 (lightweight and effective for ReID)
#         reid_model = torchreid.models.build_model(
#             name='osnet_x1_0',
#             num_classes=1000,
#             pretrained=True
#         )
#         reid_model = reid_model.to(DEVICE).eval()
#         USE_TORCHREID_ACTIVE = True
#         print("✓ Using TorchReID OSNet for ReID encoding")
#     except Exception as e:
#         print(f"⚠ TorchReID OSNet init failed: {e}")
#         print("   Falling back to ResNet50 for ReID")
#         USE_TORCHREID_ACTIVE = False
#         reid_model = None

# # ResNet50 fallback encoder
# def build_resnet_encoder():
#     model = resnet50(pretrained=True).eval().to(DEVICE)
#     transform = T.Compose([
#         T.ToPILImage(),
#         T.Resize((128, 256)),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
#     ])
#     return model, transform

# # Initialize ResNet50 as fallback (always available)
# reid_model_resnet, reid_tf = build_resnet_encoder()

# if not USE_TORCHREID_ACTIVE:
#     reid_model = reid_model_resnet  # Use ResNet50 as primary if TorchReID not available
#     print("✓ Using ResNet50 for ReID encoding")
# else:
#     # Keep ResNet50 ready as fallback
#     print("✓ ResNet50 ready as fallback for ReID encoding")

# # Initialize DeepSORT if available (as ByteTrack replacement)
# if USE_DEEPSORT:
#     try:
#         cap_temp = cv2.VideoCapture(VIDEO_PATH)
#         fps = cap_temp.get(cv2.CAP_PROP_FPS) or 30.0
#         cap_temp.release()
#         # For persons: require confirmed tracks (n_init=3) to avoid ID churn
#         byte_tracker = _DeepSortByteTrackCompat(track_buffer=60, require_confirmation=True)
#         print(f"✓ DeepSORT initialized for persons (FPS: {fps:.1f}, n_init=3 for stability)")
#         # For small objects like phones, use IoU-based tracking instead (simpler and faster)
#         # DeepSORT's appearance embedder is not needed for small objects
#         byte_tracker_objects = None  # Use fallback IoU-based tracking for objects
#         print(f"✓ IoU-based tracking enabled for objects (simpler for small objects)")
#     except Exception as e:
#         print(f"⚠ DeepSORT initialization failed: {e}")
#         USE_DEEPSORT = False
#         byte_tracker = None
#         byte_tracker_objects = None
# else:
#     print("⚠ DeepSORT not available - using IOU-based tracking only")
#     byte_tracker_objects = None

# def reid_encode(img):
#     """Encode person image for ReID. Uses TorchReID if available, otherwise ResNet50."""
#     if USE_TORCHREID_ACTIVE and reid_model is not None:
#         # Use TorchReID OSNet (preferred method)
#         try:
#             # Convert BGR to RGB and resize for ReID
#             rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             pil_img = T.ToPILImage()(rgb_img)
#             pil_img = T.Resize((256, 128))(pil_img)  # ReID standard size (width, height)
#             tensor_img = T.ToTensor()(pil_img).unsqueeze(0).to(DEVICE)
            
#             # Normalize for ImageNet (torchreid models expect this)
#             normalize_tf = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#             tensor_img = normalize_tf(tensor_img)
            
#             with torch.no_grad():
#                 feat = reid_model(tensor_img).cpu().numpy().squeeze()
            
#             feat = feat / (np.linalg.norm(feat) + 1e-8)
#             return feat.astype(np.float32)
#         except Exception as e:
#             # If TorchReID fails, fallback to ResNet50
#             pass
    
#     # Fallback to ResNet50 (always available)
#     if reid_tf is not None:
#         try:
#             x = reid_tf(img[:,:,::-1]).unsqueeze(0).to(DEVICE)
#             with torch.no_grad():
#                 feat = reid_model_resnet(x).cpu().numpy().squeeze()
#             feat = feat / (np.linalg.norm(feat)+1e-8)
#             return feat.astype(np.float32)
#         except Exception:
#             return None
    
#     return None

# def clip_encode(img):
#     """Encode a BGR image using CLIP; returns normalized embedding or None."""
#     if not USE_CLIP or clip_model is None or clip_preprocess is None:
#         return None
#     try:
#         rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         pil_img = Image.fromarray(rgb)
#         with torch.no_grad():
#             image_tensor = clip_preprocess(pil_img).unsqueeze(0).to(DEVICE)
#             features = clip_model.encode_image(image_tensor)
#             features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
#         return features.squeeze().cpu().numpy().astype(np.float32)
#     except Exception:
#         return None

# def get_dominant_color(crop):
#     """Extract dominant clothing color from an image crop.
#     Prioritizes brown detection to avoid misclassifying it as white.
#     Returns a simple color name or None.
#     """
#     if crop is None or crop.size == 0:
#         return None

#     try:
#         h0, w0 = crop.shape[:2]
#         if h0 < 10 or w0 < 10:
#             return None

#         # Downscale to stabilize and blur to reduce noise
#         scale_h = max(20, h0 // 2)
#         scale_w = max(20, w0 // 2)
#         small = cv2.resize(crop, (scale_w, scale_h), interpolation=cv2.INTER_AREA)
#         small = cv2.GaussianBlur(small, (5, 5), 0)

#         hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

#         H = hsv[:, :, 0].reshape(-1)
#         S = hsv[:, :, 1].reshape(-1)
#         V = hsv[:, :, 2].reshape(-1)
        
#         # Calculate mean values for overall assessment
#         v_mean = float(np.mean(V))
#         s_mean = float(np.mean(S))

#         # FIRST: Check for BROWN before white (brown can look desaturated)
#         # Brown detection in darker warm hues - STRICT to avoid false positives
#         # Require higher saturation and narrower hue range for brown
#         brown_mask = ((H < 20) | (H > 165)) & (V >= 40) & (V < 130) & (S > 40)  # Stricter: narrower hue, higher sat, not too dark
#         brown_ratio = float(np.sum(brown_mask) / H.size) if H.size > 0 else 0
#         # Also detect tan/light brown: warm hues with moderate saturation
#         tan_mask = ((H < 20) | (H > 165)) & (V >= 130) & (V < 180) & (S >= 35) & (S < 85)
#         tan_ratio = float(np.sum(tan_mask) / H.size) if H.size > 0 else 0
        
#         # Only return brown if VERY dominant (>40%) to avoid false positives from shadows
#         if brown_ratio > 0.40 or tan_ratio > 0.40:
#             return "brown"
#         # Or if both together are strong and mean saturation suggests brown
#         if (brown_ratio + tan_ratio) > 0.50 and s_mean > 35 and 50 < v_mean < 160:
#             return "brown"

#         # PRIORITY 1: BLACK detection (very dark)
#         black_mask = (V < 60)
#         black_ratio = float(np.sum(black_mask) / H.size) if H.size > 0 else 0
#         # Allow slightly brighter blacks to count if saturation is low (matte black) and brown is not dominant
#         if (v_mean < 65 and s_mean < 85 and brown_ratio < 0.10) or black_ratio > 0.25:
#             return "black"

#         # PRIORITY 2: WHITE detection (very bright + desaturated)
#         # White = very high brightness + very low saturation, but NOT warm-hued
#         white_mask = (V > 190) & (S < 50)
#         white_ratio = float(np.sum(white_mask) / H.size) if H.size > 0 else 0
        
#         # Only white if no warm/brown hues dominate
#         warm_hues = ((H < 30) | (H > 160))
#         warm_ratio = float(np.sum(warm_hues) / H.size) if H.size > 0 else 0
        
#         # Return white ONLY if very bright, very desaturated, AND not dominated by warm hues
#         if v_mean > 205 and s_mean < 50 and warm_ratio < 0.30:
#             return "white"
#         if v_mean > 200 and s_mean < 40 and warm_ratio < 0.25:
#             return "white"
#         if white_ratio > 0.35 and s_mean < 45 and warm_ratio < 0.25:
#             return "white"
        
#         # PRIORITY 3: GRAY detection (only if clearly not white/brown)
#         # Gray = low saturation, moderate brightness (neither white nor black nor brown)
#         gray_mask = (S < 45) & (V >= 50) & (V <= 190)
#         gray_ratio = float(np.sum(gray_mask) / H.size) if H.size > 0 else 0
        
#         # Gray only if saturation is very low and brightness is moderate, and not brown
#         if gray_ratio > 0.45 and v_mean < 195 and brown_ratio < 0.15:
#             return "gray"
#         if s_mean < 35 and 80 < v_mean < 190 and brown_ratio < 0.15:
#             return "gray"

#         # Filter to colorful pixels for hue analysis
#         valid = (V > 35) & (V < 245) & (S > 30)
#         if np.sum(valid) < 80:
#             # Not enough colorful pixels: choose closest achromatic class by averages
#             if v_mean < 55:
#                 return "black"
#             if v_mean > 195 and s_mean < 45 and brown_ratio < 0.20:
#                 return "white"
#             return "gray" if s_mean < 50 else None

#         Hv = H[valid]
#         Sv = S[valid]
#         Vv = V[valid]

#         # Histogram of hue
#         bins = np.array([0, 10, 25, 35, 80, 100, 130, 150, 170, 181])
#         hist, _ = np.histogram(Hv, bins=bins)
#         idx = int(np.argmax(hist))

#         color_map = {
#             0: "red", 1: "orange", 2: "yellow", 3: "green",
#             4: "cyan", 5: "blue", 6: "purple", 7: "magenta", 8: "red",
#         }

#         if np.mean(Sv) < 55:
#             return "gray"

#         dominant_color = color_map.get(idx, None)
        
#         # BROWN OVERRIDE: If dominant is orange/red but brightness is low or saturation suggests brown
#         if dominant_color in ("orange", "red") and (np.mean(Vv) < 140 or np.mean(Sv) < 50):
#             return "brown"
        
#         # WHITE OVERRIDE: only for truly desaturated warm tones
#         if dominant_color in ("orange", "yellow") and np.mean(Sv) < 65 and np.mean(Vv) > 210:
#             return "white"
        
#         return dominant_color
#     except Exception:
#         return None

# def mask_upper_by_face(crop_person, face_boxes_in_frame, person_box):
#     """Mask out face area from the upper half of the person crop to avoid skin tones.
#     Returns a modified upper_part image with the face region blacked out.
#     face_boxes_in_frame: list of (x1,y1,x2,y2) in full-frame coords
#     person_box: (x1,y1,x2,y2) of person in full-frame coords
#     """
#     if crop_person is None or crop_person.size == 0:
#         return None
#     h, w = crop_person.shape[:2]
#     upper = crop_person[:h//2, :].copy()
#     if not face_boxes_in_frame:
#         return upper
#     px1, py1, px2, py2 = person_box
#     # Iterate faces that intersect person_box and map to crop coordinates
#     for (fx1, fy1, fx2, fy2) in face_boxes_in_frame:
#         # Check intersection with person box
#         ix1 = max(px1, fx1); iy1 = max(py1, fy1)
#         ix2 = min(px2, fx2); iy2 = min(py2, fy2)
#         if ix2 <= ix1 or iy2 <= iy1:
#             continue
#         # Map to crop local coordinates
#         lx1 = max(0, ix1 - px1)
#         ly1 = max(0, iy1 - py1)
#         lx2 = min(w, ix2 - px1)
#         ly2 = min(h//2, iy2 - py1)  # only mask within upper half
#         if lx2 > lx1 and ly2 > ly1:
#             upper[ly1:ly2, lx1:lx2] = 0  # black out face region
#     return upper

# def detect_person_attributes(crop, face_crop=None):
#     """Detect person attributes from crop image with strict validation.
    
#     Returns dict: {
#         "has_hat": bool,       # Detected hat or head covering
#         "has_hood": bool,      # Detected hood on clothing
#         "has_glasses": bool,   # Detected glasses/sunglasses
#     }
    
#     Uses conservative heuristics to minimize false positives:
#     - hat: Detects distinct dark/light regions at top of person crop
#     - hood: Detects pointed/curved top region with strong edge definition
#     - glasses: Looks for symmetric dark oval regions in face area
#     """
#     attributes = {
#         "has_hat": False,
#         "has_hood": False,
#         "has_glasses": False,
#     }
    
#     if crop is None or crop.size == 0:
#         return attributes
    
#     try:
#         crop_h, crop_w = crop.shape[:2]
        
#         # === HAT DETECTION (STRICT) ===
#         # Only detect if there's a DISTINCT shape at top (not just texture)
#         top_region = crop[:max(1, crop_h // 6), :]  # Top 16% only
#         if top_region.size > 0 and top_region.shape[0] > 5:
#             gray = cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY)
#             # Use higher threshold for Laplacian - only strong edge/shape definition counts
#             laplacian = cv2.Laplacian(gray, cv2.CV_64F)
#             variance = np.var(laplacian)
#             # VERY HIGH THRESHOLD - only distinct hat shapes
#             # Normal texture variance: 50-150
#             # Hat shape variance: 300+
#             if variance > 400:
#                 # Additional check: ensure it's a localized blob, not just noisy texture
#                 edges = cv2.Canny(gray, 50, 150)
#                 edge_pixels = np.sum(edges > 0)
#                 # Hat should have concentrated edges at top
#                 if edge_pixels > top_region.size * 0.05:  # At least 5% edges
#                     attributes["has_hat"] = True
        
#         # === HOOD DETECTION (STRICTER) ===
#         # Only mark hood if there is a strong peaked silhouette AND hat is not already detected
#         top_quarter = crop[:crop_h // 5, :]
#         if top_quarter.size > 0 and top_quarter.shape[0] > 5 and not attributes["has_hat"]:
#             gray_top = cv2.cvtColor(top_quarter, cv2.COLOR_BGR2GRAY)
#             edges = cv2.Canny(gray_top, 60, 180)  # Slightly stricter edges

#             # Peak check at top 2 rows (hood tip) and shoulders (rows 3-6)
#             top_rows = edges[0:2, :]
#             shoulder_rows = edges[2:6, :]
#             mid_col = top_rows.shape[1] // 2
#             left_top = np.sum(top_rows[:, :mid_col])
#             right_top = np.sum(top_rows[:, mid_col:])
#             left_shoulder = np.sum(shoulder_rows[:, :mid_col])
#             right_shoulder = np.sum(shoulder_rows[:, mid_col:])

#             edge_density = np.sum(edges > 0) / edges.size

#             # Hood criteria (much stricter):
#             # 1) High edge density > 0.22
#             # 2) Symmetric peak at top (both sides > 12 edge pixels)
#             # 3) Shoulders also have edges (> 20 per side) to indicate fabric fold
#             if (
#                 edge_density > 0.22
#                 and min(left_top, right_top) > 12
#                 and min(left_shoulder, right_shoulder) > 20
#             ):
#                 attributes["has_hood"] = True
        
#         # === GLASSES DETECTION (STRICT) ===
#         if face_crop is not None and face_crop.size > 0:
#             face_h, face_w = face_crop.shape[:2]
#             if face_h > 20 and face_w > 20:
#                 # Analyze ONLY eye region (40-60% of face height, middle width)
#                 eye_start = int(face_h * 0.35)
#                 eye_end = int(face_h * 0.65)
#                 eye_left = int(face_w * 0.2)
#                 eye_right = int(face_w * 0.8)
                
#                 eye_region = face_crop[eye_start:eye_end, eye_left:eye_right]
#                 if eye_region.size > 100:
#                     gray_eyes = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
#                     # Look for VERY dark pixels (glasses frames are black)
#                     very_dark_pixels = np.sum(gray_eyes < 60)  # Very dark threshold
#                     dark_ratio = very_dark_pixels / gray_eyes.size
#                     # HIGH threshold: >25% very dark pixels in eye region
#                     if dark_ratio > 0.25:
#                         attributes["has_glasses"] = True
        
#         return attributes
        
#     except Exception as e:
#         # If detection fails, return neutral attributes (all False)
#         return attributes

# # ----------------------
# # Build reference embeddings
# # ----------------------
# ref_face_embs = []
# ref_reid_embs = []

# def normalize(v):
#     v = v.astype(np.float32)
#     return v / (np.linalg.norm(v)+1e-8)

# for path in REF_FACE_PATHS:
#     img = cv2.imread(path)
#     if img is None:
#         print("Warning: could not load reference image", path)
#         continue

#     # face embedding
#     face_emb = None
#     if fa:
#         faces = fa.get(img)
#         if faces and len(faces) > 0:
#             face_emb = normalize(np.array(faces[0].embedding))
#             ref_face_embs.append(face_emb)
#             print(f"✓ Loaded reference face embedding from {path} (face detected, embedding size: {len(face_emb)})")
#         else:
#             print(f"⚠ Warning: No face detected in reference image {path} - face recognition will not work!")
#     else:
#         print("⚠ Warning: InsightFace not available - face recognition disabled!")

#     # reid embedding
#     reid_emb = reid_encode(img)
#     if reid_emb is not None:
#         ref_reid_embs.append(reid_emb)
#         print(f"✓ Loaded reference ReID embedding from {path} (embedding size: {len(reid_emb)})")

# print(f"\n=== Reference Embeddings Summary ===")
# print(f"Face embeddings: {len(ref_face_embs)}")
# print(f"ReID embeddings: {len(ref_reid_embs)}")
# if len(ref_face_embs) == 0:
#     print("⚠ CRITICAL: No face embeddings loaded! Face recognition will be disabled.")
    
# # Determine if reference is face-only (face-only images are not suitable for ReID)
# USE_REID_VERIFICATION = len(ref_face_embs) > 0 and len(ref_reid_embs) > 0
# # If we have face embeddings but reference might be face-only, disable ReID verification
# # ReID is unreliable when reference is face-only (not full body)
# if len(ref_face_embs) > 0:
#     print("⚠ IMPORTANT: Reference contains face images. ReID verification DISABLED (unreliable for face-only references).")
#     print("   Only face recognition will be used for verification.")
#     USE_REID_VERIFICATION = False
# print("=" * 40 + "\n")

# # ===== DEBUG: Identify embedding dimensions =====
# FACE_EMBEDDING_DIM = len(ref_face_embs[0]) if len(ref_face_embs) > 0 else None
# REID_EMBEDDING_DIM = len(ref_reid_embs[0]) if len(ref_reid_embs) > 0 else None

# print("=" * 40)
# print("🔍 EMBEDDING DIMENSIONS DETECTED:")
# print(f"  Face embedding (InsightFace): {FACE_EMBEDDING_DIM}D")
# print(f"  ReID embedding (TorchReID OSNet): {REID_EMBEDDING_DIM}D")
# print("=" * 40 + "\n")

# # ----------------------
# # Helper functions
# # ----------------------
# def cos_dist(a, b):
#     if a is None or b is None: return 1.0
#     return 1.0 - np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8)

# def upscale_bicubic(img, factor=2):
#     """Upscale image using bicubic interpolation."""
#     return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

# def sr_enhance(img, factor=2):
#     """Use Real-ESRGAN if available, otherwise bicubic upscale.
#     For very small faces, use larger upscale factor."""
#     if USE_REAL_ESRGAN and sr is not None:
#         try:
#             # Real-ESRGAN expects RGB
#             rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             enhanced = sr.predict(rgb)
#             # convert back to BGR
#             enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
#             return enhanced_bgr
#         except Exception:
#             return upscale_bicubic(img, factor=factor)
#     else:
#         return upscale_bicubic(img, factor=factor)

# def is_face_match(emb, face_w):
#     if emb is None or len(ref_face_embs)==0:
#         return False, None

#     # Adaptive thresholds based on face size
#     # IMPORTANT: Lower cosine distance = better match
#     # Good matches: < 0.3 (excellent), 0.3-0.5 (good), 0.5-0.68 (acceptable for small faces)
#     # Scores > 0.70 are typically NOT matches (different people)
    
#     # For small/distant faces, we need to be more lenient because:
#     # 1. Embeddings from small faces are inherently less accurate
#     # 2. But we still need to prevent false positives
#     if face_w >= 100:
#         # Large clear faces - can use strict threshold
#         thr = FACE_STRICT
#     elif face_w >= 60:
#         # Medium faces - slightly more lenient but still strict
#         thr = FACE_LOOSE
#     elif face_w >= 40:
#         # Small-medium faces (40-60px) - more lenient for distant faces
#         # This is the critical range for distant recognition
#         thr = min(FACE_LOOSE + 0.08, FACE_SMALL_MAX)  # 0.68 max (was 0.65)
#     elif face_w >= 25:
#         # Small faces (25-40px) - even more lenient but capped
#         thr = min(FACE_LOOSE + 0.10, FACE_SMALL_MAX)  # 0.70 max (was 0.68)
#     else:
#         # Very small faces (< 25px) - most lenient but still capped
#         # These are very challenging, so we allow higher threshold
#         thr = min(FACE_LOOSE + 0.12, FACE_SMALL_MAX)  # 0.72 max (was 0.70)
    
#     best = min([cos_dist(emb, r) for r in ref_face_embs])
#     is_match = best < thr
    
#     # Safety check: reject very high scores (more lenient for small faces)
#     # For faces < 40px, allow up to 0.70; for 40-60px, allow up to 0.68; larger faces cap at 0.65
#     if face_w < 25:
#         max_allowed = 0.72
#     elif face_w < 40:
#         max_allowed = 0.70
#     elif face_w < 60:
#         max_allowed = 0.68  # More lenient for 40-60px range
#     else:
#         max_allowed = 0.65
    
#     if best > max_allowed:
#         is_match = False
    
#     return is_match, best

# def is_reid_match(emb):
#     if emb is None or len(ref_reid_embs)==0:
#         return False, None

#     best = min([cos_dist(emb, r) for r in ref_reid_embs])
#     return best < REID_THRESHOLD_CPU, best

# def iou(a, b):
#     xA = max(a[0], b[0]); yA = max(a[1], b[1])
#     xB = min(a[2], b[2]); yB = min(a[3], b[3])
#     interW = max(0, xB-xA); interH = max(0, yB-yA)
#     inter = interW * interH
#     areaA = (a[2]-a[0])*(a[3]-a[1])
#     areaB = (b[2]-b[0])*(b[3]-b[1])
#     if areaA + areaB - inter == 0: return 0
#     return inter / (areaA + areaB - inter)

# def box_center(box):
#     x1,y1,x2,y2 = box
#     return ((x1+x2)/2.0, (y1+y2)/2.0)

# def box_inside(inner, outer):
#     # return True if center of inner lies inside outer
#     cx, cy = box_center(inner)
#     x1,y1,x2,y2 = outer
#     return (cx >= x1 and cx <= x2 and cy >= y1 and cy <= y2)

# def apply_nms(boxes, scores, iou_threshold=0.5):
#     """Apply Non-Maximum Suppression to filter overlapping boxes.
#     Returns indices of boxes to keep."""
#     if len(boxes) == 0:
#         return []
    
#     # Convert boxes to format for NMS: [x1, y1, x2, y2]
#     boxes_array = np.array(boxes, dtype=np.float32)
#     scores_array = np.array(scores, dtype=np.float32)
    
#     # Use OpenCV's NMS
#     indices = cv2.dnn.NMSBoxes(boxes, scores_array, score_threshold=0.0, nms_threshold=iou_threshold)
    
#     if len(indices) == 0:
#         return []
    
#     return indices.flatten().tolist()

# def merge_object_tracklets(object_tracklets, iou_thresh=0.6):
#     """Merge object tracklets that represent the same object (same class, high IoU).
#     Keeps the oldest ID and merges bbox/conf history to avoid ID churn for small objects.
#     """
#     try:
#         oids = list(object_tracklets.keys())
#         to_merge = []
#         for i in range(len(oids)):
#             oid_i = oids[i]
#             ti = object_tracklets.get(oid_i)
#             if ti is None:
#                 continue
#             bi = ti.get_latest_bbox()
#             for j in range(i+1, len(oids)):
#                 oid_j = oids[j]
#                 tj = object_tracklets.get(oid_j)
#                 if tj is None:
#                     continue
#                 # Only merge same class
#                 if ti.class_name != tj.class_name:
#                     continue
#                 bj = tj.get_latest_bbox()
#                 if bi is None or bj is None:
#                     continue
#                 if iou(bi, bj) >= iou_thresh:
#                     # Merge j into i (keep smaller id for stability)
#                     keep_id, drop_id = (oid_i, oid_j) if int(oid_i) <= int(oid_j) else (oid_j, oid_i)
#                     to_merge.append((keep_id, drop_id))
#         # Execute merges
#         for keep_id, drop_id in to_merge:
#             if keep_id not in object_tracklets or drop_id not in object_tracklets:
#                 continue
#             keep = object_tracklets[keep_id]
#             drop = object_tracklets[drop_id]
#             # Append drop history into keep
#             for b in drop.bboxes:
#                 keep.bboxes.append(b)
#             for c in drop.confidences:
#                 keep.confidences.append(c)
#             keep.last_frame = max(keep.last_frame, drop.last_frame)
#             # Remove dropped id
#             del object_tracklets[drop_id]
#     except Exception:
#         pass

# def reassign_small_object_ids(object_tracklets, class_keywords, iou_thresh=0.4, max_center_dist=100, require_same_class=False):
#     """Map newly created tracklets to existing ones to prevent ID churn for specific object classes.
#     Prefer the oldest existing ID when two tracks overlap strongly OR are close.
#     Uses predicted positions for fast-moving objects.
    
#     Args:
#         object_tracklets: Dictionary of object tracklets
#         class_keywords: List of keywords to match in class_name (e.g., ["phone", "cell phone"])
#         iou_thresh: IoU threshold for considering objects as same
#         max_center_dist: Maximum center distance in pixels for merging
#     """
#     try:
#         # Collect active tracks matching any keyword
#         def matches_keywords(class_name):
#             return any(keyword.lower() in class_name.lower() for keyword in class_keywords)
        
#         objects = [(oid, t) for oid, t in object_tracklets.items() if matches_keywords(t.class_name)]
#         # Sort by first_frame to prefer older IDs
#         objects.sort(key=lambda x: x[1].first_frame)
#         def center(b):
#             x1,y1,x2,y2 = b
#             return ((x1+x2)/2.0, (y1+y2)/2.0)
#         for i in range(len(objects)):
#             oid_i, ti = objects[i]
#             bi = ti.get_latest_bbox()
#             if bi is None:
#                 continue
#             ci = center(bi)
#             # Also get predicted position for fast motion
#             bi_pred = ti.predict_position(frames_ahead=1)
#             ci_pred = center(bi_pred) if bi_pred else ci
            
#             for j in range(i+1, len(objects)):
#                 oid_j, tj = objects[j]
#                 bj = tj.get_latest_bbox()
#                 if bj is None:
#                     continue
#                 cj = center(bj)
#                 bj_pred = tj.predict_position(frames_ahead=1)
#                 cj_pred = center(bj_pred) if bj_pred else cj
                
#                 # Optionally enforce same class (backpack vs handbag should not merge unless chosen)
#                 if require_same_class and ti.class_name.lower() != tj.class_name.lower():
#                     continue

#                 # Check multiple criteria (OR logic for aggressive matching)
#                 iou_val = iou(bi, bj)
#                 dist_current = (abs(ci[0]-cj[0])**2 + abs(ci[1]-cj[1])**2) ** 0.5
#                 dist_predicted = (abs(ci_pred[0]-cj_pred[0])**2 + abs(ci_pred[1]-cj_pred[1])**2) ** 0.5
                
#                 # Calculate size similarity (helps match same object across frames)
#                 area_i = (bi[2] - bi[0]) * (bi[3] - bi[1])
#                 area_j = (bj[2] - bj[0]) * (bj[3] - bj[1])
#                 size_ratio = min(area_i, area_j) / max(area_i, area_j) if max(area_i, area_j) > 0 else 0
                
#                 # Merge if ANY of these conditions met (very aggressive)
#                 should_merge = (
#                     iou_val >= iou_thresh or  # overlapping
#                     dist_current <= max_center_dist or  # close now
#                     dist_predicted <= max_center_dist or  # will be close soon
#                     (iou_val >= 0.2 and dist_current <= max_center_dist * 1.5) or  # some overlap + nearby
#                     (dist_current <= max_center_dist * 1.8 and size_ratio >= 0.5)  # similar size + close
#                 )
                
#                 if should_merge:
#                     keep_id, drop_id = oid_i, oid_j
#                     if drop_id in object_tracklets and keep_id in object_tracklets:
#                         keep = object_tracklets[keep_id]
#                         drop = object_tracklets[drop_id]
#                         for b in drop.bboxes:
#                             keep.bboxes.append(b)
#                         for c in drop.confidences:
#                             keep.confidences.append(c)
#                         keep.last_frame = max(keep.last_frame, drop.last_frame)
#                         # Merge velocity too
#                         if drop.velocity:
#                             keep.velocity = drop.velocity
#                         del object_tracklets[drop_id]
#     except Exception:
#         pass

# # ----------------------
# # Qdrant insertion function for verified tracklets
# # ----------------------
# import uuid
# from datetime import datetime

# def insert_tracklet_to_qdrant(client, tracklet, video_id=1, segment_id=None, frame_rate=30.0):
#     """
#     Insert a tracklet into Qdrant person_tracks collection.
    
#     Stores averaged face and ReID embeddings with metadata for similarity search.
#     Works for both verified and unverified tracklets.
    
#     Args:
#         client: QdrantClient instance
#         tracklet: Tracklet object (verified or unverified) with avg face/reid embeddings
#         video_id: Video ID from PostgreSQL videos table
#         segment_id: Optional segment ID for video_segments table link
#         frame_rate: Video frame rate for time calculations
    
#     Returns:
#         bool: True if insertion succeeded, False otherwise
#     """
#     if not client:
#         return False
    
#     try:
#         # Get averaged embeddings (may be None for unverified tracklets)
#         face_avg = tracklet.avg_face()
#         reid_avg = tracklet.avg_reid()
        
#         # For unverified tracklets, we still need at least ReID embedding to insert
#         # If both are None, skip insertion (no useful data)
#         if face_avg is None and reid_avg is None:
#             print(f"⚠ Skipping tracklet {tracklet.id} - no embeddings available")
#             return False
        
#         # Use zero vectors as fallback if embeddings are missing
#         if face_avg is None:
#             face_avg = np.zeros(512, dtype=np.float32)
#         if reid_avg is None:
#             reid_avg = np.zeros(512, dtype=np.float32)
        
#         # Generate unique ID for this tracklet entry
#         point_id = str(uuid.uuid4())
        
#         # Calculate time range based on when the tracklet first/last appeared
#         start_frame = tracklet.first_frame if hasattr(tracklet, 'first_frame') else 0
#         end_frame = tracklet.last_frame if hasattr(tracklet, 'last_frame') else start_frame
#         num_frames = max(1, end_frame - start_frame + 1)
        
#         # Estimate time in seconds using frame rate
#         start_time_sec = start_frame / max(frame_rate, 1.0)
#         end_time_sec = end_frame / max(frame_rate, 1.0)
        
#         # Format as HH:MM:SS.mmm (keep milliseconds to avoid truncation)
#         def seconds_to_hms_ms(secs):
#             h = int(secs // 3600)
#             m = int((secs % 3600) // 60)
#             s = int(secs % 60)
#             ms = int((secs - int(secs)) * 1000)
#             return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
        
#         start_time_str = seconds_to_hms_ms(start_time_sec)
#         end_time_str = seconds_to_hms_ms(end_time_sec)
        
#         # Build payload per spec (NO camera_id)
#         payload = {
#             "video_id": video_id,
#             "track_id": tracklet.id,
#             "start_time": start_time_str,
#             "end_time": end_time_str,
#             "num_frames": num_frames,
#             "avg_confidence": 0.85,  # placeholder, can extract from tracklet if available
#             "timestamp": datetime.now().isoformat(),
#         }
        
#         # Add optional segment_id if provided
#         if segment_id is not None:
#             payload["segment_id"] = segment_id
        
#         # Optional fields (set None/empty for now, extend later with attribute detection)
#         payload["person_gender"] = None
#         payload["upper_color"] = tracklet.get_dominant_upper_color()
#         payload["lower_color"] = tracklet.get_dominant_lower_color()
#         payload["attributes"] = tracklet.get_attribute_summary()  # Dictionary of detected attributes
#         payload["object_carried"] = tracklet.carried_summary()
#         payload["verified"] = tracklet.verified  # Indicate if this tracklet was verified against reference
        
#         # Convert embeddings to lists for Qdrant
#         face_vec = face_avg.tolist() if isinstance(face_avg, np.ndarray) else list(face_avg)
#         reid_vec = reid_avg.tolist() if isinstance(reid_avg, np.ndarray) else list(reid_avg)
        
#         # multi_vec: prefer CLIP embedding, fallback to face embedding padded to 768
#         clip_avg = tracklet.avg_clip()

#         def pad_to_768(vec):
#             """Pad vector to 768D by appending zeros."""
#             vec_list = vec.tolist() if isinstance(vec, np.ndarray) else list(vec)
#             if len(vec_list) >= 768:
#                 return vec_list[:768]
#             return vec_list + [0.0] * (768 - len(vec_list))
        
#         if clip_avg is not None:
#             multi_vec = pad_to_768(clip_avg)
#         else:
#             multi_vec = pad_to_768(face_avg)
        
#         # Build vectors dict for NamedVectors (per spec: face_vec 512D, reid_vec 512D, multi_vec 768D)
#         vectors = {
#             "face_vec": face_vec,          # 512D (InsightFace)
#             "reid_vec": reid_vec,          # 512D (TorchReID)
#             "multi_vec": multi_vec,        # 768D (CLIP or padded face_vec)
#         }
        
#         # DEBUG: Print what's being sent to Qdrant
#         print("\n" + "="*80)
#         print(f"🔍 DEBUG: Inserting Tracklet {tracklet.id} to Qdrant ({'VERIFIED' if tracklet.verified else 'UNVERIFIED'})")
#         print("="*80)
#         print(f"Point ID: {point_id}")
#         print(f"\n📦 PAYLOAD:")
#         print(f"  video_id: {payload['video_id']}")
#         print(f"  track_id: {payload['track_id']}")
#         print(f"  start_time: {payload['start_time']}")
#         print(f"  end_time: {payload['end_time']}")
#         print(f"  num_frames: {payload['num_frames']}")
#         print(f"  avg_confidence: {payload['avg_confidence']}")
#         print(f"  timestamp: {payload['timestamp']}")
#         print(f"  person_gender: {payload['person_gender']}")
#         print(f"  upper_color: {payload['upper_color']}")
#         print(f"  lower_color: {payload['lower_color']}")
#         print(f"  object_carried: {payload['object_carried']}")
#         print(f"  verified: {payload['verified']}")
#         if 'segment_id' in payload:
#             print(f"  segment_id: {payload['segment_id']}")
        
#         print(f"\n🔢 VECTORS:")
#         face_status = "InsightFace" if len(tracklet.face_embs) > 0 else "Zero (no face detected)"
#         reid_status = "TorchReID/ResNet50" if len(tracklet.reid_embs) > 0 else "Zero (no ReID)"
#         print(f"  face_vec: {len(face_vec)}D ({face_status})")
#         print(f"    Sample: [{face_vec[0]:.6f}, {face_vec[1]:.6f}, {face_vec[2]:.6f}, ...]")
#         print(f"  reid_vec: {len(reid_vec)}D ({reid_status})")
#         print(f"    Sample: [{reid_vec[0]:.6f}, {reid_vec[1]:.6f}, {reid_vec[2]:.6f}, ...]")
#         print(f"  multi_vec: {len(multi_vec)}D ({'CLIP' if clip_avg is not None else 'Face (padded)'})")
#         print(f"    Sample: [{multi_vec[0]:.6f}, {multi_vec[1]:.6f}, {multi_vec[2]:.6f}, ...]")
        
#         print(f"\n📊 TRACKLET STATS:")
#         print(f"  Verified: {tracklet.verified}")
#         print(f"  Face embeddings collected: {len(tracklet.face_embs)}")
#         print(f"  ReID embeddings collected: {len(tracklet.reid_embs)}")
#         print(f"  CLIP embeddings collected: {len(tracklet.clip_embs)}")
#         print(f"  Carried objects observed: {len(tracklet.carried_objects)}")
#         print("="*80 + "\n")
        
#         # Insert into Qdrant
#         from qdrant_client.models import PointStruct
#         point = PointStruct(
#             id=point_id,
#             vector=vectors,  # NamedVectors
#             payload=payload
#         )
        
#         client.upsert(
#             collection_name="person_tracks",
#             points=[point]
#         )
        
#         status = "VERIFIED" if tracklet.verified else "UNVERIFIED"
#         print(f"✅ Inserted {status} tracklet {tracklet.id} to Qdrant (ID: {point_id})")
#         return True
        
#     except Exception as e:
#         print(f"⚠ Failed to insert tracklet {tracklet.id} to Qdrant: {e}")
#         return False

# def insert_object_track_to_qdrant(client, obj_track, video_id=1, segment_id=None, frame_rate=30.0):
#     """
#     Insert an object tracklet into Qdrant object_tracks collection.
    
#     Stores object tracking data with metadata for similarity search and retrieval.
    
#     Args:
#         client: QdrantClient instance
#         obj_track: ObjectTracklet object with tracking history
#         video_id: Video ID from PostgreSQL videos table
#         segment_id: Optional segment ID for video_segments table link
#         frame_rate: Video frame rate for time calculations
    
#     Returns:
#         bool: True if insertion succeeded, False otherwise
#     """
#     if not client:
#         return False
    
#     try:
#         # Generate unique ID for this object track entry
#         point_id = str(uuid.uuid4())
        
#         # Calculate time range
#         start_frame = obj_track.first_frame
#         end_frame = obj_track.last_frame
#         num_frames = max(1, end_frame - start_frame + 1)
        
#         start_time_sec = start_frame / max(frame_rate, 1.0)
#         end_time_sec = end_frame / max(frame_rate, 1.0)
        
#         def seconds_to_hms_ms(secs):
#             h = int(secs // 3600)
#             m = int((secs % 3600) // 60)
#             s = int(secs % 60)
#             ms = int((secs - int(secs)) * 1000)
#             return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
        
#         start_time_str = seconds_to_hms_ms(start_time_sec)
#         end_time_str = seconds_to_hms_ms(end_time_sec)
        
#         # Build payload for object_tracks collection
#         payload = {
#             "video_id": video_id,
#             "track_id": obj_track.id,
#             "object_type": obj_track.class_name,
#             "object_color": obj_track.get_dominant_color(),  # Add color
#             "start_time": start_time_str,
#             "end_time": end_time_str,
#             "num_frames": num_frames,
#             "avg_confidence": obj_track.get_avg_confidence(),
#             "timestamp": datetime.now().isoformat(),
#         }
        
#         if segment_id is not None:
#             payload["segment_id"] = segment_id
        
#         # Use averaged CLIP embeddings if available, otherwise zeros
#         avg_clip_emb = obj_track.avg_clip()
#         if avg_clip_emb is not None:
#             # Pad to 768D if needed
#             if len(avg_clip_emb) < 768:
#                 object_vec = np.concatenate([avg_clip_emb, np.zeros(768 - len(avg_clip_emb), dtype=np.float32)]).tolist()
#             else:
#                 object_vec = avg_clip_emb[:768].tolist()
#             multi_vec = object_vec  # Use same embedding for multi_vec
#         else:
#             # Fallback to zeros if no CLIP embeddings collected
#             object_vec = np.zeros(768, dtype=np.float32).tolist()
#             multi_vec = np.zeros(768, dtype=np.float32).tolist()
        
#         vectors = {
#             "object_vec": object_vec,  # 768D (placeholder for future CLIP embeddings)
#             "multi_vec": multi_vec,     # 768D (placeholder)
#         }
        
#         # Insert into Qdrant
#         from qdrant_client.models import PointStruct
#         point = PointStruct(
#             id=point_id,
#             vector=vectors,
#             payload=payload
#         )
        
#         client.upsert(
#             collection_name="object_tracks",
#             points=[point]
#         )
        
#         print(f"✅ Inserted object track {obj_track.id} ({obj_track.class_name}) to Qdrant (ID: {point_id}, frames: {num_frames}, conf: {payload['avg_confidence']:.3f})")
#         return True
        
#     except Exception as e:
#         print(f"⚠ Failed to insert object track {obj_track.id} to Qdrant: {e}")
#         return False

# # ----------------------
# # Tracklet class (for persons)
# # ----------------------
# class Tracklet:
#     def __init__(self, tid, bbox, frame_idx):
#         self.id = tid
#         self.bboxes = deque(maxlen=AGGREGATION_FRAMES)
#         self.bboxes.append(bbox)
#         self.first_frame = frame_idx
#         self.last_frame = frame_idx
#         self.face_embs = []
#         self.face_sizes = []  # Store actual detected face sizes
#         self.reid_embs = []
#         self.clip_embs = []
#         self.carried_objects = deque(maxlen=50)  # recent object class names observed with this person
#         self.upper_colors = deque(maxlen=20)  # Track detected upper body colors
#         self.lower_colors = deque(maxlen=20)  # Track detected lower body colors
        
#         # Attributes dictionary: tracks person attributes (has_hat, has_glasses, etc.)
#         self.attributes = {
#             "has_hat": deque(maxlen=20),       # Boolean: person wearing hat
#             "has_hood": deque(maxlen=20),      # Boolean: person wearing hood
#             "has_glasses": deque(maxlen=20),   # Boolean: person wearing glasses
#         }
        
#         self.verified = False
#         self.inserted = False  # set True once pushed to DB
#         self.tracker = None

#     def update(self, bbox, idx, face_emb=None, reid_emb=None, face_size=0, clip_emb=None, carried=None, upper_color=None, lower_color=None, attributes=None):
#         self.bboxes.append(bbox)
#         self.last_frame = idx
#         if face_emb is not None: 
#             self.face_embs.append(face_emb)
#             if face_size > 0:
#                 self.face_sizes.append(face_size)
#         if reid_emb is not None: self.reid_embs.append(reid_emb)
#         if clip_emb is not None: self.clip_embs.append(clip_emb)
#         if carried:
#             # carried can be a list or single string
#             if isinstance(carried, (list, tuple)):
#                 for name in carried:
#                     self.carried_objects.append(name)
#             else:
#                 self.carried_objects.append(carried)
#         if upper_color is not None:
#             self.upper_colors.append(upper_color)
#         if lower_color is not None:
#             self.lower_colors.append(lower_color)
        
#         # Update attributes (dict of attribute_name -> bool)
#         if attributes:
#             for attr_name, attr_value in attributes.items():
#                 if attr_name in self.attributes and isinstance(attr_value, bool):
#                     self.attributes[attr_name].append(attr_value)

#     def avg_face(self):
#         if not self.face_embs: return None
#         avg = np.mean(self.face_embs, axis=0)
#         return normalize(avg)

#     def avg_reid(self):
#         if not self.reid_embs: return None
#         avg = np.mean(self.reid_embs, axis=0)
#         return normalize(avg)
    
#     def avg_clip(self):
#         if not self.clip_embs:
#             return None
#         avg = np.mean(self.clip_embs, axis=0)
#         return normalize(avg)
    
#     def avg_face_size(self):
#         """Get average detected face size, or estimate from bbox if no sizes recorded"""
#         if self.face_sizes:
#             return int(np.mean(self.face_sizes))
#         # Fallback: estimate from last bbox (face is roughly upper 1/3 of person height)
#         if self.bboxes:
#             last = self.bboxes[-1]
#             return max(20, (last[2] - last[0]) // 4)  # Conservative estimate
#         return 0
    
#     def carried_summary(self):
#         """Return filtered list of carried objects based on recent evidence.

#         We require that an object name appears at least 3 times in the recent
#         history and was also observed in the last 10 entries. This prevents
#         transient background detections from lingering.
#         """
#         if not self.carried_objects:
#             return []
#         counts = {}
#         for name in self.carried_objects:
#             counts[name] = counts.get(name, 0) + 1
#         recent = list(self.carried_objects)[-10:]
#         recent_set = set(recent)
#         result = []
#         for name, c in counts.items():
#             if c >= 3 and name in recent_set:
#                 result.append(name)
#         return result
    
#     def get_dominant_upper_color(self):
#         """Get most frequently observed upper body color."""
#         if not self.upper_colors:
#             return None
#         color_counts = {}
#         for color in self.upper_colors:
#             color_counts[color] = color_counts.get(color, 0) + 1
#         return max(color_counts, key=color_counts.get) if color_counts else None
    
#     def get_dominant_lower_color(self):
#         """Get most frequently observed lower body color."""
#         if not self.lower_colors:
#             return None
#         color_counts = {}
#         for color in self.lower_colors:
#             color_counts[color] = color_counts.get(color, 0) + 1
#         return max(color_counts, key=color_counts.get) if color_counts else None
    
#     def get_attribute_summary(self):
#         """Get summary of detected attributes (most frequently observed values).
#         Returns dict of attribute_name -> bool (True if detected, False if not, None if unknown)"""
#         attr_summary = {}
#         for attr_name, attr_deque in self.attributes.items():
#             if not attr_deque:
#                 attr_summary[attr_name] = None
#             else:
#                 # Count True vs False observations
#                 true_count = sum(1 for x in attr_deque if x)
#                 false_count = sum(1 for x in attr_deque if not x)
#                 # Return True if more often observed as True
#                 attr_summary[attr_name] = true_count > false_count
#         return attr_summary

# # ----------------------
# # ObjectTracklet class (for objects like backpacks, laptops, etc.)
# # ----------------------
# class ObjectTracklet:
#     def __init__(self, oid, class_name, bbox, confidence, frame_idx):
#         self.id = oid
#         self.class_name = class_name
#         self.class_votes = {class_name: 1}  # Track class label votes to stabilize naming
#         self.class_locked = False  # Once class is stable, don't change it easily
#         self.class_lock_frame = None  # Frame when class became locked
#         self.bboxes = deque(maxlen=30)  # Keep last 30 bboxes
#         self.bboxes.append(bbox)
#         self.confidences = deque(maxlen=30)
#         self.confidences.append(confidence)
#         self.last_frame = frame_idx
#         self.first_frame = frame_idx
#         self.velocity = None  # Track velocity for motion prediction
#         self.clip_embs = []  # CLIP embeddings of object crops for semantic search
#         self.colors = deque(maxlen=20)  # Track detected object colors
#         self.inserted = False  # Track if already inserted to Qdrant

#     def update(self, bbox, confidence, frame_idx, class_name=None, clip_emb=None, color=None):
#         # Update velocity if we have previous bbox
#         if len(self.bboxes) > 0:
#             prev_bbox = self.bboxes[-1]
#             # Calculate center displacement
#             prev_cx = (prev_bbox[0] + prev_bbox[2]) / 2
#             prev_cy = (prev_bbox[1] + prev_bbox[3]) / 2
#             curr_cx = (bbox[0] + bbox[2]) / 2
#             curr_cy = (bbox[1] + bbox[3]) / 2
#             self.velocity = (curr_cx - prev_cx, curr_cy - prev_cy)
#         self.bboxes.append(bbox)
#         self.confidences.append(confidence)
#         self.last_frame = frame_idx
        
#         # Store CLIP embedding if provided
#         if clip_emb is not None:
#             self.clip_embs.append(clip_emb)
        
#         # Store color if provided
#         if color is not None:
#             self.colors.append(color)
        
#         # Update class vote to stabilize label - but respect locked classes
#         if class_name:
#             # If class is locked, only accept same class or very high confidence changes
#             if self.class_locked:
#                 # Only change if new class has significantly more votes AND higher confidence
#                 if class_name != self.class_name:
#                     new_conf = confidence
#                     curr_avg_conf = self.get_avg_confidence()
#                     # Require new class to have 0.15+ higher confidence to override locked class
#                     if new_conf > curr_avg_conf + 0.15:
#                         self.class_votes[class_name] = self.class_votes.get(class_name, 0) + 1
#                     # Otherwise ignore the conflicting class
#             else:
#                 # Not locked yet - accumulate votes
#                 self.class_votes[class_name] = self.class_votes.get(class_name, 0) + 1
                
#                 # Lock class once it has 3+ votes (stable choice)
#                 top_class, top_votes = max(self.class_votes.items(), key=lambda kv: kv[1])
#                 if top_votes >= 3:
#                     self.class_name = top_class
#                     self.class_locked = True
#                     self.class_lock_frame = frame_idx
#                 else:
#                     # Before locking, use voting system
#                     self.class_name = top_class
    
#     def get_latest_bbox(self):
#         """Get the most recent bounding box"""
#         if self.bboxes:
#             return self.bboxes[-1]
#         return None
    
#     def get_avg_confidence(self):
#         """Get average confidence score"""
#         if self.confidences:
#             return np.mean(list(self.confidences))
#         return 0.0
    
#     def avg_clip(self):
#         """Get average CLIP embedding for semantic search"""
#         if not self.clip_embs:
#             return None
#         avg = np.mean(self.clip_embs, axis=0)
#         return avg / (np.linalg.norm(avg) + 1e-8)  # Normalize
    
#     def get_dominant_color(self):
#         """Get most frequent color (voting system)"""
#         if not self.colors:
#             return None
#         color_counts = {}
#         for color in self.colors:
#             color_counts[color] = color_counts.get(color, 0) + 1
#         return max(color_counts, key=color_counts.get)
    
#     def predict_position(self, frames_ahead=1):
#         """Predict future position based on velocity"""
#         if not self.bboxes or not self.velocity:
#             return self.get_latest_bbox()
#         last_bbox = self.bboxes[-1]
#         vx, vy = self.velocity
#         # Predict center position
#         cx = (last_bbox[0] + last_bbox[2]) / 2 + vx * frames_ahead
#         cy = (last_bbox[1] + last_bbox[3]) / 2 + vy * frames_ahead
#         # Keep same size
#         w = last_bbox[2] - last_bbox[0]
#         h = last_bbox[3] - last_bbox[1]
#         return (int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2))

# # ----------------------
# # Main loop
# # ----------------------
# if not os.path.exists(VIDEO_PATH):
#     print(f"ERROR: VIDEO_PATH not found: {VIDEO_PATH}")
#     import sys as _sys
#     _sys.exit(2)
# cap = cv2.VideoCapture(VIDEO_PATH)
# frame_idx = 0
# tracklets = {}  # tid -> Tracklet (for persons)
# object_tracklets = {}  # oid -> ObjectTracklet (for objects)
# next_tid = 1
# next_obj_id = 1
# OBJ_TRACK_MAX_AGE = 30  # Keep object tracks for 30 frames after last detection
# OBJ_IOU_THRESHOLD = 0.3  # IoU threshold for matching object detections

# # small helper to map face boxes per frame
# face_boxes_frame = []

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_idx += 1
#     h, w = frame.shape[:2]

#     # On non-detection frames, update ByteTrack with empty detections to maintain tracking
#     # This is critical for tracking continuity - ByteTrack needs to be updated every frame
#     # Note: person and object detection now run on different schedules
#     is_person_detection_frame = (frame_idx % DETECT_EVERY_N_FRAMES == 0)
#     is_object_detection_frame = (frame_idx % DETECT_OBJECTS_EVERY_N_FRAMES == 0)
    
#     if USE_BYTETRACK and byte_tracker is not None and not is_person_detection_frame:
#         try:
#             # Update ByteTrack with empty detections to maintain existing tracks
#             # ByteTrack will predict positions for existing tracks even without new detections
#             empty_detections = np.array([], dtype=np.float32).reshape(0, 5)
#             img_info = (h, w)
#             img_size = (w, h)
#             tracked_objects = byte_tracker.update(empty_detections, img_info, img_size, frame=frame)
            
#             # Update tracklets with ByteTrack predictions
#             for track in tracked_objects:
#                 track_id = int(track.track_id)
#                 x1_bt, y1_bt, x2_bt, y2_bt = int(track.tlbr[0]), int(track.tlbr[1]), int(track.tlbr[2]), int(track.tlbr[3])
                
#                 # Validate bbox
#                 if x2_bt <= x1_bt or y2_bt <= y1_bt:
#                     continue
                
#                 x1_bt, y1_bt, x2_bt, y2_bt = max(0, x1_bt), max(0, y1_bt), min(w, x2_bt), min(h, y2_bt)
#                 if x2_bt <= x1_bt or y2_bt <= y1_bt:
#                     continue
                
#                 byte_track_bbox = (x1_bt, y1_bt, x2_bt, y2_bt)
                
#                 # Update tracklet if it exists
#                 if track_id in tracklets:
#                     tracklets[track_id].bboxes.append(byte_track_bbox)
#                     tracklets[track_id].last_frame = frame_idx
#         except Exception as e:
#             pass  # ByteTrack update failed, continue

#     # Initialize detected_objects for visualization (empty if not detecting this frame)
#     detected_objects = []
    
#     # ============================================
#     # PERSON DETECTION (every 13 frames)
#     # ============================================
#     if is_person_detection_frame:
#         # person detection - use larger imgsz for close-ups and lower confidence
#         # imgsz=1280 handles both distant and close-up persons better
#         # ByteTrack's strength is using low-confidence detections for better association
#         p_results = yolo_person.predict(frame, imgsz=1280, conf=0.25, classes=[0], verbose=False)
#         person_boxes = []
#         person_detections = []  # For ByteTrack: [x1, y1, x2, y2, score]
        
#         if len(p_results):
#             all_boxes = []
#             all_scores = []
#             all_detections = []
            
#             # Debug: count how many detections before filtering
#             raw_detection_count = len(p_results[0].boxes)
#             filtered_counts = {"invalid": 0, "too_small": 0, "aspect_ratio": 0, "kept": 0}
            
#             for b in p_results[0].boxes:
#                 x1,y1,x2,y2 = b.xyxy[0].cpu().numpy().astype(int)
#                 conf = float(b.conf[0].cpu().numpy())
#                 x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w,x2),min(h,y2)
                
#                 # Skip invalid boxes
#                 if x2 <= x1 or y2 <= y1:
#                     filtered_counts["invalid"] += 1
#                     continue
                
#                 # Filter out small/partial detections (likely hands, arms, etc.)
#                 box_width = x2 - x1
#                 box_height = y2 - y1
#                 box_area = box_width * box_height
#                 frame_area = w * h
                
#                 # Skip very small boxes (likely body parts, not full persons)  
#                 # Minimum size: at least 1.5% of frame area, or minimum 80x120 pixels
#                 # Relaxed from 2% to handle more cases
#                 min_area = max(frame_area * 0.015, 80 * 120)
#                 if box_area < min_area:
#                     filtered_counts["too_small"] += 1
#                     continue
                
#                 # Skip boxes with extreme aspect ratios
#                 # Relaxed aspect ratio checks for close-ups (close-up faces/upper body can be wider)
#                 aspect_ratio = box_width / max(box_height, 1)
#                 if aspect_ratio > 1.2:  # Too wide (relaxed from 0.8 for close-ups)
#                     filtered_counts["aspect_ratio"] += 1
#                     continue
                
#                 # Skip boxes that are too tall and narrow (likely not a person)
#                 if aspect_ratio < 0.2:  # Too narrow
#                     filtered_counts["aspect_ratio"] += 1
#                     continue
                
#                 filtered_counts["kept"] += 1
#                 all_boxes.append([x1, y1, x2, y2])
#                 all_scores.append(conf)
#                 all_detections.append((x1,y1,x2,y2))
            
#             # Debug output for first few frames or when no detections kept
#             if frame_idx <= DETECT_EVERY_N_FRAMES * 3 or filtered_counts["kept"] == 0:
#                 print(f"[Person Detection Debug] Frame {frame_idx}: "
#                       f"Raw detections: {raw_detection_count}, "
#                       f"Filtered: {filtered_counts}, "
#                       f"Frame size: {w}x{h}")
            
#             # Apply NMS to filter overlapping detections (same person detected multiple times)
#             # Use moderate IoU threshold (0.45) - ByteTrack can handle some overlapping detections
#             # Too strict NMS might remove valid detections that ByteTrack could use for association
#             if len(all_boxes) > 0:
#                 nms_indices = apply_nms(all_boxes, all_scores, iou_threshold=0.45)
                
#                 person_boxes = []
#                 person_detections = []
#                 for idx in nms_indices:
#                     person_boxes.append(all_detections[idx])
#                     x1, y1, x2, y2 = all_detections[idx]
#                     person_detections.append([x1, y1, x2, y2, all_scores[idx]])
#             else:
#                 person_boxes = []
#                 person_detections = []
        
#         # Update ByteTrack with detections (or empty if no detections)
#         # ByteTrack MUST be updated every detection frame to maintain tracking continuity
#         tracked_objects = []
#         if USE_BYTETRACK and byte_tracker is not None:
#             try:
#                 if len(person_detections) > 0:
#                     detections_array = np.array(person_detections, dtype=np.float32)
#                 else:
#                     # Update with empty detections to maintain existing tracks
#                     detections_array = np.array([], dtype=np.float32).reshape(0, 5)
                
#                 # ByteTrack.update() requires: (output_results, img_info, img_size)
#                 img_info = (h, w)  # Height, width
#                 img_size = (w, h)  # Width, height
#                 tracked_objects = byte_tracker.update(detections_array, img_info, img_size, frame=frame)
#                 # Debug: print number of tracked objects
#                 if frame_idx <= DETECT_EVERY_N_FRAMES * 3:
#                     print(f"[ByteTrack] Frame {frame_idx}: {len(person_detections)} detections -> {len(tracked_objects)} tracked objects")
#             except Exception as e:
#                 print(f"ByteTrack update error: {e}")
#                 tracked_objects = []
        
#         # face detection (full frame) - lower confidence to catch more faces
#         f_results = yolo_face.predict(frame, imgsz=640, conf=0.25, verbose=False)
#         face_boxes = []
#         if len(f_results):
#             for b in f_results[0].boxes:
#                 fx1,fy1,fx2,fy2 = b.xyxy[0].cpu().numpy().astype(int)
#                 fx1,fy1,fx2,fy2 = max(0,fx1),max(0,fy1),min(w,fx2),min(h,fy2)
#                 face_boxes.append((fx1,fy1,fx2,fy2))

#         # keep face boxes for this frame (used when matching)
#         face_boxes_frame = face_boxes
    
#     # ============================================
#     # OBJECT DETECTION (every 5 frames - faster for phones)
#     # ============================================
#     if is_object_detection_frame:
#         # General object detection (laptops, phones, bags, etc.)
#         detected_objects = []  # List of (class_name, bbox, confidence)
#         # Lower confidence threshold for better detection of objects like backpacks
#         obj_conf_threshold = 0.2  # Lowered to 0.2 for better detection of backpacks/bags
        
#         if OBJECT_CLASSES is not None:
#             # Detect specific classes only - use larger imgsz for better small object detection
#             obj_results = yolo_objects.predict(frame, imgsz=1280, conf=obj_conf_threshold, classes=OBJECT_CLASSES, verbose=False)
#         else:
#             # Detect all COCO classes
#             obj_results = yolo_objects.predict(frame, imgsz=1280, conf=obj_conf_threshold, verbose=False)
        
#         if len(obj_results):
#             # COCO class names
#             class_names = yolo_objects.names
#             current_detections = []  # List of (class_name, bbox, confidence)
            
#             for b in obj_results[0].boxes:
#                 cls_id = int(b.cls[0].cpu().numpy())
#                 conf = float(b.conf[0].cpu().numpy())
#                 x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
#                 x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                
#                 # Skip invalid boxes
#                 if x2 <= x1 or y2 <= y1:
#                     continue
                
#                 class_name = class_names.get(cls_id, f"class_{cls_id}")
#                 current_detections.append((class_name, (x1, y1, x2, y2), conf))
            
#             # Consolidate overlapping handbag/backpack detections - prefer backpack
#             # YOLO sometimes detects the same bag as both handbag and backpack
#             # Force consolidation: any bag that overlaps with another bag should be merged into one
#             consolidated_detections = []
#             used_indices = set()
            
#             for i, (name_i, bbox_i, conf_i) in enumerate(current_detections):
#                 if i in used_indices:
#                     continue
                
#                 # Check if this is a bag-type object
#                 is_bag_i = any(keyword in name_i.lower() for keyword in ['bag', 'backpack', 'handbag'])
                
#                 if is_bag_i:
#                     # Find ALL overlapping bags and merge them into ONE
#                     merged_bboxes = [bbox_i]
#                     merged_confs = [conf_i]
#                     merged_is_backpack = 'backpack' in name_i.lower()
                    
#                     for j in range(i + 1, len(current_detections)):
#                         if j in used_indices:
#                             continue
                        
#                         name_j, bbox_j, conf_j = current_detections[j]
#                         is_bag_j = any(keyword in name_j.lower() for keyword in ['bag', 'backpack', 'handbag'])
                        
#                         if not is_bag_j:
#                             continue
                        
#                         # Check if they overlap significantly (merge all nearby bags)
#                         iou_val = iou(bbox_i, bbox_j)
#                         if iou_val > 0.3:  # Any significant overlap - force merge
#                             merged_bboxes.append(bbox_j)
#                             merged_confs.append(conf_j)
#                             is_backpack_j = 'backpack' in name_j.lower()
#                             # Prefer backpack if ANY detection says backpack
#                             if is_backpack_j:
#                                 merged_is_backpack = True
#                             used_indices.add(j)
                    
#                     # Use average bbox and highest confidence
#                     avg_bbox = (
#                         int(sum(b[0] for b in merged_bboxes) / len(merged_bboxes)),
#                         int(sum(b[1] for b in merged_bboxes) / len(merged_bboxes)),
#                         int(sum(b[2] for b in merged_bboxes) / len(merged_bboxes)),
#                         int(sum(b[3] for b in merged_bboxes) / len(merged_bboxes))
#                     )
#                     max_conf = max(merged_confs)
#                     final_class = 'backpack' if merged_is_backpack else name_i
                    
#                     consolidated_detections.append((final_class, avg_bbox, max_conf))
#                     used_indices.add(i)
#                 else:
#                     # Not a bag - keep as is
#                     consolidated_detections.append((name_i, bbox_i, conf_i))
#                     used_indices.add(i)
            
#             current_detections = consolidated_detections
            
#             # Generate CLIP embeddings and detect colors for each detected object
#             object_clip_embeddings = {}  # Maps (class_name, bbox) -> clip_embedding
#             object_colors = {}  # Maps (class_name, bbox) -> color
#             if USE_CLIP and clip_model is not None and len(current_detections) > 0:
#                 for class_name, bbox, conf in current_detections:
#                     x1, y1, x2, y2 = bbox
#                     obj_crop = frame[y1:y2, x1:x2]
#                     if obj_crop.size > 0:
#                         try:
#                             obj_pil = Image.fromarray(cv2.cvtColor(obj_crop, cv2.COLOR_BGR2RGB))
#                             obj_input = clip_preprocess(obj_pil).unsqueeze(0).to(DEVICE)
#                             with torch.no_grad():
#                                 obj_emb = clip_model.encode_image(obj_input).cpu().numpy().flatten()
#                                 # Normalize
#                                 obj_emb = obj_emb / (np.linalg.norm(obj_emb) + 1e-8)
#                                 object_clip_embeddings[(class_name, bbox)] = obj_emb
#                         except Exception as e:
#                             if frame_idx <= DETECT_EVERY_N_FRAMES * 3:
#                                 print(f"[CLIP] Failed to encode object {class_name}: {e}")
                        
#                         # Detect object color
#                         try:
#                             obj_color = get_dominant_color(obj_crop)
#                             if obj_color:
#                                 object_colors[(class_name, bbox)] = obj_color
#                         except Exception as e:
#                             if frame_idx <= DETECT_EVERY_N_FRAMES * 3:
#                                 print(f"[Color] Failed to detect color for {class_name}: {e}")
            
#             # Use ByteTrack for object tracking (handles sudden movements better)
#             if USE_BYTETRACK and byte_tracker_objects is not None:
#                 try:
#                     # Prepare detections for ByteTrack: [x1, y1, x2, y2, score]
#                     obj_detections_array = []
#                     obj_detection_info = []  # Store (class_name, bbox) for each detection
                    
#                     for class_name, bbox, conf in current_detections:
#                         x1, y1, x2, y2 = bbox
#                         obj_detections_array.append([x1, y1, x2, y2, conf])
#                         obj_detection_info.append((class_name, bbox))
                    
#                     if len(obj_detections_array) > 0:
#                         detections_array = np.array(obj_detections_array, dtype=np.float32)
#                     else:
#                         detections_array = np.array([], dtype=np.float32).reshape(0, 5)
                    
#                     img_info = (h, w)
#                     img_size = (w, h)
#                     tracked_obj_tracks = byte_tracker_objects.update(detections_array, img_info, img_size, frame=frame)
                    
#                     # Match ByteTrack tracks to detections and update object_tracklets
#                     # Create a mapping from ByteTrack ID to class_name and bbox
#                     for track in tracked_obj_tracks:
#                         track_id = int(track.track_id)
#                         x1_bt, y1_bt, x2_bt, y2_bt = int(track.tlbr[0]), int(track.tlbr[1]), int(track.tlbr[2]), int(track.tlbr[3])
                        
#                         if x2_bt <= x1_bt or y2_bt <= y1_bt:
#                             continue
                        
#                         x1_bt, y1_bt, x2_bt, y2_bt = max(0, x1_bt), max(0, y1_bt), min(w, x2_bt), min(h, y2_bt)
#                         if x2_bt <= x1_bt or y2_bt <= y1_bt:
#                             continue
                        
#                         byte_track_bbox = (x1_bt, y1_bt, x2_bt, y2_bt)
                        
#                         # Find best matching detection (using IoU first, then distance)
#                         # ByteTrack bboxes might be slightly different from detection bboxes
#                         best_detection = None
#                         best_iou_val = 0.05  # Very low threshold to catch any overlap
#                         best_distance = float('inf')
                        
#                         for class_name, bbox, conf in current_detections:
#                             iou_val = iou(byte_track_bbox, bbox)
#                             if iou_val > best_iou_val:
#                                 best_iou_val = iou_val
#                                 best_detection = (class_name, bbox, conf)
                            
#                             # Also calculate distance for fallback
#                             cx_bt = (x1_bt + x2_bt) / 2
#                             cy_bt = (y1_bt + y2_bt) / 2
#                             cx_det = (bbox[0] + bbox[2]) / 2
#                             cy_det = (bbox[1] + bbox[3]) / 2
#                             dist = ((cx_bt - cx_det)**2 + (cy_bt - cy_det)**2)**0.5
#                             if dist < best_distance:
#                                 best_distance = dist
#                                 if not best_detection:  # If no IoU match, use closest by distance
#                                     best_detection = (class_name, bbox, conf)
                        
#                         # Use detection info if available, otherwise use ByteTrack bbox
#                         if best_detection:
#                             class_name, det_bbox, conf = best_detection
#                             # Prefer backpack label over handbag if conflicting detections
#                             if track_id in object_tracklets:
#                                 existing_class = object_tracklets[track_id].class_name.lower()
#                                 if 'backpack' in class_name.lower() and 'backpack' not in existing_class:
#                                     pass  # allow upgrade to backpack
#                                 elif 'backpack' in existing_class and 'backpack' not in class_name.lower():
#                                     class_name = object_tracklets[track_id].class_name  # keep backpack label
#                             # Use detection bbox (more accurate) but ByteTrack ID
#                             bbox_to_store = det_bbox
#                         else:
#                             # No matching detection - use ByteTrack bbox and try to get class from existing track
#                             bbox_to_store = byte_track_bbox
#                             if track_id in object_tracklets:
#                                 # Existing track - use its class name
#                                 class_name = object_tracklets[track_id].class_name
#                                 conf = object_tracklets[track_id].get_avg_confidence()
#                             else:
#                                 # New track without detection match - ALWAYS match to closest detection
#                                 if len(current_detections) > 0:
#                                     # Find closest by distance (no threshold - ByteTrack tracks come from detections)
#                                     closest = None
#                                     min_dist = float('inf')
#                                     for class_name_det, bbox_det, conf_det in current_detections:
#                                         cx_bt = (x1_bt + x2_bt) / 2
#                                         cy_bt = (y1_bt + y2_bt) / 2
#                                         cx_det = (bbox_det[0] + bbox_det[2]) / 2
#                                         cy_det = (bbox_det[1] + bbox_det[3]) / 2
#                                         dist = ((cx_bt - cx_det)**2 + (cy_bt - cy_det)**2)**0.5
#                                         if dist < min_dist:
#                                             min_dist = dist
#                                             closest = (class_name_det, bbox_det, conf_det)
                                    
#                                     if closest:
#                                         class_name, bbox_to_store, conf = closest
#                                         if frame_idx <= DETECT_EVERY_N_FRAMES * 3:
#                                             print(f"[Object Tracking] Track {track_id} matched to {class_name} by distance ({min_dist:.1f}px)")
#                                 else:
#                                     # No detections at all - skip
#                                     continue
                        
#                         # Update or create object tracklet with ByteTrack ID
#                         # Get CLIP embedding and color for this detection if available
#                         clip_emb = object_clip_embeddings.get((class_name, bbox_to_store), None)
#                         obj_color = object_colors.get((class_name, bbox_to_store), None)
                        
#                         if track_id in object_tracklets:
#                             object_tracklets[track_id].update(bbox_to_store, conf, frame_idx, class_name=class_name, clip_emb=clip_emb, color=obj_color)
#                         else:
#                             obj_tracklet = ObjectTracklet(track_id, class_name, bbox_to_store, conf, frame_idx)
#                             if clip_emb is not None:
#                                 obj_tracklet.clip_embs.append(clip_emb)
#                             if obj_color is not None:
#                                 obj_tracklet.colors.append(obj_color)
#                             object_tracklets[track_id] = obj_tracklet
                    
#                     # Clean up old tracks
#                     tracks_to_remove = []
#                     for oid, obj_track in object_tracklets.items():
#                         if frame_idx - obj_track.last_frame > OBJ_TRACK_MAX_AGE:
#                             tracks_to_remove.append(oid)
#                     for oid in tracks_to_remove:
#                         # Insert to Qdrant before removing
#                         obj_track = object_tracklets[oid]
#                         if not obj_track.inserted and client:
#                             obj_track.inserted = insert_object_track_to_qdrant(client, obj_track, video_id=VIDEO_ID, segment_id=None, frame_rate=30.0)
#                         del object_tracklets[oid]
                    
#                     if frame_idx <= DETECT_EVERY_N_FRAMES * 3:
#                         obj_names = [obj[0] for obj in current_detections] if current_detections else []
#                         matched_count = sum(1 for tid in object_tracklets.keys() if object_tracklets[tid].last_frame == frame_idx)
#                         print(f"[Object Tracking] Frame {frame_idx}: Detected {len(current_detections)} objects: {obj_names}, ByteTrack tracks: {len(tracked_obj_tracks)}, Active object_tracklets: {len(object_tracklets)}, New/Updated this frame: {matched_count}")
                
#                 except Exception as e:
#                     print(f"ByteTrack object tracking error: {e}")
#                     # Fallback to IoU-based tracking
#                     tracked_obj_tracks = []
#             else:
#                 # Fallback: IoU-based tracking if ByteTrack not available
#                 matched_track_ids = set()
#                 for class_name, bbox, conf in current_detections:
#                     best_match_id = None
#                     best_iou = OBJ_IOU_THRESHOLD
                    
#                     # Check if this is a bag-type object
#                     is_bag = any(keyword in class_name.lower() for keyword in ['bag', 'backpack', 'handbag'])
                    
#                     for oid, obj_track in object_tracklets.items():
#                         # For bags, allow matching across bag types (backpack can match handbag)
#                         if is_bag:
#                             is_track_bag = any(keyword in obj_track.class_name.lower() for keyword in ['bag', 'backpack', 'handbag'])
#                             if not is_track_bag:
#                                 continue  # Only match bags to bags
#                         else:
#                             # For non-bags, require exact class match
#                             if obj_track.class_name != class_name:
#                                 continue
                        
#                         if frame_idx - obj_track.last_frame > OBJ_TRACK_MAX_AGE:
#                             continue
                        
#                         last_bbox = obj_track.get_latest_bbox()
#                         if last_bbox:
#                             iou_val = iou(bbox, last_bbox)
#                             if iou_val > best_iou:
#                                 best_iou = iou_val
#                                 best_match_id = oid
                    
#                     if best_match_id is not None:
#                         clip_emb = object_clip_embeddings.get((class_name, bbox), None)
#                         obj_color = object_colors.get((class_name, bbox), None)
#                         object_tracklets[best_match_id].update(bbox, conf, frame_idx, class_name=class_name, clip_emb=clip_emb, color=obj_color)
#                     else:
#                         obj_tracklet = ObjectTracklet(next_obj_id, class_name, bbox, conf, frame_idx)
#                         clip_emb = object_clip_embeddings.get((class_name, bbox), None)
#                         obj_color = object_colors.get((class_name, bbox), None)
#                         if clip_emb is not None:
#                             obj_tracklet.clip_embs.append(clip_emb)
#                         if obj_color is not None:
#                             obj_tracklet.colors.append(obj_color)
#                         object_tracklets[next_obj_id] = obj_tracklet
#                         next_obj_id += 1
        
#     # Update object tracker on non-object-detection frames (maintain tracking continuity)
#     if USE_BYTETRACK and byte_tracker_objects is not None and not is_object_detection_frame:
#             try:
#                 empty_detections = np.array([], dtype=np.float32).reshape(0, 5)
#                 img_info = (h, w)
#                 img_size = (w, h)
#                 tracked_obj_tracks = byte_tracker_objects.update(empty_detections, img_info, img_size, frame=frame)
                
#                 # Update object_tracklets with ByteTrack predictions
#                 for track in tracked_obj_tracks:
#                     track_id = int(track.track_id)
#                     x1_bt, y1_bt, x2_bt, y2_bt = int(track.tlbr[0]), int(track.tlbr[1]), int(track.tlbr[2]), int(track.tlbr[3])
                    
#                     if x2_bt <= x1_bt or y2_bt <= y1_bt:
#                         continue
                    
#                     x1_bt, y1_bt, x2_bt, y2_bt = max(0, x1_bt), max(0, y1_bt), min(w, x2_bt), min(h, y2_bt)
#                     if x2_bt <= x1_bt or y2_bt <= y1_bt:
#                         continue
                    
#                     byte_track_bbox = (x1_bt, y1_bt, x2_bt, y2_bt)
                    
#                     if track_id in object_tracklets:
#                         # Update with ByteTrack predicted position
#                         obj_track = object_tracklets[track_id]
#                         # Use average confidence from track history
#                         avg_conf = obj_track.get_avg_confidence()
#                         object_tracklets[track_id].update(byte_track_bbox, avg_conf, frame_idx, class_name=obj_track.class_name)
#                     # Note: Don't create new tracklets on non-detection frames - wait for next detection frame
#                 # After prediction step, keep phones and backpacks alive longer by not pruning aggressively
#                 tracks_to_remove = []
#                 for oid, obj_track in object_tracklets.items():
#                     # Check if this is a phone or backpack (prone to ID churn)
#                     is_phone = any(keyword in obj_track.class_name.lower() for keyword in ["phone", "cell phone", "mobile phone"])
#                     is_bag = any(keyword in obj_track.class_name.lower() for keyword in ["backpack", "bag", "handbag", "suitcase"])
                    
#                     if is_phone:
#                         # Phones: double age (very small, fast-moving)
#                         if frame_idx - obj_track.last_frame > OBJ_TRACK_MAX_AGE * 2:
#                             tracks_to_remove.append(oid)
#                     elif is_bag:
#                         # Bags: triple age (detection can be intermittent due to class confusion)
#                         if frame_idx - obj_track.last_frame > OBJ_TRACK_MAX_AGE * 3:
#                             tracks_to_remove.append(oid)
#                     else:
#                         if frame_idx - obj_track.last_frame > OBJ_TRACK_MAX_AGE:
#                             tracks_to_remove.append(oid)
#                 for oid in tracks_to_remove:
#                     # Insert to Qdrant before removing
#                     obj_track = object_tracklets[oid]
#                     if not obj_track.inserted and client:
#                         obj_track.inserted = insert_object_track_to_qdrant(client, obj_track, video_id=VIDEO_ID, segment_id=None, frame_rate=30.0)
#                     del object_tracklets[oid]
#             except Exception as e:
#                 pass  # ByteTrack update failed, continue
    
#     # Update detected_objects list with tracked objects (for visualization)
#     detected_objects = []
#     for oid, obj_track in object_tracklets.items():
#         if frame_idx - obj_track.last_frame <= OBJ_TRACK_MAX_AGE:
#             bbox = obj_track.get_latest_bbox()
#             if bbox:
#                 avg_conf = obj_track.get_avg_confidence()
#                 detected_objects.append((obj_track.class_name, bbox, avg_conf))
    
#     # Debug: Also try detecting ALL classes to see if backpack is detected with different settings
#     # This helps debug if the class ID is correct or if backpack needs even lower threshold
#     if is_object_detection_frame and frame_idx <= DETECT_OBJECTS_EVERY_N_FRAMES * 10:
#         all_obj_results = yolo_objects.predict(frame, imgsz=1280, conf=0.15, verbose=False)
#         if len(all_obj_results):
#             class_names = yolo_objects.names
#             all_detected = {}
#             for b in all_obj_results[0].boxes:
#                 cls_id = int(b.cls[0].cpu().numpy())
#                 conf = float(b.conf[0].cpu().numpy())
#                 class_name = class_names.get(cls_id, f"class_{cls_id}")
#                 if class_name not in all_detected or conf > all_detected[class_name]:
#                     all_detected[class_name] = conf
            
#             # Check if backpack was detected in all classes
#             if 'backpack' in all_detected:
#                 print(f"[Debug] ✓ Backpack detected with confidence {all_detected['backpack']:.3f} (class ID: 24)")
#             elif any('bag' in name.lower() or 'pack' in name.lower() for name in all_detected.keys()):
#                 bag_related = [(name, conf) for name, conf in all_detected.items() if 'bag' in name.lower() or 'pack' in name.lower()]
#                 print(f"[Debug] Bag-related objects detected: {bag_related}")
#             # Show all detected objects for debugging
#             if frame_idx <= DETECT_EVERY_N_FRAMES * 2:
#                 print(f"[Debug] All objects detected (conf >= 0.15): {list(all_detected.keys())}")

#     # ============================================
#     # PERSON TRACKING UPDATE (runs on person detection frames)
#     # ============================================
#     if is_person_detection_frame:
#         # Process ByteTrack tracked objects - ByteTrack already handles ID assignment and matching
#         # ByteTrack's internal logic handles duplicate/overlapping tracks, so we trust its output
#         if USE_BYTETRACK and len(tracked_objects) > 0:
#             # Process each ByteTrack track directly - ByteTrack handles association internally
#             for track in tracked_objects:
#                 track_id = int(track.track_id)
#                 # Get ByteTrack bbox
#                 x1_bt, y1_bt, x2_bt, y2_bt = int(track.tlbr[0]), int(track.tlbr[1]), int(track.tlbr[2]), int(track.tlbr[3])
                
#                 # Validate ByteTrack bbox - skip if invalid
#                 if x2_bt <= x1_bt or y2_bt <= y1_bt:
#                     continue  # Skip invalid ByteTrack bboxes
                
#                 x1_bt, y1_bt, x2_bt, y2_bt = max(0, x1_bt), max(0, y1_bt), min(w, x2_bt), min(h, y2_bt)
                
#                 # Double-check after clamping
#                 if x2_bt <= x1_bt or y2_bt <= y1_bt:
#                     continue  # Skip if still invalid after clamping
                
#                 byte_track_bbox = (x1_bt, y1_bt, x2_bt, y2_bt)
                
#                 # Find the closest person box to this ByteTrack bbox (for face/ReID extraction)
#                 best_box = None
#                 best_iou_val = 0
#                 for box in person_boxes:
#                     iou_val = iou(byte_track_bbox, box)
#                     if iou_val > best_iou_val:
#                         best_iou_val = iou_val
#                         best_box = box
                
#                 # Use best matching person box for face/ReID, or ByteTrack bbox if no good match
#                 # Prefer person detection box (more accurate) over ByteTrack predicted box
#                 # Lower threshold to 0.2 to catch more matches (ByteTrack bboxes might be slightly off)
#                 if best_box and best_iou_val > 0.2:  # More lenient overlap threshold
#                     box = best_box
#                     # Use person detection box for visualization (more accurate)
#                     vis_bbox = best_box
#                 else:
#                     box = byte_track_bbox
#                     # Use ByteTrack bbox if no person box matches - still visualize it!
#                     vis_bbox = byte_track_bbox
                
#                 x1, y1, x2, y2 = box
                
#                 # Validate crop dimensions before extracting
#                 if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > w or y2 > h:
#                     continue  # Skip invalid crops
                
#                 crop_person = frame[y1:y2, x1:x2].copy()
                
#                 # Check if crop is valid (not empty)
#                 if crop_person.size == 0 or crop_person.shape[0] == 0 or crop_person.shape[1] == 0:
#                     continue  # Skip empty crops
                
#                 # Use ByteTrack ID directly - ByteTrack maintains ID consistency
#                 current_tid = track_id
                
#                 # Update or create tracklet with accurate person detection box (for visualization)
#                 # Store the person detection box if available, otherwise use ByteTrack bbox
#                 if current_tid not in tracklets:
#                     tracklets[current_tid] = Tracklet(current_tid, vis_bbox, frame_idx)
#                 else:
#                     # Update existing tracklet with accurate bbox (person detection preferred)
#                     tracklets[current_tid].bboxes.append(vis_bbox)
#                     tracklets[current_tid].last_frame = frame_idx
                
#                 # ---- Face detection: Try multiple methods for best results
#                 face_emb = None
#                 face_size = 0
#                 carried_objs = []

#                 # Method 1: Try YOLO face boxes first (if available)
#                 matched_face = None
#                 for fb in face_boxes_frame:
#                     if box_inside(fb, box) or iou(fb, box) > 0.1:
#                         matched_face = fb
#                         break

#                 if matched_face is not None:
#                     fx1,fy1,fx2,fy2 = matched_face
#                     # ensure clamp
#                     fx1,fy1,fx2,fy2 = max(0,fx1),max(0,fy1),min(w,fx2),min(h,fy2)
#                     face_crop = frame[fy1:fy2, fx1:fx2].copy()
#                     face_size = (fx2-fx1)

#                     # For faces < 60px, ALWAYS upscale before getting embedding for better quality
#                     # This improves recognition accuracy for distant faces
#                     if face_size < 60:
#                         # Use larger upscale factor for very small faces
#                         upscale_factor = 4 if face_size < 30 else (3 if face_size < 45 else 2)
#                         face_crop_up = sr_enhance(face_crop, factor=upscale_factor)
                        
#                         # Try InsightFace on upscaled crop first (better quality)
#                         if fa:
#                             try:
#                                 faces_up = fa.get(face_crop_up)
#                                 if faces_up and len(faces_up) > 0:
#                                     # Debug: check what attributes are available
#                                     if frame_idx <= DETECT_EVERY_N_FRAMES * 3:
#                                         print(f"  → InsightFace found {len(faces_up)} face(s) in upscaled crop, has embedding: {hasattr(faces_up[0], 'embedding')}")
#                                     if hasattr(faces_up[0], 'embedding') and faces_up[0].embedding is not None:
#                                         face_emb = normalize(np.array(faces_up[0].embedding))
#                                         if frame_idx <= DETECT_EVERY_N_FRAMES * 3:
#                                             print(f"  → ✓ Got embedding from upscaled face crop!")
#                                     # Update face_size from upscaled detection
#                                     if hasattr(faces_up[0], 'bbox'):
#                                         detected_w = int(faces_up[0].bbox[2] - faces_up[0].bbox[0])
#                                         if detected_w > 0:
#                                             face_size = detected_w
#                             except Exception as e:
#                                 if frame_idx <= DETECT_EVERY_N_FRAMES * 3:
#                                     print(f"  → Error getting embedding from upscaled crop: {e}")
#                                 pass
#                     else:
#                         # For larger faces (>= 60px), try InsightFace on original crop
#                         if fa:
#                             try:
#                                 faces = fa.get(face_crop)
#                                 if faces and len(faces) > 0:
#                                     # Try to get embedding directly
#                                     if hasattr(faces[0], 'embedding') and faces[0].embedding is not None:
#                                         face_emb = normalize(np.array(faces[0].embedding))
#                                     # Update face_size from InsightFace detection if available
#                                     if hasattr(faces[0], 'bbox'):
#                                         detected_w = int(faces[0].bbox[2] - faces[0].bbox[0])
#                                         if detected_w > 0:
#                                             face_size = detected_w
#                             except Exception as e:
#                                 pass
                    
#                     # Fallback: if upscaling didn't work for small faces, try original
#                     if face_emb is None and face_size < 60:
#                         if fa:
#                             try:
#                                 faces = fa.get(face_crop)
#                                 if faces and len(faces) > 0:
#                                     if hasattr(faces[0], 'embedding') and faces[0].embedding is not None:
#                                         face_emb = normalize(np.array(faces[0].embedding))
#                             except Exception as e:
#                                 pass
                    
#                     # Final fallback: if still no embedding and face is small, try upscaling
#                     if face_emb is None and face_size < 80:
#                         # Use larger upscale factor for very small faces
#                         upscale_factor = 4 if face_size < 30 else (3 if face_size < 50 else 2)
#                         face_crop_up = sr_enhance(face_crop, factor=upscale_factor)
                        
#                         # Try InsightFace on upscaled crop
#                         if fa:
#                             try:
#                                 faces_up = fa.get(face_crop_up)
#                                 if faces_up and len(faces_up) > 0:
#                                     if hasattr(faces_up[0], 'embedding') and faces_up[0].embedding is not None:
#                                         face_emb = normalize(np.array(faces_up[0].embedding))
#                                     # Update face_size from upscaled detection
#                                     if hasattr(faces_up[0], 'bbox'):
#                                         detected_w = int(faces_up[0].bbox[2] - faces_up[0].bbox[0])
#                                         if detected_w > 0:
#                                             face_size = detected_w
#                             except Exception as e:
#                                 pass

#                 # Method 2: ALWAYS try InsightFace on person crop (most reliable, works even if YOLO misses faces)
#                 # This is critical because InsightFace is better at detecting faces in person crops
#                 if face_emb is None and fa:
#                     try:
#                         # First try on person crop directly - this should work!
#                         # InsightFace is very good at detecting faces in person crops
#                         faces = fa.get(crop_person)
#                         if faces and len(faces) > 0:
#                             f = faces[0]
#                             # Debug output
#                             if frame_idx <= DETECT_EVERY_N_FRAMES * 3:
#                                 print(f"  → InsightFace found {len(faces)} face(s) in person crop, has embedding: {hasattr(f, 'embedding')}")
                            
#                             # face bbox is relative to crop_person: compute absolute width
#                             if hasattr(f, 'bbox') and f.bbox is not None:
#                                 fw = int(f.bbox[2] - f.bbox[0])
#                                 face_size = max(face_size, fw)  # Use larger of YOLO or InsightFace size
                            
#                             # ALWAYS try to get embedding directly first (even for small faces)
#                             # InsightFace embeddings work well even on small faces
#                             if hasattr(f, 'embedding') and f.embedding is not None:
#                                 face_emb = normalize(np.array(f.embedding))
#                                 if frame_idx <= DETECT_EVERY_N_FRAMES * 3:
#                                     print(f"  → ✓ Got embedding directly from person crop!")
                            
#                             # For faces < 60px, ALSO try upscaling for potentially better quality
#                             # This improves recognition accuracy for distant faces
#                             if face_size < 60:
#                             # Extract face region and upscale it for better embedding quality
#                                 if hasattr(f, 'bbox') and f.bbox is not None:
#                                     bx1 = max(0, int(f.bbox[0])); by1 = max(0, int(f.bbox[1]))
#                                     bx2 = min(crop_person.shape[1], int(f.bbox[2])); by2 = min(crop_person.shape[0], int(f.bbox[3]))
#                                     if bx2 > bx1 and by2 > by1:
#                                         face_region = crop_person[by1:by2, bx1:bx2].copy()
#                                         if face_region.size > 0:
#                                             # Use larger upscale factor for very small faces
#                                             upscale_factor = 4 if face_size < 30 else (3 if face_size < 45 else 2)
#                                             face_region_up = sr_enhance(face_region, factor=upscale_factor)
                                            
#                                             # Get embedding from upscaled face (better quality)
#                                             faces_up = fa.get(face_region_up)
#                                             if faces_up and len(faces_up) > 0:
#                                                 if hasattr(faces_up[0], 'embedding') and faces_up[0].embedding is not None:
#                                                     # Use upscaled embedding if we don't have one, or if it's better quality
#                                                     emb_up = normalize(np.array(faces_up[0].embedding))
#                                                     if face_emb is None:
#                                                         face_emb = emb_up
#                                                         if frame_idx <= DETECT_EVERY_N_FRAMES * 3:
#                                                             print(f"  → ✓ Got embedding from upscaled face region!")
#                                                 # Update face_size from upscaled detection
#                                                 if hasattr(faces_up[0], 'bbox'):
#                                                     detected_w = int(faces_up[0].bbox[2] - faces_up[0].bbox[0])
#                                                     if detected_w > 0:
#                                                         face_size = detected_w
                            
#                             # Fallback: if upscaling didn't work, try direct embedding
#                             if face_emb is None and face_size >= 60:
#                                 if hasattr(f, 'embedding') and f.embedding is not None:
#                                     face_emb = normalize(np.array(f.embedding))
                            
#                             # Final fallback: if still no embedding and face is small, try upscaling
#                             if face_emb is None and face_size < 80:
#                             # Extract face region and upscale it for better quality
#                                 if hasattr(f, 'bbox') and f.bbox is not None:
#                                     bx1 = max(0, int(f.bbox[0])); by1 = max(0, int(f.bbox[1]))
#                                     bx2 = min(crop_person.shape[1], int(f.bbox[2])); by2 = min(crop_person.shape[0], int(f.bbox[3]))
#                                     if bx2 > bx1 and by2 > by1:
#                                         face_region = crop_person[by1:by2, bx1:bx2].copy()
#                                         if face_region.size > 0:
#                                             # Use larger upscale factor for very small faces
#                                             upscale_factor = 4 if face_size < 30 else (3 if face_size < 50 else 2)
#                                             face_region_up = sr_enhance(face_region, factor=upscale_factor)
                                            
#                                             # Get embedding from upscaled face (better quality)
#                                             faces_up = fa.get(face_region_up)
#                                             if faces_up and len(faces_up) > 0:
#                                                 if hasattr(faces_up[0], 'embedding') and faces_up[0].embedding is not None:
#                                                     face_emb = normalize(np.array(faces_up[0].embedding))
#                                                 # Update face_size from upscaled detection
#                                                 if hasattr(faces_up[0], 'bbox'):
#                                                     detected_w = int(faces_up[0].bbox[2] - faces_up[0].bbox[0])
#                                                     if detected_w > 0:
#                                                         face_size = detected_w
#                         else:
#                             # If InsightFace didn't find face in person crop, try on expanded region around person
#                             # Expand person bbox slightly and try again
#                             expand = 30  # Increased expansion
#                             x1_exp = max(0, x1 - expand)
#                             y1_exp = max(0, y1 - expand)
#                             x2_exp = min(w, x2 + expand)
#                             y2_exp = min(h, y2 + expand)
#                             expanded_crop = frame[y1_exp:y2_exp, x1_exp:x2_exp].copy()
#                             if expanded_crop.size > 0:
#                                 faces_exp = fa.get(expanded_crop)
#                                 if faces_exp and len(faces_exp) > 0:
#                                     f = faces_exp[0]
#                                     if hasattr(f, 'bbox') and f.bbox is not None:
#                                         fw = int(f.bbox[2] - f.bbox[0])
#                                         face_size = fw
#                                     if hasattr(f, 'embedding') and f.embedding is not None:
#                                         face_emb = normalize(np.array(f.embedding))
#                     except Exception as e:
#                         # Add debug info for failures
#                         if frame_idx <= DETECT_EVERY_N_FRAMES * 3:
#                             print(f"  → InsightFace error on person crop: {e}")
#                         pass

#                 # ---- ReID embedding (on person crop)
#                 reid_emb = reid_encode(crop_person)
                
#                 # ---- Carried objects association (EXTREMELY STRICT spatial checks)
#                 # For an object to be marked as carried, ALL conditions must be true:
#                 # 1. Object MUST be COMPLETELY INSIDE person's bbox (no parts sticking out)
#                 # 2. Object must be in UPPER portion of person (hands/shoulders, not waist/hips)
#                 # 3. Object must be SMALL (handbag < 50% of person height)
#                 # 4. Object center must be CLOSE to person's horizontal center (±30% person width)
#                 # This ONLY marks items that are clearly being held/worn
#                 carried_obj_bboxes = []  # Store object bboxes for combined CLIP crop
                
#                 if detected_objects:
#                     # Person body measurements
#                     person_x1, person_y1, person_x2, person_y2 = vis_bbox
#                     person_height = person_y2 - person_y1
#                     person_width = person_x2 - person_x1
#                     person_center_x = (person_x1 + person_x2) / 2.0
#                     person_center_y = (person_y1 + person_y2) / 2.0
                    
#                     # Upper body area: top to ~50% down (shoulders/hands where items are held)
#                     upper_bound_y = person_y1 + person_height * 0.5
                    
#                     for class_name, obj_bbox, obj_conf in detected_objects:
#                         obj_x1, obj_y1, obj_x2, obj_y2 = obj_bbox
#                         obj_center_x = (obj_x1 + obj_x2) / 2.0
#                         obj_center_y = (obj_y1 + obj_y2) / 2.0
#                         obj_width = obj_x2 - obj_x1
#                         obj_height = obj_y2 - obj_y1
                        
#                         # Check 1: Object MUST be COMPLETELY inside person bbox (all 4 corners inside)
#                         completely_inside = (person_x1 <= obj_x1 and obj_x2 <= person_x2 and
#                                            person_y1 <= obj_y1 and obj_y2 <= person_y2)
                        
#                         # Check 2: Object must be in UPPER portion (not below 50% mark)
#                         is_in_upper_portion = obj_center_y < upper_bound_y
                        
#                         # Check 3: Object must be SMALL (max 50% of person height)
#                         size_ratio = obj_height / person_height if person_height > 0 else 1.0
#                         is_reasonably_small = size_ratio < 0.5
                        
#                         # Check 4: Object center must be CLOSE to person's horizontal center
#                         # Allow ±30% of person width from center
#                         horizontal_center_dist = abs(obj_center_x - person_center_x)
#                         horizontal_tolerance = person_width * 0.3
#                         is_horizontally_centered = horizontal_center_dist <= horizontal_tolerance
                        
#                         # Mark as carried ONLY if ALL conditions met
#                         is_carried = (completely_inside and 
#                                      is_in_upper_portion and 
#                                      is_reasonably_small and 
#                                      is_horizontally_centered)
                        
#                         if is_carried:
#                             carried_objs.append(class_name)
#                             carried_obj_bboxes.append(obj_bbox)
                
#                 # ---- CLIP embedding (person + nearby objects for context-aware embedding)
#                 # If person is carrying objects, create combined crop for better semantic understanding
#                 if carried_obj_bboxes:
#                     # Create expanded bbox that includes person + all carried objects
#                     all_boxes = [vis_bbox] + carried_obj_bboxes
#                     combined_x1 = min(box[0] for box in all_boxes)
#                     combined_y1 = min(box[1] for box in all_boxes)
#                     combined_x2 = max(box[2] for box in all_boxes)
#                     combined_y2 = max(box[3] for box in all_boxes)
                    
#                     # Clamp to frame boundaries
#                     combined_x1 = max(0, combined_x1)
#                     combined_y1 = max(0, combined_y1)
#                     combined_x2 = min(w, combined_x2)
#                     combined_y2 = min(h, combined_y2)
                    
#                     # Extract combined crop (person + objects)
#                     if combined_x2 > combined_x1 and combined_y2 > combined_y1:
#                         combined_crop = frame[combined_y1:combined_y2, combined_x1:combined_x2].copy()
#                         clip_emb = clip_encode(combined_crop)  # CLIP sees person WITH object
#                     else:
#                         clip_emb = clip_encode(crop_person)  # Fallback to person-only
#                 else:
#                     # No objects detected - use person-only crop
#                     clip_emb = clip_encode(crop_person)
                
#                 # Detect clothing colors from upper and lower parts of person crop
#                 person_h, person_w = crop_person.shape[:2]
#                 # Upper color: exclude face region to avoid skin tones
#                 upper_part = mask_upper_by_face(crop_person, face_boxes_frame, vis_bbox)
#                 lower_part = crop_person[person_h//2:, :]  # Bottom half
#                 upper_color = get_dominant_color(upper_part)
#                 lower_color = get_dominant_color(lower_part)
#                 # Debug color extraction
#                 if frame_idx % max(1, DETECT_EVERY_N_FRAMES * 2) == 0:
#                     print(f"   → Colors t{current_tid}: upper={upper_color}, lower={lower_color} (crop {person_w}x{person_h})")
                
#                 # Detect person attributes (hat, hood, glasses, backpack)
#                 detected_attributes = detect_person_attributes(crop_person, None)

#                 # Update tracklet with face and ReID embeddings
#                 # Use vis_bbox (person detection box) for accurate visualization
#                 t = tracklets[current_tid]
#                 t.update(vis_bbox, frame_idx, face_emb, reid_emb, face_size, clip_emb, carried_objs, upper_color, lower_color, detected_attributes)
                
#                 # Debug output for first few detections
#                 if frame_idx <= DETECT_EVERY_N_FRAMES * 3:
#                     face_status = "✓" if face_emb is not None else "✗"
#                     yolo_faces = len(face_boxes_frame)
#                     person_h, person_w = crop_person.shape[:2]
#                     upscale_info = ""
#                     if face_size > 0 and face_size < 80:
#                         upscale_factor = 4 if face_size < 30 else (3 if face_size < 50 else 2)
#                         upscale_info = f" (upscaled {upscale_factor}x for better quality)"
#                     print(f"[Frame {frame_idx}] Tracklet {current_tid}: "
#                           f"YOLO faces: {yolo_faces}, Face detected: {face_status} (size: {face_size}px){upscale_info}, "
#                           f"Person crop: {person_w}x{person_h}px, ReID: {'✓' if reid_emb is not None else '✗'}")
#                     if face_emb is None:
#                         if yolo_faces == 0:
#                             print(f"  → No YOLO faces found, trying InsightFace on person crop ({person_w}x{person_h}px)...")
#                         else:
#                             print(f"  → YOLO found {yolo_faces} face(s) but InsightFace failed to extract embedding!")
#                     if face_emb is not None:
#                         print(f"  → ✓ Face successfully detected and embedded! (size: {face_size}px)")
        
#         # Process person boxes that weren't matched to ByteTrack tracks (fallback for missed detections)
#         # This ensures all detected people are visualized, even if ByteTrack didn't track them
#         if USE_BYTETRACK and len(tracked_objects) > 0:
#             # Find person boxes that weren't matched to any ByteTrack track
#             matched_person_boxes = set()
#             for track in tracked_objects:
#                 track_id = int(track.track_id)
#                 x1_bt, y1_bt, x2_bt, y2_bt = int(track.tlbr[0]), int(track.tlbr[1]), int(track.tlbr[2]), int(track.tlbr[3])
#                 if x2_bt <= x1_bt or y2_bt <= y1_bt:
#                     continue
#                 byte_track_bbox = (x1_bt, y1_bt, x2_bt, y2_bt)
#                 # Find matching person box
#                 for box in person_boxes:
#                     if iou(byte_track_bbox, box) > 0.2:
#                         matched_person_boxes.add(box)
            
#             # Process unmatched person boxes with IOU-based matching
#             for box in person_boxes:
#                 if box in matched_person_boxes:
#                     continue  # Already processed by ByteTrack
                
#                 # Use IOU-based matching for unmatched person boxes
#                 x1,y1,x2,y2 = box
#                 crop_person = frame[y1:y2, x1:x2].copy()
                
#                 if crop_person.size == 0 or crop_person.shape[0] == 0 or crop_person.shape[1] == 0:
#                     continue
                
#                 best_tid, best_iouv = None, 0
#                 for tid, t in tracklets.items():
#                     if len(t.bboxes) == 0:
#                         continue
#                     val = iou(box, t.bboxes[-1])
#                     if val > best_iouv:
#                         best_tid, best_iouv = tid, val
                
#                 if best_iouv > IOU_THRESHOLD:
#                     current_tid = best_tid
#                     tracklets[current_tid].bboxes.append(box)
#                     tracklets[current_tid].last_frame = frame_idx
#                 else:
#                     # Create new tracklet for unmatched person
#                     current_tid = next_tid
#                     next_tid += 1
#                     tracklets[current_tid] = Tracklet(current_tid, box, frame_idx)
                
#                 # Extract face and ReID for this unmatched person box
#                 face_emb = None
#                 face_size = 0
                
#                 # Try face detection
#                 matched_face = None
#                 for fb in face_boxes_frame:
#                     if box_inside(fb, box) or iou(fb, box) > 0.1:
#                         matched_face = fb
#                         break
                
#                 if matched_face is not None:
#                     fx1,fy1,fx2,fy2 = matched_face
#                     fx1,fy1,fx2,fy2 = max(0,fx1),max(0,fy1),min(w,fx2),min(h,fy2)
#                     face_crop = frame[fy1:fy2, fx1:fx2].copy()
#                     face_size = (fx2-fx1)
#                     if fa:
#                         try:
#                             faces = fa.get(face_crop)
#                             if faces and len(faces) > 0:
#                                 if hasattr(faces[0], 'embedding') and faces[0].embedding is not None:
#                                     face_emb = normalize(np.array(faces[0].embedding))
#                         except:
#                             pass
                
#                 if face_emb is None and fa:
#                     try:
#                         faces = fa.get(crop_person)
#                         if faces and len(faces) > 0:
#                             f = faces[0]
#                             if hasattr(f, 'embedding') and f.embedding is not None:
#                                 face_emb = normalize(np.array(f.embedding))
#                     except:
#                         pass
                
#                 reid_emb = reid_encode(crop_person)
                
#                 # ---- Carried objects + context-aware CLIP (EXTREMELY STRICT)
#                 carried_obj_bboxes = []
#                 if detected_objects:
#                     # Person body measurements for strict checks
#                     px1, py1, px2, py2 = box
#                     p_h = max(1, py2 - py1)
#                     p_w = max(1, px2 - px1)
#                     p_cx = (px1 + px2) / 2.0
#                     upper_bound_y = py1 + p_h * 0.5

#                     for class_name, obj_bbox, obj_conf in detected_objects:
#                         ox1, oy1, ox2, oy2 = obj_bbox
#                         ocx = (ox1 + ox2) / 2.0
#                         ocy = (oy1 + oy2) / 2.0
#                         o_h = max(1, oy2 - oy1)

#                         # 1) Completely inside person bbox
#                         completely_inside = (px1 <= ox1 and ox2 <= px2 and py1 <= oy1 and oy2 <= py2)
#                         # 2) In upper portion of person
#                         in_upper = ocy < upper_bound_y
#                         # 3) Reasonably small (< 50% of person height)
#                         small_enough = (o_h / p_h) < 0.5
#                         # 4) Horizontally centered (±30% of person width)
#                         horiz_centered = abs(ocx - p_cx) <= (p_w * 0.3)

#                         if completely_inside and in_upper and small_enough and horiz_centered:
#                             carried_objs.append(class_name)
#                             carried_obj_bboxes.append(obj_bbox)
                
#                 # CLIP with person + objects if available
#                 if carried_obj_bboxes:
#                     all_boxes = [box] + carried_obj_bboxes
#                     combined_x1 = max(0, min(b[0] for b in all_boxes))
#                     combined_y1 = max(0, min(b[1] for b in all_boxes))
#                     combined_x2 = min(w, max(b[2] for b in all_boxes))
#                     combined_y2 = min(h, max(b[3] for b in all_boxes))
#                     if combined_x2 > combined_x1 and combined_y2 > combined_y1:
#                         combined_crop = frame[combined_y1:combined_y2, combined_x1:combined_x2].copy()
#                         clip_emb = clip_encode(combined_crop)
#                     else:
#                         clip_emb = clip_encode(crop_person)
#                 else:
#                     clip_emb = clip_encode(crop_person)

#                 # Detect clothing colors
#                 person_h, person_w = crop_person.shape[:2]
#                 upper_part = mask_upper_by_face(crop_person, face_boxes_frame, box)
#                 lower_part = crop_person[person_h//2:, :]
#                 upper_color = get_dominant_color(upper_part)
#                 lower_color = get_dominant_color(lower_part)
#                 if frame_idx % max(1, DETECT_EVERY_N_FRAMES * 2) == 0:
#                     print(f"   → Colors t{current_tid}: upper={upper_color}, lower={lower_color} (crop {person_w}x{person_h})")
                
#                 # Detect person attributes
#                 detected_attributes = detect_person_attributes(crop_person, None)

#                 t = tracklets[current_tid]
#                 t.update(box, frame_idx, face_emb, reid_emb, face_size, clip_emb, carried_objs, upper_color, lower_color, detected_attributes)
        
#         # Fallback: If ByteTrack is not available or no tracked objects, use IOU-based matching
#         if not USE_BYTETRACK or len(tracked_objects) == 0:
#             # Process each person box with IOU-based matching
#             for box in person_boxes:
#                 x1,y1,x2,y2 = box
#                 crop_person = frame[y1:y2, x1:x2].copy()
                
#                 # IOU-based matching (original logic)
#                 best_tid, best_iouv = None, 0
#                 for tid, t in tracklets.items():
#                     val = iou(box, t.bboxes[-1])
#                     if val > best_iouv:
#                         best_tid, best_iouv = tid, val
                
#                 if best_iouv > IOU_THRESHOLD:
#                     current_tid = best_tid
#                 else:
#                     current_tid = next_tid
#                     next_tid += 1
#                     tracklets[current_tid] = Tracklet(current_tid, box, frame_idx)
                
#                 # Face and ReID extraction (same as above)
#                 face_emb = None
#                 face_size = 0
#                 carried_objs = []
                
#                 # Method 1: Try YOLO face boxes first
#                 matched_face = None
#                 for fb in face_boxes_frame:
#                     if box_inside(fb, box) or iou(fb, box) > 0.1:
#                         matched_face = fb
#                         break
                
#                 if matched_face is not None:
#                     fx1,fy1,fx2,fy2 = matched_face
#                     fx1,fy1,fx2,fy2 = max(0,fx1),max(0,fy1),min(w,fx2),min(h,fy2)
#                     face_crop = frame[fy1:fy2, fx1:fx2].copy()
#                     face_size = (fx2-fx1)
                    
#                     if face_size < 60:
#                         upscale_factor = 4 if face_size < 30 else (3 if face_size < 45 else 2)
#                         face_crop_up = sr_enhance(face_crop, factor=upscale_factor)
#                         if fa:
#                             try:
#                                 faces_up = fa.get(face_crop_up)
#                                 if faces_up and len(faces_up) > 0:
#                                     if hasattr(faces_up[0], 'embedding') and faces_up[0].embedding is not None:
#                                         face_emb = normalize(np.array(faces_up[0].embedding))
#                                     if hasattr(faces_up[0], 'bbox'):
#                                         detected_w = int(faces_up[0].bbox[2] - faces_up[0].bbox[0])
#                                         if detected_w > 0:
#                                             face_size = detected_w
#                             except Exception:
#                                 pass
#                     else:
#                         if fa:
#                             try:
#                                 faces = fa.get(face_crop)
#                                 if faces and len(faces) > 0:
#                                     if hasattr(faces[0], 'embedding') and faces[0].embedding is not None:
#                                         face_emb = normalize(np.array(faces[0].embedding))
#                                     if hasattr(faces[0], 'bbox'):
#                                         detected_w = int(faces[0].bbox[2] - faces[0].bbox[0])
#                                         if detected_w > 0:
#                                             face_size = detected_w
#                             except Exception:
#                                 pass
                
#                 # Method 2: Always try InsightFace on person crop
#                 if face_emb is None and fa:
#                     try:
#                         faces = fa.get(crop_person)
#                         if faces and len(faces) > 0:
#                             f = faces[0]
#                             if hasattr(f, 'bbox') and f.bbox is not None:
#                                 fw = int(f.bbox[2] - f.bbox[0])
#                                 face_size = max(face_size, fw)
#                             if hasattr(f, 'embedding') and f.embedding is not None:
#                                 face_emb = normalize(np.array(f.embedding))
#                     except Exception:
#                         pass
                
#                 # ReID embedding
#                 reid_emb = reid_encode(crop_person)
                
#                 # ---- Carried objects + context-aware CLIP (EXTREMELY STRICT)
#                 carried_obj_bboxes = []
#                 if detected_objects:
#                     # Person body measurements for strict checks
#                     px1, py1, px2, py2 = box
#                     p_h = max(1, py2 - py1)
#                     p_w = max(1, px2 - px1)
#                     p_cx = (px1 + px2) / 2.0
#                     upper_bound_y = py1 + p_h * 0.5

#                     for class_name, obj_bbox, obj_conf in detected_objects:
#                         ox1, oy1, ox2, oy2 = obj_bbox
#                         ocx = (ox1 + ox2) / 2.0
#                         ocy = (oy1 + oy2) / 2.0
#                         o_h = max(1, oy2 - oy1)

#                         # 1) Completely inside person bbox
#                         completely_inside = (px1 <= ox1 and ox2 <= px2 and py1 <= oy1 and oy2 <= py2)
#                         # 2) In upper portion of person
#                         in_upper = ocy < upper_bound_y
#                         # 3) Reasonably small (< 50% of person height)
#                         small_enough = (o_h / p_h) < 0.5
#                         # 4) Horizontally centered (±30% of person width)
#                         horiz_centered = abs(ocx - p_cx) <= (p_w * 0.3)

#                         if completely_inside and in_upper and small_enough and horiz_centered:
#                             carried_objs.append(class_name)
#                             carried_obj_bboxes.append(obj_bbox)
                
#                 # CLIP with person + objects if available
#                 if carried_obj_bboxes:
#                     all_boxes = [box] + carried_obj_bboxes
#                     combined_x1 = max(0, min(b[0] for b in all_boxes))
#                     combined_y1 = max(0, min(b[1] for b in all_boxes))
#                     combined_x2 = min(w, max(b[2] for b in all_boxes))
#                     combined_y2 = min(h, max(b[3] for b in all_boxes))
#                     if combined_x2 > combined_x1 and combined_y2 > combined_y1:
#                         combined_crop = frame[combined_y1:combined_y2, combined_x1:combined_x2].copy()
#                         clip_emb = clip_encode(combined_crop)
#                     else:
#                         clip_emb = clip_encode(crop_person)
#                 else:
#                     clip_emb = clip_encode(crop_person)
                
#                 # Detect clothing colors
#                 person_h, person_w = crop_person.shape[:2]
#                 upper_part = mask_upper_by_face(crop_person, face_boxes_frame, box)
#                 lower_part = crop_person[person_h//2:, :]
#                 upper_color = get_dominant_color(upper_part)
#                 lower_color = get_dominant_color(lower_part)
#                 if frame_idx % max(1, DETECT_EVERY_N_FRAMES * 2) == 0:
#                     print(f"   → Colors t{current_tid}: upper={upper_color}, lower={lower_color} (crop {person_w}x{person_h})")
                
#                 # Detect person attributes
#                 detected_attributes = detect_person_attributes(crop_person, None)
                
#                 # Update tracklet
#                 t = tracklets[current_tid]
#                 t.update(box, frame_idx, face_emb, reid_emb, face_size, clip_emb, carried_objs, upper_color, lower_color, detected_attributes)

#     # --------------------------
#     # Verification logic and tracklet insertion
#     # --------------------------
#     for tid, t in list(tracklets.items()):

#         if frame_idx - t.last_frame > TRACKLET_MAX_AGE:
#             # Track has ended (no updates for TRACKLET_MAX_AGE frames)
#             # Insert ALL tracklets (verified and unverified) when they end
#             if not t.inserted and client:
#                 t.inserted = insert_tracklet_to_qdrant(client, t, video_id=VIDEO_ID, segment_id=None, frame_rate=30.0)
#             del tracklets[tid]
#             continue

#         if not t.verified and (len(t.face_embs)+len(t.reid_embs)) >= AGGREGATION_FRAMES:

#             face_avg = t.avg_face()
#             reid_avg = t.avg_reid()
            
#             # Use actual detected face size instead of estimate
#             face_width = t.avg_face_size()

#             # FACE RECOGNITION ONLY: Since reference is face-only, ReID is unreliable and disabled
#             face_ok, face_score = False, None
            
#             # Only verify via face recognition - ReID is disabled for face-only references
#             if face_avg is not None and len(t.face_embs) > 0:
#                 face_ok, face_score = is_face_match(face_avg, face_width)
#                 if face_ok:
#                     t.verified = True
#                     print(f"[VERIFIED] Tracklet {tid} via FACE RECOGNITION  face_score={face_score:.4f}  face_size={face_width}px  face_embs={len(t.face_embs)}")
#                 else:
#                     # Face detected but doesn't match - do NOT verify (ReID disabled for face-only refs)
#                     print(f"[REJECTED] Tracklet {tid} face detected but NO MATCH (score={face_score:.4f}) - ReID disabled for face-only reference")
#             else:
#                 # No face embeddings collected - face detection failed
#                 # DO NOT use ReID as fallback when reference is face-only (unreliable)
#                 print(f"[REJECTED] Tracklet {tid} NO FACE DETECTED - Cannot verify without face recognition (ReID disabled for face-only reference)")
#                 print(f"  → Face embeddings: {len(t.face_embs)}, ReID embeddings: {len(t.reid_embs)}")

#             # ByteTrack handles tracking automatically, no need for manual tracker initialization

#     # --------------------------
#     # Visualization
#     # --------------------------
#     vis = frame.copy()
#     # Merge overlapping same-class tracks and stabilize object IDs
#     merge_object_tracklets(object_tracklets, iou_thresh=0.6)
#     # Run reassignment multiple times for better convergence
#     for _ in range(5):
#         # Phones: keep very aggressive to maintain perfect tracking
#         reassign_small_object_ids(object_tracklets,
#                                   class_keywords=["phone", "cell phone", "mobile phone"],
#                                   iou_thresh=0.35, max_center_dist=120)
#     # Bags: use fewer, stricter passes to avoid merging two nearby bags into one ID
#     for _ in range(2):
#         reassign_small_object_ids(object_tracklets,
#                                   class_keywords=["backpack", "bag", "handbag", "suitcase"],
#                                   iou_thresh=0.55, max_center_dist=110,
#                                   require_same_class=True)
    
#     # Draw tracked persons
#     for tid, t in tracklets.items():
#         if len(t.bboxes) == 0:
#             continue  # Skip tracklets with no bboxes
        
#         x1,y1,x2,y2 = t.bboxes[-1]
        
#         # Validate bbox coordinates
#         if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > w or y2 > h:
#             continue  # Skip invalid bboxes
        
#         # Ensure bbox is within frame bounds
#         x1 = max(0, min(x1, w-1))
#         y1 = max(0, min(y1, h-1))
#         x2 = max(x1+1, min(x2, w))
#         y2 = max(y1+1, min(y2, h))
        
#         color = (0,255,0) if t.verified else (0,0,255)
#         cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
#         cv2.putText(vis, f"Person ID:{tid}{' V' if t.verified else ''}",
#                     (x1, max(15, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
#     # Draw tracked objects (laptops, phones, bags, etc.)
#     for oid, obj_track in object_tracklets.items():
#         if frame_idx - obj_track.last_frame > OBJ_TRACK_MAX_AGE:
#             continue  # Skip old tracks
        
#         bbox = obj_track.get_latest_bbox()
#         if bbox is None:
#             continue
        
#         x1, y1, x2, y2 = bbox
#         # Validate bbox
#         if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > w or y2 > h:
#             continue
        
#         # Use different color for objects (cyan)
#         obj_color = (255, 255, 0)  # Cyan
#         cv2.rectangle(vis, (x1, y1), (x2, y2), obj_color, 2)
#         avg_conf = obj_track.get_avg_confidence()
#         label = f"{obj_track.class_name} ID:{oid} {avg_conf:.2f}"
#         cv2.putText(vis, label, (x1, max(15, y1-8)), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj_color, 2)

#     # Display with scaling if needed
#     if DISPLAY_SCALE != 1.0:
#         vis_display = cv2.resize(vis, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_LINEAR)
#     else:
#         vis_display = vis
    
#     cv2.imshow("Hybrid Face+ReID CPU Pipeline (YOLO-face integrated)", vis_display)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # After loop ends, insert any remaining tracklets (verified and unverified) that were not inserted yet
# for tid, t in list(tracklets.items()):
#     if not t.inserted and client:
#         t.inserted = insert_tracklet_to_qdrant(client, t, video_id=VIDEO_ID, segment_id=None, frame_rate=30.0)

# # Also insert any remaining object tracklets
# for oid, obj_track in list(object_tracklets.items()):
#     if not obj_track.inserted and client:
#         obj_track.inserted = insert_object_track_to_qdrant(client, obj_track, video_id=VIDEO_ID, segment_id=None, frame_rate=30.0)

# cap.release()
# cv2.destroyAllWindows()

# # ========================
# # POST-PROCESSING: Interactive Query Feature
# # ========================
# print("\n" + "="*80)
# print("🎬 VIDEO PROCESSING COMPLETE!")
# print("="*80)
# print(f"Total tracklets processed: {len(tracklets)}")
# print(f"Verified tracklets: {sum(1 for t in tracklets.values() if t.verified)}")
# print("="*80 + "\n")

# def query_objects_by_text(text_prompt, top_k=5):
#     """
#     Query Qdrant for OBJECTS matching a text description using CLIP embeddings.
#     Searches the object_tracks collection for laptops, phones, bags, etc. using semantic search.
#     """
#     if not client:
#         print("❌ Qdrant client not connected")
#         return []

#     try:
#         print(f"\n🔍 Searching for objects: '{text_prompt}'")
#         query_lower = text_prompt.lower()
        
#         # Check if we have CLIP embeddings in object_tracks
#         # If all embeddings are zeros, fall back to keyword matching
#         test_point, _ = client.scroll(
#             collection_name="object_tracks",
#             limit=1,
#             with_vectors=True,
#         )
        
#         use_clip_search = False
#         if test_point:
#             vec = test_point[0].vector.get("object_vec", []) if hasattr(test_point[0], "vector") else []
#             # Check if embedding is non-zero
#             if isinstance(vec, list) and len(vec) > 0 and sum(abs(x) for x in vec) > 0.01:
#                 use_clip_search = True
        
#         if use_clip_search and USE_CLIP and clip_model is not None:
#             # Semantic search using CLIP embeddings
#             print(f"   🔮 Using semantic CLIP search")
            
#             # Encode text query
#             text_token = clip.tokenize([text_prompt]).to(DEVICE)
#             with torch.no_grad():
#                 text_emb = clip_model.encode_text(text_token).cpu().numpy().flatten()
#                 text_emb = text_emb / (np.linalg.norm(text_emb) + 1e-8)  # Normalize
            
#             # Pad to 768D if needed
#             if len(text_emb) < 768:
#                 text_vec = np.concatenate([text_emb, np.zeros(768 - len(text_emb), dtype=np.float32)])
#             else:
#                 text_vec = text_emb[:768]
            
#             # Scroll all objects and compute similarities manually
#             points, _ = client.scroll(
#                 collection_name="object_tracks",
#                 limit=1000,
#                 with_payload=True,
#                 with_vectors=True,
#             )
            
#             if not points:
#                 print("   ⚠ No matching objects found")
#                 return []
            
#             # Calculate similarities with class-prior boosting
#             # Define query intent flags (object types)
#             q_is_phone = any(k in query_lower for k in ["phone", "mobile", "cell", "smartphone", "iphone"])
#             q_is_laptop = any(k in query_lower for k in ["laptop", "computer", "notebook", "macbook", "pc"])
#             q_is_bag = any(k in query_lower for k in ["bag", "backpack", "handbag", "rucksack", "pack"]) and not q_is_laptop and not q_is_phone
#             q_is_suitcase = any(k in query_lower for k in ["suitcase", "luggage", "trolley", "carry-on"]) and not q_is_laptop and not q_is_phone
            
#             # Define color query flags
#             q_color = None
#             color_keywords = ["black", "white", "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "gray", "grey", "silver", "gold"]
#             for color_kw in color_keywords:
#                 if color_kw in query_lower:
#                     q_color = color_kw if color_kw != "grey" else "gray"  # Normalize grey->gray
#                     break

#             similarities = []
#             for p in points:
#                 if not hasattr(p, "vector") or p.vector is None:
#                     continue
                
#                 vec = None
#                 if isinstance(p.vector, dict):
#                     vec = p.vector.get("object_vec") or p.vector.get("multi_vec")
#                 else:
#                     vec = p.vector
                
#                 if vec is None:
#                     continue
                
#                 vec_np = np.array(vec, dtype=np.float32)
#                 if vec_np.size == 0 or vec_np.shape[0] != text_vec.shape[0]:
#                     continue
                
#                 # Check if vector is non-zero (has real embeddings)
#                 if np.sum(np.abs(vec_np)) < 0.01:
#                     continue  # Skip zero embeddings
                
#                 # Compute cosine similarity
#                 sim = np.dot(text_vec, vec_np) / (np.linalg.norm(text_vec) * np.linalg.norm(vec_np) + 1e-8)

#                 # Class-prior boosting based on query keywords and object type
#                 payload = p.payload if hasattr(p, "payload") else {}
#                 obj_type_l = str(payload.get("object_type", "")).lower()
#                 obj_color = str(payload.get("object_color", "")).lower() if payload.get("object_color") else None
                
#                 is_phone = ("phone" in obj_type_l) or ("cell" in obj_type_l)
#                 is_laptop = ("laptop" in obj_type_l) or ("computer" in obj_type_l)
#                 is_bag = any(k in obj_type_l for k in ["bag", "backpack", "handbag"]) and not is_laptop and not is_phone
#                 is_suitcase = "suitcase" in obj_type_l or "luggage" in obj_type_l

#                 boost = 0.0
                
#                 # Class type boosting
#                 if q_is_phone:
#                     if is_phone:
#                         boost += 0.08
#                     elif is_bag or is_suitcase:
#                         boost -= 0.03
#                 if q_is_laptop:
#                     if is_laptop:
#                         boost += 0.08
#                     elif is_bag or is_suitcase or is_phone:
#                         boost -= 0.02
#                 if q_is_bag:
#                     if is_bag:
#                         boost += 0.08
#                     elif is_suitcase:
#                         boost += 0.03
#                     elif is_phone or is_laptop:
#                         boost -= 0.02
#                 if q_is_suitcase:
#                     if is_suitcase:
#                         boost += 0.08
#                     elif is_bag:
#                         boost += 0.03
#                     elif is_phone or is_laptop:
#                         boost -= 0.02
                
#                 # Color boosting (strong signal if color matches)
#                 if q_color and obj_color:
#                     if q_color == obj_color:
#                         boost += 0.12  # Strong boost for color match
#                     else:
#                         boost -= 0.05  # Penalty for wrong color

#                 similarities.append((sim + boost, p))
            
#             if not similarities:
#                 print("   ⚠ No matching objects found")
#                 return []
            
#             # Sort by similarity
#             similarities.sort(key=lambda x: x[0], reverse=True)
#             top_results = similarities[:top_k]
            
#             print(f"\n{'='*80}")
#             print(f"📦 TOP {len(top_results)} OBJECT RESULTS FOR: '{text_prompt}' (CLIP Semantic)")
#             print(f"{'='*80}")
            
#             results = []
#             for idx, (score, p) in enumerate(top_results, 1):
#                 payload = p.payload if hasattr(p, "payload") else {}
                
#                 track_id = payload.get("track_id", "Unknown")
#                 object_type = payload.get("object_type", "Unknown")
#                 object_color = payload.get("object_color", None)
#                 video_id = payload.get("video_id", "Unknown")
#                 start_time = payload.get("start_time", "Unknown")
#                 end_time = payload.get("end_time", "Unknown")
#                 num_frames = payload.get("num_frames", 0)
#                 avg_confidence = payload.get("avg_confidence", 0.0)
                
#                 # Display with color if available
#                 color_str = f" ({object_color})" if object_color else ""
#                 print(f"\n{idx}. Track ID: {track_id} | Object: {object_type}{color_str} | Similarity: {score:.3f}")
#                 print(f"   Video ID: {video_id} | Time: {start_time}s - {end_time}s | Frames: {num_frames}")
#                 print(f"   Confidence: {avg_confidence:.2f}")
                
#                 # Return format consistent with interactive loop: (track_id, score, payload)
#                 results.append((track_id, score, payload))
            
#             return results
        
#         else:
#             # Fallback: Keyword matching (no CLIP embeddings available)
#             print(f"   📝 Using keyword matching (no CLIP embeddings)")
            
#             # Scroll all object tracks
#             points, _ = client.scroll(
#                 collection_name="object_tracks",
#                 limit=1000,
#                 with_payload=True,
#                 with_vectors=False,
#             )

#             if not points:
#                 print("   ⚠ No objects found in collection")
#                 return []

#             matches = []
#             query_lower = text_prompt.lower()
            
#             for p in points:
#                 payload = p.payload if hasattr(p, "payload") else {}
#                 object_type = payload.get("object_type", "").lower()
                
#                 # Simple keyword matching for objects
#                 score = 0.0
                
#                 # Direct object type matching
#                 if any(keyword in query_lower for keyword in ["laptop", "computer"]):
#                     if "laptop" in object_type:
#                         score = 1.0
#                 elif any(keyword in query_lower for keyword in ["phone", "mobile", "cell"]):
#                     if "phone" in object_type or "cell" in object_type:
#                         score = 1.0
#                 elif any(keyword in query_lower for keyword in ["backpack", "bag"]):
#                     if "backpack" in object_type or "bag" in object_type:
#                         score = 1.0
#                 elif any(keyword in query_lower for keyword in ["suitcase", "luggage"]):
#                     if "suitcase" in object_type:
#                         score = 1.0
#                 # Fallback: partial match
#                 else:
#                     for keyword in query_lower.split():
#                         if len(keyword) > 3 and keyword in object_type:
#                             score = 0.8
#                             break
                
#                 if score > 0:
#                     matches.append((score, p))
            
#             if not matches:
#                 print("   ⚠ No matching objects found")
#                 return []
            
#             matches.sort(key=lambda x: x[0], reverse=True)
#             top_results = matches[:top_k]

#             print(f"\n{'='*80}")
#             print(f"📦 TOP {len(top_results)} OBJECT RESULTS FOR: '{text_prompt}' (Keyword)")
#             print(f"{'='*80}")

#             results = []
#             for idx, (score, p) in enumerate(top_results, 1):
#                 payload = p.payload if hasattr(p, "payload") else {}
                
#                 track_id = payload.get("track_id", "Unknown")
#                 object_type = payload.get("object_type", "Unknown")
#                 video_id = payload.get("video_id", "Unknown")
#                 start_time = payload.get("start_time", "Unknown")
#                 end_time = payload.get("end_time", "Unknown")
#                 num_frames = payload.get("num_frames", 0)
#                 avg_confidence = payload.get("avg_confidence", 0.0)
                
#                 results.append((track_id, score, payload))
                
#                 print(f"\n📦 Result #{idx}")
#                 print(f"   Object Type: {object_type}")
#                 print(f"   Track ID: {track_id}")
#                 print(f"   Match Score: {score:.2%}")
#                 print(f"   Video ID: {video_id}")
#                 print(f"   Time Range: {start_time} → {end_time}")
#                 print(f"   Duration: {num_frames} frames")
#                 print(f"   Avg Confidence: {avg_confidence:.3f}")
            
#             print(f"\n{'='*80}\n")
#             return results
        
#     except Exception as e:
#         print(f"❌ Error during object query: {e}")
#         import traceback
#         traceback.print_exc()
#         return []

# def query_tracklets_by_text(text_prompt, top_k=5):
#     """
#     Query Qdrant for tracklets matching a text description using CLIP embeddings.
#     This uses scroll with vectors to stay compatible with older client versions.
#     """
#     if not USE_CLIP or clip_model is None:
#         print("❌ CLIP not available - cannot perform text-based search")
#         return []

#     if not client:
#         print("❌ Qdrant client not connected")
#         return []

#     try:
#         print(f"\n🔍 Searching for: '{text_prompt}'")
#         print("   Encoding text prompt with CLIP...")

#         with torch.no_grad():
#             text_tokens = clip.tokenize(text_prompt).to(DEVICE)
#             text_features = clip_model.encode_text(text_tokens)
#             text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
#             text_embedding = text_features.squeeze().cpu().numpy().astype(np.float32)

#         # Pad to 768D (same as multi_vec in Qdrant)
#         if len(text_embedding) < 768:
#             text_embedding = np.concatenate([text_embedding, np.zeros(768 - len(text_embedding), dtype=np.float32)])
#         else:
#             text_embedding = text_embedding[:768]

#         print(f"   Text embedding generated: {len(text_embedding)}D")

#         # Scroll all points with vectors and payloads
#         points, _ = client.scroll(
#             collection_name="person_tracks",
#             limit=1000,
#             with_payload=True,
#             with_vectors=True,
#         )

#         if not points:
#             print("   ⚠ No points found in collection")
#             return []

#         similarities = []
#         for p in points:
#             if not hasattr(p, "vector") or p.vector is None:
#                 continue

#             vec = None
#             if isinstance(p.vector, dict):
#                 vec = p.vector.get("multi_vec") or p.vector.get("face_vec") or p.vector.get("reid_vec")
#             else:
#                 vec = p.vector

#             if vec is None:
#                 continue

#             vec_np = np.array(vec, dtype=np.float32)
#             if vec_np.size == 0:
#                 continue

#             # Only compare if dimensions match
#             if vec_np.shape[0] != text_embedding.shape[0]:
#                 continue

#             sim = np.dot(text_embedding, vec_np) / (np.linalg.norm(text_embedding) * np.linalg.norm(vec_np) + 1e-8)
            
#             # Apply smart boosting based on query keywords and metadata
#             boosted_score = sim
#             payload = p.payload if hasattr(p, "payload") else {}
#             carried_objs = payload.get("object_carried", [])
#             verified = payload.get("verified", False)
            
#             # Keyword-based boosting for object queries
#             query_lower = text_prompt.lower()
#             if any(keyword in query_lower for keyword in ["phone", "mobile", "cell"]):
#                 if any("phone" in obj.lower() for obj in carried_objs):
#                     boosted_score += 0.15  # Strong boost for matching object
#             elif any(keyword in query_lower for keyword in ["laptop", "computer"]):
#                 if any("laptop" in obj.lower() for obj in carried_objs):
#                     boosted_score += 0.15
#             elif any(keyword in query_lower for keyword in ["bag", "backpack", "handbag"]):
#                 if any(keyword in obj.lower() for obj in carried_objs for keyword in ["bag", "backpack", "handbag"]):
#                     boosted_score += 0.15
#             elif any(keyword in query_lower for keyword in ["holding", "carrying", "with"]):
#                 # Generic "holding/carrying" query - small boost if ANY object present
#                 if carried_objs:
#                     boosted_score += 0.05
            
#             # Clothing color boosting
#             color_keywords = ["red", "blue", "green", "yellow", "black", "white", "gray", "grey", "brown", "pink", "purple", "orange", "cyan", "magenta", "navy", "beige", "tan"]
#             upper_color = payload.get("upper_color")
#             lower_color = payload.get("lower_color")
            
#             for color in color_keywords:
#                 if color in query_lower:
#                     qcolor = "gray" if color == "grey" else color
#                     matched = (upper_color and (upper_color.lower() == qcolor)) or (lower_color and (lower_color.lower() == qcolor))
#                     if matched:
#                         boosted_score += 0.14
#                         print(f"   → Color boost applied (+0.14): query has '{color}', payload upper={upper_color}, lower={lower_color}")
#                     else:
#                         boosted_score -= 0.03
#                         print(f"   → Color penalty applied (-0.03): query has '{color}' but payload colors upper={upper_color}, lower={lower_color}")
#                     break
            
#             # Check for clothing keywords (shirt, pants, jacket, etc.)
#             clothing_keywords = ["shirt", "pants", "jacket", "dress", "skirt", "sweater", "coat", "jeans", "hat", "hood"]
#             if any(keyword in query_lower for keyword in clothing_keywords):
#                 # If query mentions clothing and we have color data, moderate boost
#                 if upper_color or lower_color:
#                     boosted_score += 0.05
            
#             # === ATTRIBUTE-BASED BOOSTING (more decisive, with mild penalty when absent) ===
#             attributes = payload.get("attributes", {})

#             def keyword_hit(words):
#                 return any(w in query_lower for w in words)

#             # Hat / Cap / Beanie
#             if keyword_hit(["hat", "cap", "beanie", "wearing a hat", "wearing a cap", "with hat", "with cap", "has hat", "has cap"]):
#                 if attributes.get("has_hat"):
#                     boosted_score += 0.25  # stronger boost when hat is detected
#                 else:
#                     boosted_score -= 0.04  # slight penalty if query asks for hat but none detected

#             # Hood
#             if keyword_hit(["hood", "hooded", "wearing hood", "has hood", "with hood"]):
#                 if attributes.get("has_hood"):
#                     boosted_score += 0.22
#                     print("   → Hood boost applied (+0.22): has_hood=True")
#                 else:
#                     # Fallback heuristic: gray/white upper without hat often indicates hood
#                     uc = (upper_color or "").lower()
#                     if (uc in ["gray", "grey", "white"]) and not attributes.get("has_hat"):
#                         boosted_score += 0.08
#                         print(f"   → Hood heuristic boost (+0.08): upper={upper_color}, has_hat={attributes.get('has_hat')}")
#                     else:
#                         boosted_score -= 0.03

#             # Glasses / Sunglasses
#             if keyword_hit(["glasses", "sunglasses", "wearing glasses", "has glasses", "with glasses"]):
#                 if attributes.get("has_glasses"):
#                     boosted_score += 0.18
#                 else:
#                     boosted_score -= 0.02
            
#             # Small boost for verified tracklets (reference person match)
#             if "verified" in query_lower and verified:
#                 boosted_score += 0.03
            
#             similarities.append((boosted_score, p))

#         if not similarities:
#             print("   ⚠ No valid vectors to compare")
#             return []

#         similarities.sort(key=lambda x: x[0], reverse=True)
#         top_results = similarities[:top_k]

#         print(f"\n{'='*80}")
#         print(f"📊 TOP {len(top_results)} RESULTS FOR: '{text_prompt}'")
#         print(f"{'='*80}")

#         matches = []
#         for idx, (sim, p) in enumerate(top_results, 1):
#             similarity_norm = (sim + 1) / 2  # map cosine [-1,1] to [0,1]
#             payload = p.payload if hasattr(p, "payload") else {}

#             track_id = payload.get("track_id", "Unknown")
#             video_id = payload.get("video_id", "Unknown")
#             start_time = payload.get("start_time", "Unknown")
#             end_time = payload.get("end_time", "Unknown")
#             num_frames = payload.get("num_frames", 0)
#             verified = payload.get("verified", False)
#             carried_objs = payload.get("object_carried", [])

#             matches.append((track_id, similarity_norm, payload))

#             print(f"\n🎯 Result #{idx}")
#             print(f"   Track ID: {track_id}")
#             print(f"   Similarity Score: {similarity_norm:.4f} (0=no match, 1=perfect match)")
#             print(f"   Video ID: {video_id}")
#             print(f"   Time Range: {start_time} → {end_time}")
#             print(f"   Duration: {num_frames} frames")
#             print(f"   Verified: {'✓ YES' if verified else '✗ NO'}")
            
#             # Display clothing colors
#             upper_color = payload.get("upper_color")
#             lower_color = payload.get("lower_color")
#             if upper_color or lower_color:
#                 color_info = []
#                 if upper_color:
#                     color_info.append(f"upper={upper_color}")
#                 if lower_color:
#                     color_info.append(f"lower={lower_color}")
#                 print(f"   Clothing Colors: {', '.join(color_info)}")
#             else:
#                 print("   Clothing Colors: Not detected")
            
#             # Display detected attributes
#             attributes = payload.get("attributes", {})
#             detected_attrs = []
#             if attributes.get("has_hat"):
#                 detected_attrs.append("👒 Hat")
#             if attributes.get("has_hood"):
#                 detected_attrs.append("🧥 Hood")
#             if attributes.get("has_glasses"):
#                 detected_attrs.append("👓 Glasses")
            
#             if detected_attrs:
#                 print(f"   Attributes: {', '.join(detected_attrs)}")
#             else:
#                 print("   Attributes: None detected")
            
#             if carried_objs:
#                 print(f"   Objects Carried: {', '.join(carried_objs)}")
#             else:
#                 print("   Objects Carried: None detected")

#         print(f"\n{'='*80}\n")
#         return matches

#     except Exception as e:
#         print(f"❌ Error during query: {e}")
#         import traceback
#         traceback.print_exc()
#         return []


# # ========================
# # Interactive Query Loop (COMMENTED OUT)
# # ========================
# # Disabled for Temporal worker: pipeline now exits without waiting for user input
# # 
# # print("\n" + "🎤"*40)
# # print("\n📝 INTERACTIVE QUERY MODE")
# # print("=" * 80)
# # print("Enter text prompts to search for people or objects in the video.")
# # print("Examples:")
# # print("  PERSON QUERIES:")
# # print("    - 'person holding a mobile phone'")
# # print("    - 'person wearing a red shirt'")
# # print("    - 'person with glasses'")
# # print("  OBJECT QUERIES:")
# # print("    - 'laptop'")
# # print("    - 'find me a phone'")
# # print("    - 'backpack'")
# # print("Type 'quit' or 'exit' to end.\n")
# # print("=" * 80 + "\n")
# # 
# # # Keep querying until user exits
# # while True:
# #     try:
# #         user_prompt = input("🔎 Enter your query: ").strip()
# #         
# #         if user_prompt.lower() in ['quit', 'exit', 'q']:
# #             print("\n✅ Exiting query mode. Goodbye!")
# #             break
# #         
# #         if not user_prompt:
# #             print("⚠ Empty prompt. Please try again.\n")
# #             continue
# #         
# #         # Smart routing: detect if query is for objects only or persons
# #         query_lower = user_prompt.lower()
# #         is_object_only = (
# #             # Query is object-only if it mentions object names without "person"
# #             ("person" not in query_lower and "people" not in query_lower) and
# #             any(obj_keyword in query_lower for obj_keyword in [
# #                 "laptop", "computer", "phone", "mobile", "cell",
# #                 "backpack", "bag", "handbag", "suitcase", "luggage"
# #             ])
# #         )
# #         
# #         # Route to appropriate query function
# #         if is_object_only:
# #             print("   → Detected OBJECT query, searching object_tracks...")
# #             matches = query_objects_by_text(user_prompt, top_k=5)
# #         else:
# #             print("   → Detected PERSON query, searching person_tracks...")
# #             matches = query_tracklets_by_text(user_prompt, top_k=5)
# #         
# #         if matches:
# #             print("💡 KEY FINDINGS:")
# #             if is_object_only:
# #                 # Object results
# #                 for track_id, score, payload in matches:
# #                     object_type = payload.get('object_type', 'Unknown')
# #                     print(f"   • {object_type} (Track ID: {track_id}, Match: {score:.2%})")
# #             else:
# #                 # Person results
# #                 for track_id, score, payload in matches:
# #                     carried = payload.get('object_carried', [])
# #                     print(f"   • Person Track ID: {track_id} (Confidence: {score:.2%})", end="")
# #                     if carried:
# #                         print(f" - Carrying: {', '.join(carried)}")
# #                     else:
# #                         print()
# #         
# #         print()
        
#     # except KeyboardInterrupt:
#     #     print("\n\n✅ Query interrupted by user. Goodbye!")
#     #     break
#     # except Exception as e:
#     #     print(f"❌ Error: {e}\n")

# print("\n" + "="*80)
# print("🏁 Pipeline and query session complete!")
# print("="*80)


'''
yolo face detection integrated with insightface + bicubic upscaling + reid
'''
import time
from collections import deque
import numpy as np
import cv2
import torch
import warnings
warnings.filterwarnings('ignore')
import torchvision.transforms as T
from torchvision.models import resnet50
from ultralytics import YOLO
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, NamedVector, PointStruct
from dotenv import load_dotenv
import os
import sys
import qdrant_collections  # for collection constants
from PIL import Image
import uuid

# Load environment variables
load_dotenv()
# 1. Initialize the client
# Try to be resilient: support QDRANT_URL (http) or host+port for gRPC
from urllib.parse import urlparse

client = None
qdrant_url = os.environ.get("QDRANT_URL")
qdrant_host = os.environ.get("QDRANT_HOST")
qdrant_port = os.environ.get("QDRANT_PORT") or os.environ.get("QDRANT_GRPC_PORT")

try:
    if qdrant_url:
        parsed = urlparse(qdrant_url)
        # If the URL port is the gRPC port (6334) prefer gRPC transport to avoid sending HTTP to gRPC
        if parsed.port == 6334:
            host = parsed.hostname or "localhost"
            port = parsed.port
            # prefer_grpc=True forces the client to use gRPC transport
            client = QdrantClient(host=host, port=port, prefer_grpc=True)
            # perform health check before printing success
            client.get_collections()
            print(f"Connected to Qdrant (gRPC) at {host}:{port}")
        else:
            # Default: use HTTP/REST URL
            client = QdrantClient(url=qdrant_url)
            client.get_collections()
            print(f"Connected to Qdrant (HTTP) at {qdrant_url}")
    elif qdrant_host and qdrant_port:
        # assume this is gRPC configuration
        client = QdrantClient(host=qdrant_host, port=int(qdrant_port), prefer_grpc=True)
        client.get_collections()
        print(f"Connected to Qdrant (gRPC) at {qdrant_host}:{qdrant_port}")
    else:
        # Fallback to defaults (client will try localhost:6333 HTTP)
        client = QdrantClient()
        client.get_collections()
        print("Connected to Qdrant with default settings (HTTP)")

    # Quick health check
    client.get_collections()
except Exception as e:
    print("Failed to connect to Qdrant:", e)

try:
    # NOTE: Mode will be set later after REALTIME_FACE_COMPARISON is defined
    # For now, store client for later initialization
    pass
except Exception as e:
    print("Failed to initialize Qdrant:", e)

# InsightFace
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except:
    print("InsightFace not available!")
    INSIGHTFACE_AVAILABLE = False

# Optional: Real-ESRGAN for face super-resolution (if installed)
USE_REAL_ESRGAN = False
sr = None
try:
    from realesrgan import RealESRGAN
    USE_REAL_ESRGAN = True
except Exception:
    USE_REAL_ESRGAN = False

# TorchReID for ReID (preferred over ResNet50)
USE_TORCHREID = False
try:
    import torchreid
    USE_TORCHREID = True
    print("TorchReID module found")
except Exception as e:
    print(f"TorchReID not available: {e}")
    USE_TORCHREID = False

# DeepSORT for multi-object tracking (replacement for ByteTrack)
USE_DEEPSORT = False
byte_tracker = None
byte_tracker_objects = None
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    USE_DEEPSORT = True
    print("DeepSORT module found")
except Exception as e:
    print(f"DeepSORT not available: {e}")
    USE_DEEPSORT = False

# Preserve existing checks that use USE_BYTETRACK
USE_BYTETRACK = USE_DEEPSORT

class _DeepSortByteTrackCompat:
    """Compatibility wrapper to mimic ByteTrack's update API using DeepSORT.
    - update(detections, img_info, img_size) -> returns iterable of tracks
    Each returned track has attributes: track_id, tlbr (x1,y1,x2,y2)
    """
    def __init__(self, **kwargs):
        # Use 'mobilenet' embedder (built-in, works on CPU without external dependencies)
        self.require_confirmation = kwargs.get('require_confirmation', True)  # For persons: require confirmed tracks
        n_init_val = 3 if self.require_confirmation else 1  # Higher n_init for persons to avoid ID churn
        
        self.ds = DeepSort(
            embedder='mobilenet',
            embedder_gpu=False,
            max_age=kwargs.get('track_buffer', 30),
            n_init=n_init_val,
            max_iou_distance=0.7,  # Standard gating (0.7 is default, good for persons)
            max_cosine_distance=0.2
        )

    def update(self, detections, img_info=None, img_size=None, frame=None):
        # detections: Nx5 [x1,y1,x2,y2,score]
        if frame is None:
            return []
        det_list = []
        for det in detections:
            if len(det) < 5:
                continue
            x1, y1, x2, y2, conf = det[:5]
            det_list.append(([float(x1), float(y1), float(x2), float(y2)], float(conf), None))
        
        try:
            tracks = self.ds.update_tracks(det_list, frame=frame)
        except Exception as e:
            print(f"[DeepSORT Error] update_tracks failed: {e}")
            return []
        
        out = []
        for t in tracks:
            # For persons: only return confirmed tracks to avoid ID churn
            # For objects: return all tracks for immediate tracking
            if self.require_confirmation and not t.is_confirmed():
                continue
            
            tlbr = t.to_tlbr()  # [x1,y1,x2,y2]
            out.append(type('BTTrack', (), {
                'track_id': t.track_id,
                'tlbr': tlbr
            }))
        return out


# ----------------------
# CONFIG (CPU OPTIMIZED)
# ----------------------
VIDEO_PATH = "combined.mp4"
# Override from environment if provided
VIDEO_PATH = os.environ.get("VIDEO_PATH", VIDEO_PATH)
try:
    VIDEO_ID = int(os.environ.get("VIDEO_ID", "1"))
except Exception:
    VIDEO_ID = 1

# ===== FACE COMPARISON MODE =====
# Set to True for real-time face comparison during video processing (creates collections)
# Set to False for post-processing face comparison after video is complete (validates collections exist)
# Default: True (REAL-TIME) - collections are created on first run
REALTIME_FACE_COMPARISON = os.environ.get("REALTIME_FACE_COMPARISON", "false").lower() == "true"
print(f"Face comparison mode: {'REAL-TIME' if REALTIME_FACE_COMPARISON else 'POST-PROCESSING'}")

# Initialize Qdrant schema based on mode (after REALTIME_FACE_COMPARISON is defined)
try:
    schema_ok = qdrant_collections.create_qdrant_schema(client, realtime_mode=REALTIME_FACE_COMPARISON)
    if not schema_ok:
        print("❌ Qdrant schema validation failed. In POST-PROCESSING mode, run first with REALTIME_FACE_COMPARISON=true")
        sys.exit(1)
    if REALTIME_FACE_COMPARISON:
        print("✅ Qdrant schema initialized for REAL-TIME mode")
    else:
        print("✅ Qdrant schema validated for POST-PROCESSING mode")
except Exception as e:
    print(f"❌ Failed to initialize Qdrant schema: {e}")
    sys.exit(1)

REF_FACE_PATHS = ["sabbas.jpg"]

YOLO_PERSON_MODEL = "yolov8m.pt"        # your person model
YOLO_FACE_MODEL = "yolov8m-face.pt"     # recommended: yolov8n-face or yolov8m-face
YOLO_OBJECT_MODEL = "yolov8m.pt"       # general object detection (laptops, phones, bags, etc.)
DETECT_EVERY_N_FRAMES = 13  # Person detection frequency
DETECT_OBJECTS_EVERY_N_FRAMES = 5  # Object detection frequency (faster for small moving objects)

# Display scaling - adjust if video appears zoomed in/out
# 1.0 = original size, 0.5 = half size, 2.0 = double size
DISPLAY_SCALE = 0.6  # Display at 60% of original size to fit on screen better

# Object classes to detect (COCO class IDs)
# 0: person, 24: handbag, 26: backpack, 28: suitcase, 63: laptop, 67: cell phone, etc.
OBJECT_CLASSES = [24, 26, 28, 63, 67]  # handbag, backpack, suitcase, laptop, cell phone
# Set to None to detect all 80 COCO classes
# OBJECT_CLASSES = None

# thresholds (tune as needed)
# Face recognition thresholds - LOWER scores = better matches (cosine distance)
# Typical good matches: < 0.3 (excellent), 0.3-0.5 (good), 0.5-0.65 (acceptable for small faces)
# Scores > 0.70 are typically NOT matches (different people)
FACE_STRICT = 0.50  # For large, clear faces (>= 100px)
FACE_LOOSE  = 0.60  # For medium faces (60-100px)
FACE_SMALL_MAX = 0.70  # Maximum threshold for very small faces - allows recognition of distant people
REID_THRESHOLD_CPU = 0.45  # Increased to reduce ReID false positives (was 0.45)

AGGREGATION_FRAMES = 10
TRACKLET_MAX_AGE = 60  # Increased to keep tracks longer (2x for better continuity)
IOU_THRESHOLD = 0.40

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)
print("Real-ESRGAN available:", USE_REAL_ESRGAN)

# CLIP for text-image embeddings (load after DEVICE is defined)
USE_CLIP = False
clip_model = None
clip_preprocess = None
try:
    import clip
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()
    USE_CLIP = True
    print("CLIP model loaded (ViT-B/32)")
except Exception as e:
    print(f"CLIP not available: {e}")
    USE_CLIP = False

# ----------------------
# LOAD MODELS
# ----------------------
yolo_person = YOLO(YOLO_PERSON_MODEL)
yolo_face = YOLO(YOLO_FACE_MODEL)   # face detector
yolo_objects = YOLO(YOLO_OBJECT_MODEL)  # general object detector

# Initialize RealESRGAN if user has it and want to use GPU if available
if USE_REAL_ESRGAN:
    try:
        sr = RealESRGAN(device=DEVICE)
        sr.load_weights('RealESRGAN_x4plus.pth', download=False)  # ensure weights exist
        print("Real-ESRGAN initialized on", DEVICE)
    except Exception as e:
        print("Real-ESRGAN init failed:", e)
        sr = None
        USE_REAL_ESRGAN = False

# InsightFace (bigger input size for better accuracy on small faces)
fa = None
if INSIGHTFACE_AVAILABLE:
    fa = FaceAnalysis(allowed_modules=['detection', 'landmark', 'recognition'])
    print("Preparing InsightFace...")
    # -1 = CPU. If you have GPU and insightface compiled with GPU support, set ctx_id=0
    fa.prepare(ctx_id=-1, det_size=(1024, 1024))
    print("InsightFace ready.")

# ReID encoder: Try TorchReID first, fallback to ResNet50
reid_model = None
reid_tf = None
USE_TORCHREID_ACTIVE = False

if USE_TORCHREID:
    try:
        # Try to use OSNet x1_0 (lightweight and effective for ReID)
        reid_model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            pretrained=True
        )
        reid_model = reid_model.to(DEVICE).eval()
        USE_TORCHREID_ACTIVE = True
        print("✓ Using TorchReID OSNet for ReID encoding")
    except Exception as e:
        print(f"⚠ TorchReID OSNet init failed: {e}")
        print("   Falling back to ResNet50 for ReID")
        USE_TORCHREID_ACTIVE = False
        reid_model = None

# ResNet50 fallback encoder
def build_resnet_encoder():
    model = resnet50(pretrained=True).eval().to(DEVICE)
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((128, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return model, transform

# Initialize ResNet50 as fallback (always available)
reid_model_resnet, reid_tf = build_resnet_encoder()

if not USE_TORCHREID_ACTIVE:
    reid_model = reid_model_resnet  # Use ResNet50 as primary if TorchReID not available
    print("✓ Using ResNet50 for ReID encoding")
else:
    # Keep ResNet50 ready as fallback
    print("✓ ResNet50 ready as fallback for ReID encoding")

# Initialize DeepSORT if available (as ByteTrack replacement)
if USE_DEEPSORT:
    try:
        cap_temp = cv2.VideoCapture(VIDEO_PATH)
        fps = cap_temp.get(cv2.CAP_PROP_FPS) or 30.0
        cap_temp.release()
        # For persons: require confirmed tracks (n_init=3) to avoid ID churn
        byte_tracker = _DeepSortByteTrackCompat(track_buffer=60, require_confirmation=True)
        print(f"✓ DeepSORT initialized for persons (FPS: {fps:.1f}, n_init=3 for stability)")
        # For small objects like phones, use IoU-based tracking instead (simpler and faster)
        # DeepSORT's appearance embedder is not needed for small objects
        byte_tracker_objects = None  # Use fallback IoU-based tracking for objects
        print(f"✓ IoU-based tracking enabled for objects (simpler for small objects)")
    except Exception as e:
        print(f"⚠ DeepSORT initialization failed: {e}")
        USE_DEEPSORT = False
        byte_tracker = None
        byte_tracker_objects = None
else:
    print("⚠ DeepSORT not available - using IOU-based tracking only")
    byte_tracker_objects = None

def reid_encode(img):
    """Encode person image for ReID. Uses TorchReID if available, otherwise ResNet50."""
    if USE_TORCHREID_ACTIVE and reid_model is not None:
        # Use TorchReID OSNet (preferred method)
        try:
            # Convert BGR to RGB and resize for ReID
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = T.ToPILImage()(rgb_img)
            pil_img = T.Resize((256, 128))(pil_img)  # ReID standard size (width, height)
            tensor_img = T.ToTensor()(pil_img).unsqueeze(0).to(DEVICE)
            
            # Normalize for ImageNet (torchreid models expect this)
            normalize_tf = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            tensor_img = normalize_tf(tensor_img)
            
            with torch.no_grad():
                feat = reid_model(tensor_img).cpu().numpy().squeeze()
            
            feat = feat / (np.linalg.norm(feat) + 1e-8)
            return feat.astype(np.float32)
        except Exception as e:
            # If TorchReID fails, fallback to ResNet50
            pass
    
    # Fallback to ResNet50 (always available)
    if reid_tf is not None:
        try:
            x = reid_tf(img[:,:,::-1]).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feat = reid_model_resnet(x).cpu().numpy().squeeze()
            feat = feat / (np.linalg.norm(feat)+1e-8)
            return feat.astype(np.float32)
        except Exception:
            return None
    
    return None

def clip_encode(img):
    """Encode a BGR image using CLIP; returns normalized embedding or None."""
    if not USE_CLIP or clip_model is None or clip_preprocess is None:
        return None
    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        with torch.no_grad():
            image_tensor = clip_preprocess(pil_img).unsqueeze(0).to(DEVICE)
            features = clip_model.encode_image(image_tensor)
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        return features.squeeze().cpu().numpy().astype(np.float32)
    except Exception:
        return None

def get_dominant_color(crop):
    """Extract dominant clothing color from an image crop.
    Prioritizes brown detection to avoid misclassifying it as white.
    Returns a simple color name or None.
    """
    if crop is None or crop.size == 0:
        return None

    try:
        h0, w0 = crop.shape[:2]
        if h0 < 10 or w0 < 10:
            return None

        # Downscale to stabilize and blur to reduce noise
        scale_h = max(20, h0 // 2)
        scale_w = max(20, w0 // 2)
        small = cv2.resize(crop, (scale_w, scale_h), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (5, 5), 0)

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        H = hsv[:, :, 0].reshape(-1)
        S = hsv[:, :, 1].reshape(-1)
        V = hsv[:, :, 2].reshape(-1)
        
        # Calculate mean values for overall assessment
        v_mean = float(np.mean(V))
        s_mean = float(np.mean(S))

        # FIRST: Check for BROWN before white (brown can look desaturated)
        # Brown detection in darker warm hues - STRICT to avoid false positives
        # Require higher saturation and narrower hue range for brown
        brown_mask = ((H < 20) | (H > 165)) & (V >= 40) & (V < 130) & (S > 40)  # Stricter: narrower hue, higher sat, not too dark
        brown_ratio = float(np.sum(brown_mask) / H.size) if H.size > 0 else 0
        # Also detect tan/light brown: warm hues with moderate saturation
        tan_mask = ((H < 20) | (H > 165)) & (V >= 130) & (V < 180) & (S >= 35) & (S < 85)
        tan_ratio = float(np.sum(tan_mask) / H.size) if H.size > 0 else 0
        
        # Only return brown if VERY dominant (>40%) to avoid false positives from shadows
        if brown_ratio > 0.40 or tan_ratio > 0.40:
            return "brown"
        # Or if both together are strong and mean saturation suggests brown
        if (brown_ratio + tan_ratio) > 0.50 and s_mean > 35 and 50 < v_mean < 160:
            return "brown"

        # PRIORITY 1: BLACK detection (very dark)
        black_mask = (V < 60)
        black_ratio = float(np.sum(black_mask) / H.size) if H.size > 0 else 0
        # Allow slightly brighter blacks to count if saturation is low (matte black) and brown is not dominant
        if (v_mean < 65 and s_mean < 85 and brown_ratio < 0.10) or black_ratio > 0.25:
            return "black"

        # PRIORITY 2: WHITE detection (very bright + desaturated)
        # White = very high brightness + very low saturation, but NOT warm-hued
        white_mask = (V > 190) & (S < 50)
        white_ratio = float(np.sum(white_mask) / H.size) if H.size > 0 else 0
        
        # Only white if no warm/brown hues dominate
        warm_hues = ((H < 30) | (H > 160))
        warm_ratio = float(np.sum(warm_hues) / H.size) if H.size > 0 else 0
        
        # Return white ONLY if very bright, very desaturated, AND not dominated by warm hues
        if v_mean > 205 and s_mean < 50 and warm_ratio < 0.30:
            return "white"
        if v_mean > 200 and s_mean < 40 and warm_ratio < 0.25:
            return "white"
        if white_ratio > 0.35 and s_mean < 45 and warm_ratio < 0.25:
            return "white"
        
        # PRIORITY 3: GRAY detection (only if clearly not white/brown)
        # Gray = low saturation, moderate brightness (neither white nor black nor brown)
        gray_mask = (S < 45) & (V >= 50) & (V <= 190)
        gray_ratio = float(np.sum(gray_mask) / H.size) if H.size > 0 else 0
        
        # Gray only if saturation is very low and brightness is moderate, and not brown
        if gray_ratio > 0.45 and v_mean < 195 and brown_ratio < 0.15:
            return "gray"
        if s_mean < 35 and 80 < v_mean < 190 and brown_ratio < 0.15:
            return "gray"

        # Filter to colorful pixels for hue analysis
        valid = (V > 35) & (V < 245) & (S > 30)
        if np.sum(valid) < 80:
            # Not enough colorful pixels: choose closest achromatic class by averages
            if v_mean < 55:
                return "black"
            if v_mean > 195 and s_mean < 45 and brown_ratio < 0.20:
                return "white"
            return "gray" if s_mean < 50 else None

        Hv = H[valid]
        Sv = S[valid]
        Vv = V[valid]

        # Histogram of hue
        bins = np.array([0, 10, 25, 35, 80, 100, 130, 150, 170, 181])
        hist, _ = np.histogram(Hv, bins=bins)
        idx = int(np.argmax(hist))

        color_map = {
            0: "red", 1: "orange", 2: "yellow", 3: "green",
            4: "cyan", 5: "blue", 6: "purple", 7: "magenta", 8: "red",
        }

        if np.mean(Sv) < 55:
            return "gray"

        dominant_color = color_map.get(idx, None)
        
        # BROWN OVERRIDE: If dominant is orange/red but brightness is low or saturation suggests brown
        if dominant_color in ("orange", "red") and (np.mean(Vv) < 140 or np.mean(Sv) < 50):
            return "brown"
        
        # WHITE OVERRIDE: only for truly desaturated warm tones
        if dominant_color in ("orange", "yellow") and np.mean(Sv) < 65 and np.mean(Vv) > 210:
            return "white"
        
        return dominant_color
    except Exception:
        return None

def mask_upper_by_face(crop_person, face_boxes_in_frame, person_box):
    """Mask out face area from the upper half of the person crop to avoid skin tones.
    Returns a modified upper_part image with the face region blacked out.
    face_boxes_in_frame: list of (x1,y1,x2,y2) in full-frame coords
    person_box: (x1,y1,x2,y2) of person in full-frame coords
    """
    if crop_person is None or crop_person.size == 0:
        return None
    h, w = crop_person.shape[:2]
    upper = crop_person[:h//2, :].copy()
    if not face_boxes_in_frame:
        return upper
    px1, py1, px2, py2 = person_box
    # Iterate faces that intersect person_box and map to crop coordinates
    for (fx1, fy1, fx2, fy2) in face_boxes_in_frame:
        # Check intersection with person box
        ix1 = max(px1, fx1); iy1 = max(py1, fy1)
        ix2 = min(px2, fx2); iy2 = min(py2, fy2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        # Map to crop local coordinates
        lx1 = max(0, ix1 - px1)
        ly1 = max(0, iy1 - py1)
        lx2 = min(w, ix2 - px1)
        ly2 = min(h//2, iy2 - py1)  # only mask within upper half
        if lx2 > lx1 and ly2 > ly1:
            upper[ly1:ly2, lx1:lx2] = 0  # black out face region
    return upper

def detect_person_attributes(crop, face_crop=None):
    """Detect person attributes from crop image with strict validation.
    
    Returns dict: {
        "has_hat": bool,       # Detected hat or head covering
        "has_hood": bool,      # Detected hood on clothing
        "has_glasses": bool,   # Detected glasses/sunglasses
    }
    
    Uses conservative heuristics to minimize false positives:
    - hat: Detects distinct dark/light regions at top of person crop
    - hood: Detects pointed/curved top region with strong edge definition
    - glasses: Looks for symmetric dark oval regions in face area
    """
    attributes = {
        "has_hat": False,
        "has_hood": False,
        "has_glasses": False,
    }
    
    if crop is None or crop.size == 0:
        return attributes
    
    try:
        crop_h, crop_w = crop.shape[:2]
        
        # === HAT DETECTION (STRICT) ===
        # Only detect if there's a DISTINCT shape at top (not just texture)
        top_region = crop[:max(1, crop_h // 6), :]  # Top 16% only
        if top_region.size > 0 and top_region.shape[0] > 5:
            gray = cv2.cvtColor(top_region, cv2.COLOR_BGR2GRAY)
            # Use higher threshold for Laplacian - only strong edge/shape definition counts
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = np.var(laplacian)
            # VERY HIGH THRESHOLD - only distinct hat shapes
            # Normal texture variance: 50-150
            # Hat shape variance: 300+
            if variance > 400:
                # Additional check: ensure it's a localized blob, not just noisy texture
                edges = cv2.Canny(gray, 50, 150)
                edge_pixels = np.sum(edges > 0)
                # Hat should have concentrated edges at top
                if edge_pixels > top_region.size * 0.05:  # At least 5% edges
                    attributes["has_hat"] = True
        
        # === HOOD DETECTION (STRICTER) ===
        # Only mark hood if there is a strong peaked silhouette AND hat is not already detected
        top_quarter = crop[:crop_h // 5, :]
        if top_quarter.size > 0 and top_quarter.shape[0] > 5 and not attributes["has_hat"]:
            gray_top = cv2.cvtColor(top_quarter, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_top, 60, 180)  # Slightly stricter edges

            # Peak check at top 2 rows (hood tip) and shoulders (rows 3-6)
            top_rows = edges[0:2, :]
            shoulder_rows = edges[2:6, :]
            mid_col = top_rows.shape[1] // 2
            left_top = np.sum(top_rows[:, :mid_col])
            right_top = np.sum(top_rows[:, mid_col:])
            left_shoulder = np.sum(shoulder_rows[:, :mid_col])
            right_shoulder = np.sum(shoulder_rows[:, mid_col:])

            edge_density = np.sum(edges > 0) / edges.size

            # Hood criteria (much stricter):
            # 1) High edge density > 0.22
            # 2) Symmetric peak at top (both sides > 12 edge pixels)
            # 3) Shoulders also have edges (> 20 per side) to indicate fabric fold
            if (
                edge_density > 0.22
                and min(left_top, right_top) > 12
                and min(left_shoulder, right_shoulder) > 20
            ):
                attributes["has_hood"] = True
        
        # === GLASSES DETECTION (STRICT) ===
        if face_crop is not None and face_crop.size > 0:
            face_h, face_w = face_crop.shape[:2]
            if face_h > 20 and face_w > 20:
                # Analyze ONLY eye region (40-60% of face height, middle width)
                eye_start = int(face_h * 0.35)
                eye_end = int(face_h * 0.65)
                eye_left = int(face_w * 0.2)
                eye_right = int(face_w * 0.8)
                
                eye_region = face_crop[eye_start:eye_end, eye_left:eye_right]
                if eye_region.size > 100:
                    gray_eyes = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
                    # Look for VERY dark pixels (glasses frames are black)
                    very_dark_pixels = np.sum(gray_eyes < 60)  # Very dark threshold
                    dark_ratio = very_dark_pixels / gray_eyes.size
                    # HIGH threshold: >25% very dark pixels in eye region
                    if dark_ratio > 0.25:
                        attributes["has_glasses"] = True
        
        return attributes
        
    except Exception as e:
        # If detection fails, return neutral attributes (all False)
        return attributes

def extract_face_embedding_optimized(crop_person, face_boxes_frame, person_box):
    """
    Extract face embedding from person crop using InsightFace.
    
    Args:
        crop_person: Person crop image (BGR)
        face_boxes_frame: List of face boxes detected in the frame
        person_box: Person bounding box [x1, y1, x2, y2]
    
    Returns:
        tuple: (face_embedding, face_size) where face_size is the width of the detected face
    """
    if not fa or not INSIGHTFACE_AVAILABLE:
        return None, 0
    
    try:
        # Use InsightFace to detect faces in the person crop
        faces = fa.get(crop_person)
        
        if faces and len(faces) > 0:
            # Use the first (best) face detected
            face = faces[0]
            face_emb = normalize(np.array(face.embedding))
            # Face size is the width of the bounding box
            face_size = int(face.bbox[2] - face.bbox[0])
            return face_emb, face_size
        else:
            # No face detected
            return None, 0
            
    except Exception as e:
        print(f"⚠ Face embedding extraction failed: {e}")
        return None, 0

# ----------------------
# Build reference embeddings
# ----------------------
ref_face_embs = []
ref_reid_embs = []

def normalize(v):
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v)+1e-8)

if REALTIME_FACE_COMPARISON:
    # REAL-TIME MODE: Load reference face embeddings for comparison during processing
    print("\n" + "="*80)
    print("🔴 REAL-TIME FACE COMPARISON MODE")
    print("="*80)
    print("Loading reference face images for real-time verification...")
    
    for path in REF_FACE_PATHS:
        img = cv2.imread(path)
        if img is None:
            print("Warning: could not load reference image", path)
            continue

        # face embedding
        face_emb = None
        if fa:
            faces = fa.get(img)
            if faces and len(faces) > 0:
                face_emb = normalize(np.array(faces[0].embedding))
                ref_face_embs.append(face_emb)
                print(f"✓ Reference face loaded: {len(face_emb)}D")
            else:
                print(f"⚠ Warning: No face detected in reference image {path} - face recognition will not work!")
        else:
            print("⚠ Warning: InsightFace not available - face recognition disabled!")

        # reid embedding
        reid_emb = reid_encode(img)
        if reid_emb is not None:
            ref_reid_embs.append(reid_emb)
            print(f"✓ Reference ReID loaded: {len(reid_emb)}D")

    # Determine if reference is face-only (affects verification strategy)
    USE_REID_FOR_ALL_TRACKS = True  # Enable ReID collection for all tracks regardless of verification status
    USE_REID_VERIFICATION = len(ref_face_embs) > 0 and len(ref_reid_embs) > 0
    # If we have face embeddings, enable strict face-based verification
    if len(ref_face_embs) > 0:
        print("⚠ IMPORTANT: Reference contains face images. STRICT face verification enabled.")
        print("   Only face matches = VERIFIED. ReID collected for all tracks but doesn't affect verification.")
        USE_REID_VERIFICATION = True
    print("="*80 + "\n")
else:
    # POST-PROCESSING MODE: Skip reference loading, all tracks stored as unverified
    print("\n" + "="*80)
    print("🟢 POST-PROCESSING FACE COMPARISON MODE")
    print("="*80)
    print("All person tracks will be stored in Qdrant WITHOUT verification.")
    print("Use compare_face_after_processing() after video processing to find matches.")
    print("="*80 + "\n")
    USE_REID_FOR_ALL_TRACKS = True
    USE_REID_VERIFICATION = False

# ===== DEBUG: Identify embedding dimensions =====
FACE_EMBEDDING_DIM = len(ref_face_embs[0]) if len(ref_face_embs) > 0 else None
REID_EMBEDDING_DIM = len(ref_reid_embs[0]) if len(ref_reid_embs) > 0 else None

print(f"✓ Embeddings: Face {FACE_EMBEDDING_DIM}D, ReID {REID_EMBEDDING_DIM}D")

# ----------------------
# Helper functions
# ----------------------
def cos_dist(a, b):
    if a is None or b is None: return 1.0
    return 1.0 - np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8)

def upscale_bicubic(img, factor=2):
    """Upscale image using bicubic interpolation."""
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

def sr_enhance(img, factor=2):
    """Use Real-ESRGAN if available, otherwise bicubic upscale.
    For very small faces, use larger upscale factor."""
    if USE_REAL_ESRGAN and sr is not None:
        try:
            # Real-ESRGAN expects RGB
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            enhanced = sr.predict(rgb)
            # convert back to BGR
            enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
            return enhanced_bgr
        except Exception:
            return upscale_bicubic(img, factor=factor)
    else:
        return upscale_bicubic(img, factor=factor)

def is_face_match(emb, face_w):
    if emb is None or len(ref_face_embs)==0:
        return False, None

    # Adaptive thresholds based on face size
    # IMPORTANT: Lower cosine distance = better match
    # Good matches: < 0.3 (excellent), 0.3-0.5 (good), 0.5-0.68 (acceptable for small faces)
    # Scores > 0.70 are typically NOT matches (different people)
    
    # For small/distant faces, we need to be more lenient because:
    # 1. Embeddings from small faces are inherently less accurate
    # 2. But we still need to prevent false positives
    if face_w >= 100:
        # Large clear faces - can use strict threshold
        thr = FACE_STRICT
    elif face_w >= 60:
        # Medium faces - slightly more lenient but still strict
        thr = FACE_LOOSE
    elif face_w >= 40:
        # Small-medium faces (40-60px) - more lenient for distant faces
        # This is the critical range for distant recognition
        thr = min(FACE_LOOSE + 0.08, FACE_SMALL_MAX)  # 0.68 max (was 0.65)
    elif face_w >= 25:
        # Small faces (25-40px) - even more lenient but capped
        thr = min(FACE_LOOSE + 0.10, FACE_SMALL_MAX)  # 0.70 max (was 0.68)
    else:
        # Very small faces (< 25px) - most lenient but still capped
        # These are very challenging, so we allow higher threshold
        thr = min(FACE_LOOSE + 0.12, FACE_SMALL_MAX)  # 0.72 max (was 0.70)
    
    best = min([cos_dist(emb, r) for r in ref_face_embs])
    is_match = best < thr
    
    # Safety check: reject very high scores (more lenient for small faces)
    # For faces < 40px, allow up to 0.70; for 40-60px, allow up to 0.68; larger faces cap at 0.65
    if face_w < 25:
        max_allowed = 0.72
    elif face_w < 40:
        max_allowed = 0.70
    elif face_w < 60:
        max_allowed = 0.68  # More lenient for 40-60px range
    else:
        max_allowed = 0.65
    
    if best > max_allowed:
        is_match = False
    
    return is_match, best

def is_reid_match(emb):
    if emb is None or len(ref_reid_embs)==0:
        return False, None

    best = min([cos_dist(emb, r) for r in ref_reid_embs])
    return best < REID_THRESHOLD_CPU, best

def iou(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    interW = max(0, xB-xA); interH = max(0, yB-yA)
    inter = interW * interH
    areaA = (a[2]-a[0])*(a[3]-a[1])
    areaB = (b[2]-b[0])*(b[3]-b[1])
    if areaA + areaB - inter == 0: return 0
    return inter / (areaA + areaB - inter)

def box_center(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def box_inside(inner, outer):
    # return True if center of inner lies inside outer
    cx, cy = box_center(inner)
    x1,y1,x2,y2 = outer
    return (cx >= x1 and cx <= x2 and cy >= y1 and cy <= y2)

def apply_nms(boxes, scores, iou_threshold=0.5):
    """Apply Non-Maximum Suppression to filter overlapping boxes.
    Returns indices of boxes to keep."""
    if len(boxes) == 0:
        return []
    
    # Convert boxes to format for NMS: [x1, y1, x2, y2]
    boxes_array = np.array(boxes, dtype=np.float32)
    scores_array = np.array(scores, dtype=np.float32)
    
    # Use OpenCV's NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores_array, score_threshold=0.0, nms_threshold=iou_threshold)
    
    if len(indices) == 0:
        return []
    
    return indices.flatten().tolist()

def merge_object_tracklets(object_tracklets, iou_thresh=0.6):
    """Merge object tracklets that represent the same object (same class, high IoU).
    Keeps the oldest ID and merges bbox/conf history to avoid ID churn for small objects.
    """
    try:
        oids = list(object_tracklets.keys())
        to_merge = []
        for i in range(len(oids)):
            oid_i = oids[i]
            ti = object_tracklets.get(oid_i)
            if ti is None:
                continue
            bi = ti.get_latest_bbox()
            for j in range(i+1, len(oids)):
                oid_j = oids[j]
                tj = object_tracklets.get(oid_j)
                if tj is None:
                    continue
                # Only merge same class
                if ti.class_name != tj.class_name:
                    continue
                bj = tj.get_latest_bbox()
                if bi is None or bj is None:
                    continue
                if iou(bi, bj) >= iou_thresh:
                    # Merge j into i (keep smaller id for stability)
                    keep_id, drop_id = (oid_i, oid_j) if int(oid_i) <= int(oid_j) else (oid_j, oid_i)
                    to_merge.append((keep_id, drop_id))
        # Execute merges
        for keep_id, drop_id in to_merge:
            if keep_id not in object_tracklets or drop_id not in object_tracklets:
                continue
            keep = object_tracklets[keep_id]
            drop = object_tracklets[drop_id]
            # Append drop history into keep
            for b in drop.bboxes:
                keep.bboxes.append(b)
            for c in drop.confidences:
                keep.confidences.append(c)
            keep.last_frame = max(keep.last_frame, drop.last_frame)
            # Remove dropped id
            del object_tracklets[drop_id]
    except Exception:
        pass

def reassign_small_object_ids(object_tracklets, class_keywords, iou_thresh=0.4, max_center_dist=100, require_same_class=False):
    """Map newly created tracklets to existing ones to prevent ID churn for specific object classes.
    Prefer the oldest existing ID when two tracks overlap strongly OR are close.
    Uses predicted positions for fast-moving objects.
    
    Args:
        object_tracklets: Dictionary of object tracklets
        class_keywords: List of keywords to match in class_name (e.g., ["phone", "cell phone"])
        iou_thresh: IoU threshold for considering objects as same
        max_center_dist: Maximum center distance in pixels for merging
    """
    try:
        # Collect active tracks matching any keyword
        def matches_keywords(class_name):
            return any(keyword.lower() in class_name.lower() for keyword in class_keywords)
        
        objects = [(oid, t) for oid, t in object_tracklets.items() if matches_keywords(t.class_name)]
        # Sort by first_frame to prefer older IDs
        objects.sort(key=lambda x: x[1].first_frame)
        def center(b):
            x1,y1,x2,y2 = b
            return ((x1+x2)/2.0, (y1+y2)/2.0)
        for i in range(len(objects)):
            oid_i, ti = objects[i]
            bi = ti.get_latest_bbox()
            if bi is None:
                continue
            ci = center(bi)
            # Also get predicted position for fast motion
            bi_pred = ti.predict_position(frames_ahead=1)
            ci_pred = center(bi_pred) if bi_pred else ci
            
            for j in range(i+1, len(objects)):
                oid_j, tj = objects[j]
                bj = tj.get_latest_bbox()
                if bj is None:
                    continue
                cj = center(bj)
                bj_pred = tj.predict_position(frames_ahead=1)
                cj_pred = center(bj_pred) if bj_pred else cj
                
                # Optionally enforce same class (backpack vs handbag should not merge unless chosen)
                if require_same_class and ti.class_name.lower() != tj.class_name.lower():
                    continue

                # Check multiple criteria (OR logic for aggressive matching)
                iou_val = iou(bi, bj)
                dist_current = (abs(ci[0]-cj[0])**2 + abs(ci[1]-cj[1])**2) ** 0.5
                dist_predicted = (abs(ci_pred[0]-cj_pred[0])**2 + abs(ci_pred[1]-cj_pred[1])**2) ** 0.5
                
                # Calculate size similarity (helps match same object across frames)
                area_i = (bi[2] - bi[0]) * (bi[3] - bi[1])
                area_j = (bj[2] - bj[0]) * (bj[3] - bj[1])
                size_ratio = min(area_i, area_j) / max(area_i, area_j) if max(area_i, area_j) > 0 else 0
                
                # Merge if ANY of these conditions met (very aggressive)
                should_merge = (
                    iou_val >= iou_thresh or  # overlapping
                    dist_current <= max_center_dist or  # close now
                    dist_predicted <= max_center_dist or  # will be close soon
                    (iou_val >= 0.2 and dist_current <= max_center_dist * 1.5) or  # some overlap + nearby
                    (dist_current <= max_center_dist * 1.8 and size_ratio >= 0.5)  # similar size + close
                )
                
                if should_merge:
                    keep_id, drop_id = oid_i, oid_j
                    if drop_id in object_tracklets and keep_id in object_tracklets:
                        keep = object_tracklets[keep_id]
                        drop = object_tracklets[drop_id]
                        for b in drop.bboxes:
                            keep.bboxes.append(b)
                        for c in drop.confidences:
                            keep.confidences.append(c)
                        keep.last_frame = max(keep.last_frame, drop.last_frame)
                        # Merge velocity too
                        if drop.velocity:
                            keep.velocity = drop.velocity
                        del object_tracklets[drop_id]
    except Exception:
        pass

# ----------------------
# Qdrant insertion function for verified tracklets
# ----------------------
import uuid
from datetime import datetime

def insert_tracklet_to_qdrant(client, tracklet, video_id=1, segment_id=None, frame_rate=30.0):
    """
    Insert a tracklet into Qdrant person_tracks collection.
    
    Stores averaged face and ReID embeddings with metadata for similarity search.
    Works for both verified and unverified tracklets.
    
    Args:
        client: QdrantClient instance
        tracklet: Tracklet object (verified or unverified) with avg face/reid embeddings
        video_id: Video ID from PostgreSQL videos table
        segment_id: Optional segment ID for video_segments table link
        frame_rate: Video frame rate for time calculations
    
    Returns:
        bool: True if insertion succeeded, False otherwise
    """
    # Skip inserts in POST-PROCESSING mode
    try:
        if not REALTIME_FACE_COMPARISON:
            return False
    except NameError:
        pass
    if not client:
        return False
    
    try:
        # Get averaged embeddings (may be None for unverified tracklets)
        face_avg = tracklet.avg_face()
        reid_avg = tracklet.avg_reid()
        
        # For unverified tracklets, we still need at least ReID embedding to insert
        # If both are None, skip insertion (no useful data)
        if face_avg is None and reid_avg is None:
            print(f"⚠ Skipping tracklet {tracklet.id} - no embeddings available")
            return False
        
        # Use zero vectors as fallback if embeddings are missing
        if face_avg is None:
            face_avg = np.zeros(512, dtype=np.float32)
        if reid_avg is None:
            reid_avg = np.zeros(512, dtype=np.float32)
        
        # Generate unique ID for this tracklet entry
        point_id = str(uuid.uuid4())
        
        # Calculate time range based on when the tracklet first/last appeared
        start_frame = tracklet.first_frame if hasattr(tracklet, 'first_frame') else 0
        end_frame = tracklet.last_frame if hasattr(tracklet, 'last_frame') else start_frame
        num_frames = max(1, end_frame - start_frame + 1)
        
        # Estimate time in seconds using frame rate
        start_time_sec = start_frame / max(frame_rate, 1.0)
        end_time_sec = end_frame / max(frame_rate, 1.0)
        
        # Format as HH:MM:SS.mmm (keep milliseconds to avoid truncation)
        def seconds_to_hms_ms(secs):
            h = int(secs // 3600)
            m = int((secs % 3600) // 60)
            s = int(secs % 60)
            ms = int((secs - int(secs)) * 1000)
            return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
        
        start_time_str = seconds_to_hms_ms(start_time_sec)
        end_time_str = seconds_to_hms_ms(end_time_sec)
        
        # Build payload per spec (NO camera_id)
        payload = {
            "video_id": video_id,
            "track_id": tracklet.id,
            "start_time": start_time_str,
            "end_time": end_time_str,
            "num_frames": num_frames,
            "avg_confidence": 0.85,  # placeholder, can extract from tracklet if available
            "timestamp": datetime.now().isoformat(),
        }
        
        # Add optional segment_id if provided
        if segment_id is not None:
            payload["segment_id"] = segment_id
        
        # Optional fields (set None/empty for now, extend later with attribute detection)
        payload["person_gender"] = None
        payload["upper_color"] = tracklet.get_dominant_upper_color()
        payload["lower_color"] = tracklet.get_dominant_lower_color()
        payload["attributes"] = tracklet.get_attribute_summary()  # Dictionary of detected attributes
        payload["object_carried"] = tracklet.carried_summary()
        payload["verified"] = tracklet.verified  # Indicate if this tracklet was verified against reference
        
        # Convert embeddings to lists for Qdrant
        face_vec = face_avg.tolist() if isinstance(face_avg, np.ndarray) else list(face_avg)
        reid_vec = reid_avg.tolist() if isinstance(reid_avg, np.ndarray) else list(reid_avg)
        
        # multi_vec: prefer CLIP embedding, fallback to face embedding (both 512D now)
        clip_avg = tracklet.avg_clip()

        def pad_to_512(vec):
            """Pad vector to 512D by appending zeros."""
            vec_list = vec.tolist() if isinstance(vec, np.ndarray) else list(vec)
            if len(vec_list) >= 512:
                return vec_list[:512]
            return vec_list + [0.0] * (512 - len(vec_list))
        
        if clip_avg is not None:
            multi_vec = pad_to_512(clip_avg)
        else:
            multi_vec = pad_to_512(face_avg)
        
        # Build vectors dict for NamedVectors (per spec: all 512D - InsightFace, TorchReID, CLIP ViT-B/32)
        vectors = {
            "face_vec": face_vec,          # 512D (InsightFace)
            "reid_vec": reid_vec,          # 512D (TorchReID)
            "multi_vec": multi_vec,        # 512D (CLIP ViT-B/32 or padded face_vec)
        }
        
        # Insert to Qdrant (quiet success)
        point = PointStruct(
            id=point_id,
            vector=vectors,  # NamedVectors
            payload=payload
        )
        client.upsert(
            collection_name="person_tracks",
            points=[point]
        )
        
        # Simple success message
        status = "VERIFIED" if tracklet.verified else "UNVERIFIED"
        print(f"✓ Inserted {status} tracklet {tracklet.id} to Qdrant")
        return True
        
    except Exception as e:
        print(f"⚠ Failed to insert tracklet {tracklet.id} to Qdrant: {e}")
        return False

def insert_object_track_to_qdrant(client, obj_track, video_id=1, segment_id=None, frame_rate=30.0):
    """
    Insert an object tracklet into Qdrant object_tracks collection.
    
    Stores object tracking data with metadata for similarity search and retrieval.
    
    Args:
        client: QdrantClient instance
        obj_track: ObjectTracklet object with tracking history
        video_id: Video ID from PostgreSQL videos table
        segment_id: Optional segment ID for video_segments table link
        frame_rate: Video frame rate for time calculations
    
    Returns:
        bool: True if insertion succeeded, False otherwise
    """
    # Skip inserts in POST-PROCESSING mode
    try:
        if not REALTIME_FACE_COMPARISON:
            return False
    except NameError:
        pass
    if not client:
        return False
    
    try:
        # Generate unique ID for this object track entry
        point_id = str(uuid.uuid4())
        
        # Calculate time range
        start_frame = obj_track.first_frame
        end_frame = obj_track.last_frame
        num_frames = max(1, end_frame - start_frame + 1)
        
        start_time_sec = start_frame / max(frame_rate, 1.0)
        end_time_sec = end_frame / max(frame_rate, 1.0)
        
        def seconds_to_hms_ms(secs):
            h = int(secs // 3600)
            m = int((secs % 3600) // 60)
            s = int(secs % 60)
            ms = int((secs - int(secs)) * 1000)
            return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
        
        start_time_str = seconds_to_hms_ms(start_time_sec)
        end_time_str = seconds_to_hms_ms(end_time_sec)
        
        # Build payload for object_tracks collection
        payload = {
            "video_id": video_id,
            "track_id": obj_track.id,
            "object_type": obj_track.class_name,
            "object_color": obj_track.get_dominant_color(),  # Add color
            "start_time": start_time_str,
            "end_time": end_time_str,
            "num_frames": num_frames,
            "avg_confidence": obj_track.get_avg_confidence(),
            "timestamp": datetime.now().isoformat(),
        }
        
        if segment_id is not None:
            payload["segment_id"] = segment_id
        
        # Use averaged CLIP embeddings if available, otherwise zeros
        avg_clip_emb = obj_track.avg_clip()
        if avg_clip_emb is not None:
            # CLIP ViT-B/32 outputs 512D - no padding needed
            if len(avg_clip_emb) < 512:
                object_vec = np.concatenate([avg_clip_emb, np.zeros(512 - len(avg_clip_emb), dtype=np.float32)]).tolist()
            else:
                object_vec = avg_clip_emb[:512].tolist()
            multi_vec = object_vec  # Use same embedding for multi_vec
        else:
            # Fallback to zeros if no CLIP embeddings collected
            object_vec = np.zeros(512, dtype=np.float32).tolist()
            multi_vec = np.zeros(512, dtype=np.float32).tolist()
        
        vectors = {
            "object_vec": object_vec,  # 512D (CLIP ViT-B/32)
            "multi_vec": multi_vec,     # 512D (CLIP ViT-B/32)
        }
        
        # Insert into Qdrant
        point = PointStruct(
            id=point_id,
            vector=vectors,
            payload=payload
        )
        
        # Quiet success
        client.upsert(
            collection_name="object_tracks",
            points=[point]
        )
        
        print(f"✓ Inserted object {obj_track.id} ({obj_track.class_name}) to Qdrant")
        return True
        
    except Exception as e:
        print(f"⚠ Failed to insert object track {obj_track.id} to Qdrant: {e}")
        return False

# ----------------------
# Tracklet class (for persons)
# ----------------------
class Tracklet:
    def __init__(self, tid, bbox, frame_idx):
        self.id = tid
        self.bboxes = deque(maxlen=AGGREGATION_FRAMES)
        self.bboxes.append(bbox)
        self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.face_embs = []
        self.face_sizes = []  # Store actual detected face sizes
        self.reid_embs = []
        self.clip_embs = []
        self.carried_objects = deque(maxlen=50)  # recent object class names observed with this person
        self.upper_colors = deque(maxlen=20)  # Track detected upper body colors
        self.lower_colors = deque(maxlen=20)  # Track detected lower body colors
        
        # Attributes dictionary: tracks person attributes (has_hat, has_glasses, etc.)
        self.attributes = {
            "has_hat": deque(maxlen=20),       # Boolean: person wearing hat
            "has_hood": deque(maxlen=20),      # Boolean: person wearing hood
            "has_glasses": deque(maxlen=20),   # Boolean: person wearing glasses
        }
        
        self.verified = False
        self.inserted = False  # set True once pushed to DB
        self.tracker = None

    def update(self, bbox, idx, face_emb=None, reid_emb=None, face_size=0, clip_emb=None, carried=None, upper_color=None, lower_color=None, attributes=None):
        self.bboxes.append(bbox)
        self.last_frame = idx
        if face_emb is not None: 
            self.face_embs.append(face_emb)
            if face_size > 0:
                self.face_sizes.append(face_size)
        if reid_emb is not None: self.reid_embs.append(reid_emb)
        if clip_emb is not None: self.clip_embs.append(clip_emb)
        if carried:
            # carried can be a list or single string
            if isinstance(carried, (list, tuple)):
                for name in carried:
                    self.carried_objects.append(name)
            else:
                self.carried_objects.append(carried)
        if upper_color is not None:
            self.upper_colors.append(upper_color)
        if lower_color is not None:
            self.lower_colors.append(lower_color)
        
        # Update attributes (dict of attribute_name -> bool)
        if attributes:
            for attr_name, attr_value in attributes.items():
                if attr_name in self.attributes and isinstance(attr_value, bool):
                    self.attributes[attr_name].append(attr_value)

    def avg_face(self):
        if not self.face_embs: return None
        avg = np.mean(self.face_embs, axis=0)
        return normalize(avg)

    def avg_reid(self):
        if not self.reid_embs: return None
        avg = np.mean(self.reid_embs, axis=0)
        return normalize(avg)
    
    def avg_clip(self):
        if not self.clip_embs:
            return None
        avg = np.mean(self.clip_embs, axis=0)
        return normalize(avg)
    
    def avg_face_size(self):
        """Get average detected face size, or estimate from bbox if no sizes recorded"""
        if self.face_sizes:
            return int(np.mean(self.face_sizes))
        # Fallback: estimate from last bbox (face is roughly upper 1/3 of person height)
        if self.bboxes:
            last = self.bboxes[-1]
            return max(20, (last[2] - last[0]) // 4)  # Conservative estimate
        return 0
    
    def carried_summary(self):
        """Return filtered list of carried objects based on recent evidence.

        We require that an object name appears at least 3 times in the recent
        history and was also observed in the last 10 entries. This prevents
        transient background detections from lingering.
        """
        if not self.carried_objects:
            return []
        counts = {}
        for name in self.carried_objects:
            counts[name] = counts.get(name, 0) + 1
        recent = list(self.carried_objects)[-10:]
        recent_set = set(recent)
        result = []
        for name, c in counts.items():
            if c >= 3 and name in recent_set:
                result.append(name)
        return result
    
    def get_dominant_upper_color(self):
        """Get most frequently observed upper body color."""
        if not self.upper_colors:
            return None
        color_counts = {}
        for color in self.upper_colors:
            color_counts[color] = color_counts.get(color, 0) + 1
        return max(color_counts, key=color_counts.get) if color_counts else None
    
    def get_dominant_lower_color(self):
        """Get most frequently observed lower body color."""
        if not self.lower_colors:
            return None
        color_counts = {}
        for color in self.lower_colors:
            color_counts[color] = color_counts.get(color, 0) + 1
        return max(color_counts, key=color_counts.get) if color_counts else None
    
    def get_attribute_summary(self):
        """Get summary of detected attributes (most frequently observed values).
        Returns dict of attribute_name -> bool (True if detected, False if not, None if unknown)"""
        attr_summary = {}
        for attr_name, attr_deque in self.attributes.items():
            if not attr_deque:
                attr_summary[attr_name] = None
            else:
                # Count True vs False observations
                true_count = sum(1 for x in attr_deque if x)
                false_count = sum(1 for x in attr_deque if not x)
                # Return True if more often observed as True
                attr_summary[attr_name] = true_count > false_count
        return attr_summary

# ----------------------
# ObjectTracklet class (for objects like backpacks, laptops, etc.)
# ----------------------
class ObjectTracklet:
    def __init__(self, oid, class_name, bbox, confidence, frame_idx):
        self.id = oid
        self.class_name = class_name
        self.class_votes = {class_name: 1}  # Track class label votes to stabilize naming
        self.class_locked = False  # Once class is stable, don't change it easily
        self.class_lock_frame = None  # Frame when class became locked
        self.bboxes = deque(maxlen=30)  # Keep last 30 bboxes
        self.bboxes.append(bbox)
        self.confidences = deque(maxlen=30)
        self.confidences.append(confidence)
        self.last_frame = frame_idx
        self.first_frame = frame_idx
        self.velocity = None  # Track velocity for motion prediction
        self.clip_embs = []  # CLIP embeddings of object crops for semantic search
        self.colors = deque(maxlen=20)  # Track detected object colors
        self.inserted = False  # Track if already inserted to Qdrant

    def update(self, bbox, confidence, frame_idx, class_name=None, clip_emb=None, color=None):
        # Update velocity if we have previous bbox
        if len(self.bboxes) > 0:
            prev_bbox = self.bboxes[-1]
            # Calculate center displacement
            prev_cx = (prev_bbox[0] + prev_bbox[2]) / 2
            prev_cy = (prev_bbox[1] + prev_bbox[3]) / 2
            curr_cx = (bbox[0] + bbox[2]) / 2
            curr_cy = (bbox[1] + bbox[3]) / 2
            self.velocity = (curr_cx - prev_cx, curr_cy - prev_cy)
        self.bboxes.append(bbox)
        self.confidences.append(confidence)
        self.last_frame = frame_idx
        
        # Store CLIP embedding if provided
        if clip_emb is not None:
            self.clip_embs.append(clip_emb)
        
        # Store color if provided
        if color is not None:
            self.colors.append(color)
        
        # Update class vote to stabilize label - but respect locked classes
        if class_name:
            # If class is locked, only accept same class or very high confidence changes
            if self.class_locked:
                # Only change if new class has significantly more votes AND higher confidence
                if class_name != self.class_name:
                    new_conf = confidence
                    curr_avg_conf = self.get_avg_confidence()
                    # Require new class to have 0.15+ higher confidence to override locked class
                    if new_conf > curr_avg_conf + 0.15:
                        self.class_votes[class_name] = self.class_votes.get(class_name, 0) + 1
                    # Otherwise ignore the conflicting class
            else:
                # Not locked yet - accumulate votes
                self.class_votes[class_name] = self.class_votes.get(class_name, 0) + 1
                
                # Lock class once it has 3+ votes (stable choice)
                top_class, top_votes = max(self.class_votes.items(), key=lambda kv: kv[1])
                if top_votes >= 3:
                    self.class_name = top_class
                    self.class_locked = True
                    self.class_lock_frame = frame_idx
                else:
                    # Before locking, use voting system
                    self.class_name = top_class
    
    def get_latest_bbox(self):
        """Get the most recent bounding box"""
        if self.bboxes:
            return self.bboxes[-1]
        return None
    
    def get_avg_confidence(self):
        """Get average confidence score"""
        if self.confidences:
            return np.mean(list(self.confidences))
        return 0.0
    
    def avg_clip(self):
        """Get average CLIP embedding for semantic search"""
        if not self.clip_embs:
            return None
        avg = np.mean(self.clip_embs, axis=0)
        return avg / (np.linalg.norm(avg) + 1e-8)  # Normalize
    
    def get_dominant_color(self):
        """Get most frequent color (voting system)"""
        if not self.colors:
            return None
        color_counts = {}
        for color in self.colors:
            color_counts[color] = color_counts.get(color, 0) + 1
        return max(color_counts, key=color_counts.get)
    
    def predict_position(self, frames_ahead=1):
        """Predict future position based on velocity"""
        if not self.bboxes or not self.velocity:
            return self.get_latest_bbox()
        last_bbox = self.bboxes[-1]
        vx, vy = self.velocity
        # Predict center position
        cx = (last_bbox[0] + last_bbox[2]) / 2 + vx * frames_ahead
        cy = (last_bbox[1] + last_bbox[3]) / 2 + vy * frames_ahead
        # Keep same size
        w = last_bbox[2] - last_bbox[0]
        h = last_bbox[3] - last_bbox[1]
        return (int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2))

# ----------------------
# Main loop
# ----------------------
cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0
tracklets = {}  # tid -> Tracklet (for persons)
object_tracklets = {}  # oid -> ObjectTracklet (for objects)
next_tid = 1
next_obj_id = 1
OBJ_TRACK_MAX_AGE = 30  # Keep object tracks for 30 frames after last detection
OBJ_IOU_THRESHOLD = 0.3  # IoU threshold for matching object detections

# Debug: Show Qdrant client status
print(f"🔍 Qdrant client status: {'Connected' if client else 'Not connected (None)'}")

# small helper to map face boxes per frame
face_boxes_frame = []

while REALTIME_FACE_COMPARISON:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    h, w = frame.shape[:2]

    # On non-detection frames, update ByteTrack with empty detections to maintain tracking
    # This is critical for tracking continuity - ByteTrack needs to be updated every frame
    # Note: person and object detection now run on different schedules
    is_person_detection_frame = (frame_idx % DETECT_EVERY_N_FRAMES == 0)
    is_object_detection_frame = (frame_idx % DETECT_OBJECTS_EVERY_N_FRAMES == 0)
    
    if USE_BYTETRACK and byte_tracker is not None and not is_person_detection_frame:
        try:
            # Update ByteTrack with empty detections to maintain existing tracks
            # ByteTrack will predict positions for existing tracks even without new detections
            empty_detections = np.array([], dtype=np.float32).reshape(0, 5)
            img_info = (h, w)
            img_size = (w, h)
            tracked_objects = byte_tracker.update(empty_detections, img_info, img_size, frame=frame)
            
            # Update tracklets with ByteTrack predictions
            for track in tracked_objects:
                track_id = int(track.track_id)
                x1_bt, y1_bt, x2_bt, y2_bt = int(track.tlbr[0]), int(track.tlbr[1]), int(track.tlbr[2]), int(track.tlbr[3])
                
                # Validate bbox
                if x2_bt <= x1_bt or y2_bt <= y1_bt:
                    continue
                
                x1_bt, y1_bt, x2_bt, y2_bt = max(0, x1_bt), max(0, y1_bt), min(w, x2_bt), min(h, y2_bt)
                if x2_bt <= x1_bt or y2_bt <= y1_bt:
                    continue
                
                byte_track_bbox = (x1_bt, y1_bt, x2_bt, y2_bt)
                
                # Update tracklet if it exists
                if track_id in tracklets:
                    tracklets[track_id].bboxes.append(byte_track_bbox)
                    tracklets[track_id].last_frame = frame_idx
        except Exception as e:
            pass  # ByteTrack update failed, continue

    # Initialize detected_objects for visualization (empty if not detecting this frame)
    detected_objects = []
    
    # ============================================
    # PERSON DETECTION (every 13 frames)
    # ============================================
    if is_person_detection_frame:
        # person detection - use larger imgsz for close-ups and lower confidence
        # imgsz=1280 handles both distant and close-up persons better
        # ByteTrack's strength is using low-confidence detections for better association
        p_results = yolo_person.predict(frame, imgsz=1280, conf=0.25, classes=[0], verbose=False)
        person_boxes = []
        person_detections = []  # For ByteTrack: [x1, y1, x2, y2, score]
        
        if len(p_results):
            all_boxes = []
            all_scores = []
            all_detections = []
            
            # Debug: count how many detections before filtering
            raw_detection_count = len(p_results[0].boxes)
            filtered_counts = {"invalid": 0, "too_small": 0, "aspect_ratio": 0, "kept": 0}
            
            for b in p_results[0].boxes:
                x1,y1,x2,y2 = b.xyxy[0].cpu().numpy().astype(int)
                conf = float(b.conf[0].cpu().numpy())
                x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w,x2),min(h,y2)
                
                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    filtered_counts["invalid"] += 1
                    continue
                
                # Filter out small/partial detections (likely hands, arms, etc.)
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                frame_area = w * h
                
                # Skip very small boxes (likely body parts, not full persons)  
                # Minimum size: at least 1.5% of frame area, or minimum 80x120 pixels
                # Relaxed from 2% to handle more cases
                min_area = max(frame_area * 0.015, 80 * 120)
                if box_area < min_area:
                    filtered_counts["too_small"] += 1
                    continue
                
                # Skip boxes with extreme aspect ratios
                # Relaxed aspect ratio checks for close-ups (close-up faces/upper body can be wider)
                aspect_ratio = box_width / max(box_height, 1)
                if aspect_ratio > 1.2:  # Too wide (relaxed from 0.8 for close-ups)
                    filtered_counts["aspect_ratio"] += 1
                    continue
                
                # Skip boxes that are too tall and narrow (likely not a person)
                if aspect_ratio < 0.2:  # Too narrow
                    filtered_counts["aspect_ratio"] += 1
                    continue
                
                filtered_counts["kept"] += 1
                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(conf)
                all_detections.append((x1,y1,x2,y2))
            
            # Debug output for first few frames or when no detections kept
            if frame_idx <= DETECT_EVERY_N_FRAMES * 3 or filtered_counts["kept"] == 0:
                print(f"[Person Detection Debug] Frame {frame_idx}: "
                      f"Raw detections: {raw_detection_count}, "
                      f"Filtered: {filtered_counts}, "
                      f"Frame size: {w}x{h}")
            
            # Apply NMS to filter overlapping detections (same person detected multiple times)
            # Use moderate IoU threshold (0.45) - ByteTrack can handle some overlapping detections
            # Too strict NMS might remove valid detections that ByteTrack could use for association
            if len(all_boxes) > 0:
                nms_indices = apply_nms(all_boxes, all_scores, iou_threshold=0.45)
                
                person_boxes = []
                person_detections = []
                for idx in nms_indices:
                    person_boxes.append(all_detections[idx])
                    x1, y1, x2, y2 = all_detections[idx]
                    person_detections.append([x1, y1, x2, y2, all_scores[idx]])
            else:
                person_boxes = []
                person_detections = []
        
        # Update ByteTrack with detections (or empty if no detections)
        # ByteTrack MUST be updated every detection frame to maintain tracking continuity
        tracked_objects = []
        if USE_BYTETRACK and byte_tracker is not None:
            try:
                if len(person_detections) > 0:
                    detections_array = np.array(person_detections, dtype=np.float32)
                else:
                    # Update with empty detections to maintain existing tracks
                    detections_array = np.array([], dtype=np.float32).reshape(0, 5)
                
                # ByteTrack.update() requires: (output_results, img_info, img_size)
                img_info = (h, w)  # Height, width
                img_size = (w, h)  # Width, height
                tracked_objects = byte_tracker.update(detections_array, img_info, img_size, frame=frame)
                # Debug: print number of tracked objects
                if frame_idx <= DETECT_EVERY_N_FRAMES * 3:
                    print(f"[ByteTrack] Frame {frame_idx}: {len(person_detections)} detections -> {len(tracked_objects)} tracked objects")
            except Exception as e:
                print(f"ByteTrack update error: {e}")
                tracked_objects = []
        
        # face detection (full frame) - lower confidence to catch more faces
        f_results = yolo_face.predict(frame, imgsz=640, conf=0.25, verbose=False)
        face_boxes = []
        if len(f_results):
            for b in f_results[0].boxes:
                fx1,fy1,fx2,fy2 = b.xyxy[0].cpu().numpy().astype(int)
                fx1,fy1,fx2,fy2 = max(0,fx1),max(0,fy1),min(w,fx2),min(h,fy2)
                face_boxes.append((fx1,fy1,fx2,fy2))

        # keep face boxes for this frame (used when matching)
        face_boxes_frame = face_boxes
    
    # ============================================
    # OBJECT DETECTION (every 5 frames - faster for phones)
    # ============================================
    if is_object_detection_frame:
        # General object detection (laptops, phones, bags, etc.)
        detected_objects = []  # List of (class_name, bbox, confidence)
        # Lower confidence threshold for better detection of objects like backpacks
        obj_conf_threshold = 0.2  # Lowered to 0.2 for better detection of backpacks/bags
        
        if OBJECT_CLASSES is not None:
            # Detect specific classes only - use larger imgsz for better small object detection
            obj_results = yolo_objects.predict(frame, imgsz=1280, conf=obj_conf_threshold, classes=OBJECT_CLASSES, verbose=False)
        else:
            # Detect all COCO classes
            obj_results = yolo_objects.predict(frame, imgsz=1280, conf=obj_conf_threshold, verbose=False)
        
        if len(obj_results):
            # COCO class names
            class_names = yolo_objects.names
            current_detections = []  # List of (class_name, bbox, confidence)
            
            for b in obj_results[0].boxes:
                cls_id = int(b.cls[0].cpu().numpy())
                conf = float(b.conf[0].cpu().numpy())
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                
                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue
                
                class_name = class_names.get(cls_id, f"class_{cls_id}")
                current_detections.append((class_name, (x1, y1, x2, y2), conf))
            
            # Consolidate overlapping handbag/backpack detections - prefer backpack
            # YOLO sometimes detects the same bag as both handbag and backpack
            # Force consolidation: any bag that overlaps with another bag should be merged into one
            consolidated_detections = []
            used_indices = set()
            
            for i, (name_i, bbox_i, conf_i) in enumerate(current_detections):
                if i in used_indices:
                    continue
                
                # Check if this is a bag-type object
                is_bag_i = any(keyword in name_i.lower() for keyword in ['bag', 'backpack', 'handbag'])
                
                if is_bag_i:
                    # Find ALL overlapping bags and merge them into ONE
                    merged_bboxes = [bbox_i]
                    merged_confs = [conf_i]
                    merged_is_backpack = 'backpack' in name_i.lower()
                    
                    for j in range(i + 1, len(current_detections)):
                        if j in used_indices:
                            continue
                        
                        name_j, bbox_j, conf_j = current_detections[j]
                        is_bag_j = any(keyword in name_j.lower() for keyword in ['bag', 'backpack', 'handbag'])
                        
                        if not is_bag_j:
                            continue
                        
                        # Check if they overlap significantly (merge all nearby bags)
                        iou_val = iou(bbox_i, bbox_j)
                        if iou_val > 0.3:  # Any significant overlap - force merge
                            merged_bboxes.append(bbox_j)
                            merged_confs.append(conf_j)
                            is_backpack_j = 'backpack' in name_j.lower()
                            # Prefer backpack if ANY detection says backpack
                            if is_backpack_j:
                                merged_is_backpack = True
                            used_indices.add(j)
                    
                    # Use average bbox and highest confidence
                    avg_bbox = (
                        int(sum(b[0] for b in merged_bboxes) / len(merged_bboxes)),
                        int(sum(b[1] for b in merged_bboxes) / len(merged_bboxes)),
                        int(sum(b[2] for b in merged_bboxes) / len(merged_bboxes)),
                        int(sum(b[3] for b in merged_bboxes) / len(merged_bboxes))
                    )
                    max_conf = max(merged_confs)
                    final_class = 'backpack' if merged_is_backpack else name_i
                    
                    consolidated_detections.append((final_class, avg_bbox, max_conf))
                    used_indices.add(i)
                else:
                    # Not a bag - keep as is
                    consolidated_detections.append((name_i, bbox_i, conf_i))
                    used_indices.add(i)
            
            current_detections = consolidated_detections
            
            # Generate CLIP embeddings and detect colors for each detected object
            object_clip_embeddings = {}  # Maps (class_name, bbox) -> clip_embedding
            object_colors = {}  # Maps (class_name, bbox) -> color
            if USE_CLIP and clip_model is not None and len(current_detections) > 0:
                for class_name, bbox, conf in current_detections:
                    x1, y1, x2, y2 = bbox
                    obj_crop = frame[y1:y2, x1:x2]
                    if obj_crop.size > 0:
                        try:
                            obj_pil = Image.fromarray(cv2.cvtColor(obj_crop, cv2.COLOR_BGR2RGB))
                            obj_input = clip_preprocess(obj_pil).unsqueeze(0).to(DEVICE)
                            with torch.no_grad():
                                obj_emb = clip_model.encode_image(obj_input).cpu().numpy().flatten()
                                # Normalize
                                obj_emb = obj_emb / (np.linalg.norm(obj_emb) + 1e-8)
                                object_clip_embeddings[(class_name, bbox)] = obj_emb
                        except Exception as e:
                            if frame_idx <= DETECT_EVERY_N_FRAMES * 3:
                                print(f"[CLIP] Failed to encode object {class_name}: {e}")
                        
                        # Detect object color
                        try:
                            obj_color = get_dominant_color(obj_crop)
                            if obj_color:
                                object_colors[(class_name, bbox)] = obj_color
                        except Exception as e:
                            if frame_idx <= DETECT_EVERY_N_FRAMES * 3:
                                print(f"[Color] Failed to detect color for {class_name}: {e}")
            
            # Use ByteTrack for object tracking (handles sudden movements better)
            if USE_BYTETRACK and byte_tracker_objects is not None:
                try:
                    # Prepare detections for ByteTrack: [x1, y1, x2, y2, score]
                    obj_detections_array = []
                    obj_detection_info = []  # Store (class_name, bbox) for each detection
                    
                    for class_name, bbox, conf in current_detections:
                        x1, y1, x2, y2 = bbox
                        obj_detections_array.append([x1, y1, x2, y2, conf])
                        obj_detection_info.append((class_name, bbox))
                    
                    if len(obj_detections_array) > 0:
                        detections_array = np.array(obj_detections_array, dtype=np.float32)
                    else:
                        detections_array = np.array([], dtype=np.float32).reshape(0, 5)
                    
                    img_info = (h, w)
                    img_size = (w, h)
                    tracked_obj_tracks = byte_tracker_objects.update(detections_array, img_info, img_size, frame=frame)
                    
                    # Match ByteTrack tracks to detections and update object_tracklets
                    # Create a mapping from ByteTrack ID to class_name and bbox
                    for track in tracked_obj_tracks:
                        track_id = int(track.track_id)
                        x1_bt, y1_bt, x2_bt, y2_bt = int(track.tlbr[0]), int(track.tlbr[1]), int(track.tlbr[2]), int(track.tlbr[3])
                        
                        if x2_bt <= x1_bt or y2_bt <= y1_bt:
                            continue
                        
                        x1_bt, y1_bt, x2_bt, y2_bt = max(0, x1_bt), max(0, y1_bt), min(w, x2_bt), min(h, y2_bt)
                        if x2_bt <= x1_bt or y2_bt <= y1_bt:
                            continue
                        
                        byte_track_bbox = (x1_bt, y1_bt, x2_bt, y2_bt)
                        
                        # Find best matching detection (using IoU first, then distance)
                        # ByteTrack bboxes might be slightly different from detection bboxes
                        best_detection = None
                        best_iou_val = 0.05  # Very low threshold to catch any overlap
                        best_distance = float('inf')
                        
                        for class_name, bbox, conf in current_detections:
                            iou_val = iou(byte_track_bbox, bbox)
                            if iou_val > best_iou_val:
                                best_iou_val = iou_val
                                best_detection = (class_name, bbox, conf)
                            
                            # Also calculate distance for fallback
                            cx_bt = (x1_bt + x2_bt) / 2
                            cy_bt = (y1_bt + y2_bt) / 2
                            cx_det = (bbox[0] + bbox[2]) / 2
                            cy_det = (bbox[1] + bbox[3]) / 2
                            dist = ((cx_bt - cx_det)**2 + (cy_bt - cy_det)**2)**0.5
                            if dist < best_distance:
                                best_distance = dist
                                if not best_detection:  # If no IoU match, use closest by distance
                                    best_detection = (class_name, bbox, conf)
                        
                        # Use detection info if available, otherwise use ByteTrack bbox
                        if best_detection:
                            class_name, det_bbox, conf = best_detection
                            # Prefer backpack label over handbag if conflicting detections
                            if track_id in object_tracklets:
                                existing_class = object_tracklets[track_id].class_name.lower()
                                if 'backpack' in class_name.lower() and 'backpack' not in existing_class:
                                    pass  # allow upgrade to backpack
                                elif 'backpack' in existing_class and 'backpack' not in class_name.lower():
                                    class_name = object_tracklets[track_id].class_name  # keep backpack label
                            # Use detection bbox (more accurate) but ByteTrack ID
                            bbox_to_store = det_bbox
                        else:
                            # No matching detection - use ByteTrack bbox and try to get class from existing track
                            bbox_to_store = byte_track_bbox
                            if track_id in object_tracklets:
                                # Existing track - use its class name
                                class_name = object_tracklets[track_id].class_name
                                conf = object_tracklets[track_id].get_avg_confidence()
                            else:
                                # New track without detection match - ALWAYS match to closest detection
                                if len(current_detections) > 0:
                                    # Find closest by distance (no threshold - ByteTrack tracks come from detections)
                                    closest = None
                                    min_dist = float('inf')
                                    for class_name_det, bbox_det, conf_det in current_detections:
                                        cx_bt = (x1_bt + x2_bt) / 2
                                        cy_bt = (y1_bt + y2_bt) / 2
                                        cx_det = (bbox_det[0] + bbox_det[2]) / 2
                                        cy_det = (bbox_det[1] + bbox_det[3]) / 2
                                        dist = ((cx_bt - cx_det)**2 + (cy_bt - cy_det)**2)**0.5
                                        if dist < min_dist:
                                            min_dist = dist
                                            closest = (class_name_det, bbox_det, conf_det)
                                    
                                    if closest:
                                        class_name, bbox_to_store, conf = closest
                                        if frame_idx <= DETECT_EVERY_N_FRAMES * 3:
                                            print(f"[Object Tracking] Track {track_id} matched to {class_name} by distance ({min_dist:.1f}px)")
                                else:
                                    # No detections at all - skip
                                    continue
                        
                        # Update or create object tracklet with ByteTrack ID
                        # Get CLIP embedding and color for this detection if available
                        clip_emb = object_clip_embeddings.get((class_name, bbox_to_store), None)
                        obj_color = object_colors.get((class_name, bbox_to_store), None)
                        
                        if track_id in object_tracklets:
                            object_tracklets[track_id].update(bbox_to_store, conf, frame_idx, class_name=class_name, clip_emb=clip_emb, color=obj_color)
                        else:
                            obj_tracklet = ObjectTracklet(track_id, class_name, bbox_to_store, conf, frame_idx)
                            if clip_emb is not None:
                                obj_tracklet.clip_embs.append(clip_emb)
                            if obj_color is not None:
                                obj_tracklet.colors.append(obj_color)
                            object_tracklets[track_id] = obj_tracklet
                    
                    # Clean up old tracks
                    tracks_to_remove = []
                    for oid, obj_track in object_tracklets.items():
                        if frame_idx - obj_track.last_frame > OBJ_TRACK_MAX_AGE:
                            tracks_to_remove.append(oid)
                    for oid in tracks_to_remove:
                        # Insert to Qdrant before removing
                        obj_track = object_tracklets[oid]
                        if not obj_track.inserted and client:
                            obj_track.inserted = insert_object_track_to_qdrant(client, obj_track, video_id=VIDEO_ID, segment_id=None, frame_rate=30.0)
                        del object_tracklets[oid]
                    
                    if frame_idx <= 50:  # Only first 50 frames
                        obj_names = [obj[0] for obj in current_detections] if current_detections else []
                        matched_count = sum(1 for tid in object_tracklets.keys() if object_tracklets[tid].last_frame == frame_idx)
                        print(f"[Object Tracking] Frame {frame_idx}: Detected {len(current_detections)} objects: {obj_names}, ByteTrack tracks: {len(tracked_obj_tracks)}, Active object_tracklets: {len(object_tracklets)}, New/Updated this frame: {matched_count}")
                
                except Exception as e:
                    print(f"ByteTrack object tracking error: {e}")
                    # Fallback to IoU-based tracking
                    tracked_obj_tracks = []
            else:
                # Fallback: IoU-based tracking if ByteTrack not available
                matched_track_ids = set()
                for class_name, bbox, conf in current_detections:
                    best_match_id = None
                    best_iou = OBJ_IOU_THRESHOLD
                    
                    # Check if this is a bag-type object
                    is_bag = any(keyword in class_name.lower() for keyword in ['bag', 'backpack', 'handbag'])
                    
                    for oid, obj_track in object_tracklets.items():
                        # For bags, allow matching across bag types (backpack can match handbag)
                        if is_bag:
                            is_track_bag = any(keyword in obj_track.class_name.lower() for keyword in ['bag', 'backpack', 'handbag'])
                            if not is_track_bag:
                                continue  # Only match bags to bags
                        else:
                            # For non-bags, require exact class match
                            if obj_track.class_name != class_name:
                                continue
                        
                        if frame_idx - obj_track.last_frame > OBJ_TRACK_MAX_AGE:
                            continue
                        
                        last_bbox = obj_track.get_latest_bbox()
                        if last_bbox:
                            iou_val = iou(bbox, last_bbox)
                            if iou_val > best_iou:
                                best_iou = iou_val
                                best_match_id = oid
                    
                    if best_match_id is not None:
                        clip_emb = object_clip_embeddings.get((class_name, bbox), None)
                        obj_color = object_colors.get((class_name, bbox), None)
                        object_tracklets[best_match_id].update(bbox, conf, frame_idx, class_name=class_name, clip_emb=clip_emb, color=obj_color)
                    else:
                        obj_tracklet = ObjectTracklet(next_obj_id, class_name, bbox, conf, frame_idx)
                        clip_emb = object_clip_embeddings.get((class_name, bbox), None)
                        obj_color = object_colors.get((class_name, bbox), None)
                        if clip_emb is not None:
                            obj_tracklet.clip_embs.append(clip_emb)
                        if obj_color is not None:
                            obj_tracklet.colors.append(obj_color)
                        object_tracklets[next_obj_id] = obj_tracklet
                        next_obj_id += 1
        
    # Update object tracker on non-object-detection frames (maintain tracking continuity)
    if USE_BYTETRACK and byte_tracker_objects is not None and not is_object_detection_frame:
            try:
                empty_detections = np.array([], dtype=np.float32).reshape(0, 5)
                img_info = (h, w)
                img_size = (w, h)
                tracked_obj_tracks = byte_tracker_objects.update(empty_detections, img_info, img_size, frame=frame)
                
                # Update object_tracklets with ByteTrack predictions
                for track in tracked_obj_tracks:
                    track_id = int(track.track_id)
                    x1_bt, y1_bt, x2_bt, y2_bt = int(track.tlbr[0]), int(track.tlbr[1]), int(track.tlbr[2]), int(track.tlbr[3])
                    
                    if x2_bt <= x1_bt or y2_bt <= y1_bt:
                        continue
                    
                    x1_bt, y1_bt, x2_bt, y2_bt = max(0, x1_bt), max(0, y1_bt), min(w, x2_bt), min(h, y2_bt)
                    if x2_bt <= x1_bt or y2_bt <= y1_bt:
                        continue
                    
                    byte_track_bbox = (x1_bt, y1_bt, x2_bt, y2_bt)
                    
                    if track_id in object_tracklets:
                        # Update with ByteTrack predicted position
                        obj_track = object_tracklets[track_id]
                        # Use average confidence from track history
                        avg_conf = obj_track.get_avg_confidence()
                        object_tracklets[track_id].update(byte_track_bbox, avg_conf, frame_idx, class_name=obj_track.class_name)
                    # Note: Don't create new tracklets on non-detection frames - wait for next detection frame
                # After prediction step, keep phones and backpacks alive longer by not pruning aggressively
                tracks_to_remove = []
                for oid, obj_track in object_tracklets.items():
                    # Check if this is a phone or backpack (prone to ID churn)
                    is_phone = any(keyword in obj_track.class_name.lower() for keyword in ["phone", "cell phone", "mobile phone"])
                    is_bag = any(keyword in obj_track.class_name.lower() for keyword in ["backpack", "bag", "handbag", "suitcase"])
                    
                    if is_phone:
                        # Phones: double age (very small, fast-moving)
                        if frame_idx - obj_track.last_frame > OBJ_TRACK_MAX_AGE * 2:
                            tracks_to_remove.append(oid)
                    elif is_bag:
                        # Bags: triple age (detection can be intermittent due to class confusion)
                        if frame_idx - obj_track.last_frame > OBJ_TRACK_MAX_AGE * 3:
                            tracks_to_remove.append(oid)
                    else:
                        if frame_idx - obj_track.last_frame > OBJ_TRACK_MAX_AGE:
                            tracks_to_remove.append(oid)
                for oid in tracks_to_remove:
                    # Insert to Qdrant before removing
                    obj_track = object_tracklets[oid]
                    if not obj_track.inserted and client:
                        obj_track.inserted = insert_object_track_to_qdrant(client, obj_track, video_id=VIDEO_ID, segment_id=None, frame_rate=30.0)
                    del object_tracklets[oid]
            except Exception as e:
                pass  # ByteTrack update failed, continue
    
    # Update detected_objects list with tracked objects (for visualization)
    detected_objects = []
    for oid, obj_track in object_tracklets.items():
        if frame_idx - obj_track.last_frame <= OBJ_TRACK_MAX_AGE:
            bbox = obj_track.get_latest_bbox()
            if bbox:
                avg_conf = obj_track.get_avg_confidence()
                detected_objects.append((obj_track.class_name, bbox, avg_conf))
    
    # Debug: Also try detecting ALL classes to see if backpack is detected with different settings
    # This helps debug if the class ID is correct or if backpack needs even lower threshold
    if is_object_detection_frame and frame_idx <= DETECT_OBJECTS_EVERY_N_FRAMES * 10:
        all_obj_results = yolo_objects.predict(frame, imgsz=1280, conf=0.15, verbose=False)
        if len(all_obj_results):
            class_names = yolo_objects.names
            all_detected = {}
            for b in all_obj_results[0].boxes:
                cls_id = int(b.cls[0].cpu().numpy())
                conf = float(b.conf[0].cpu().numpy())
                class_name = class_names.get(cls_id, f"class_{cls_id}")
                if class_name not in all_detected or conf > all_detected[class_name]:
                    all_detected[class_name] = conf
            
            # Check if backpack was detected in all classes
            if 'backpack' in all_detected:
                print(f"[Debug] ✓ Backpack detected with confidence {all_detected['backpack']:.3f} (class ID: 24)")
            elif any('bag' in name.lower() or 'pack' in name.lower() for name in all_detected.keys()):
                bag_related = [(name, conf) for name, conf in all_detected.items() if 'bag' in name.lower() or 'pack' in name.lower()]
                print(f"[Debug] Bag-related objects detected: {bag_related}")
            # Show all detected objects for debugging (reduced frequency)
            if frame_idx <= 30:  # Only first 30 frames
                print(f"[Debug] All objects detected (conf >= 0.15): {list(all_detected.keys())}")

    # ============================================
    # PERSON TRACKING UPDATE (runs on person detection frames)
    # ============================================
    if is_person_detection_frame:
        # Process ByteTrack tracked objects - ByteTrack already handles ID assignment and matching
        # ByteTrack's internal logic handles duplicate/overlapping tracks, so we trust its output
        if USE_BYTETRACK and len(tracked_objects) > 0:
            # Process each ByteTrack track directly - ByteTrack handles association internally
            for track in tracked_objects:
                track_id = int(track.track_id)
                # Get ByteTrack bbox
                x1_bt, y1_bt, x2_bt, y2_bt = int(track.tlbr[0]), int(track.tlbr[1]), int(track.tlbr[2]), int(track.tlbr[3])
                
                # Validate ByteTrack bbox - skip if invalid
                if x2_bt <= x1_bt or y2_bt <= y1_bt:
                    continue  # Skip invalid ByteTrack bboxes
                
                x1_bt, y1_bt, x2_bt, y2_bt = max(0, x1_bt), max(0, y1_bt), min(w, x2_bt), min(h, y2_bt)
                
                # Double-check after clamping
                if x2_bt <= x1_bt or y2_bt <= y1_bt:
                    continue  # Skip if still invalid after clamping
                
                byte_track_bbox = (x1_bt, y1_bt, x2_bt, y2_bt)
                
                # Find the closest person box to this ByteTrack bbox (for face/ReID extraction)
                best_box = None
                best_iou_val = 0
                for box in person_boxes:
                    iou_val = iou(byte_track_bbox, box)
                    if iou_val > best_iou_val:
                        best_iou_val = iou_val
                        best_box = box
                
                # Use best matching person box for face/ReID, or ByteTrack bbox if no good match
                # Prefer person detection box (more accurate) over ByteTrack predicted box
                # Lower threshold to 0.2 to catch more matches (ByteTrack bboxes might be slightly off)
                if best_box and best_iou_val > 0.2:  # More lenient overlap threshold
                    box = best_box
                    # Use person detection box for visualization (more accurate)
                    vis_bbox = best_box
                else:
                    box = byte_track_bbox
                    # Use ByteTrack bbox if no person box matches - still visualize it!
                    vis_bbox = byte_track_bbox
                
                x1, y1, x2, y2 = box
                
                # Validate crop dimensions before extracting
                if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                    continue  # Skip invalid crops
                
                crop_person = frame[y1:y2, x1:x2].copy()
                
                # Check if crop is valid (not empty)
                if crop_person.size == 0 or crop_person.shape[0] == 0 or crop_person.shape[1] == 0:
                    continue  # Skip empty crops
                
                # Use ByteTrack ID directly - ByteTrack maintains ID consistency
                current_tid = track_id
                
                # Update or create tracklet with accurate person detection box (for visualization)
                # Store the person detection box if available, otherwise use ByteTrack bbox
                if current_tid not in tracklets:
                    tracklets[current_tid] = Tracklet(current_tid, vis_bbox, frame_idx)
                else:
                    # Update existing tracklet with accurate bbox (person detection preferred)
                    tracklets[current_tid].bboxes.append(vis_bbox)
                    tracklets[current_tid].last_frame = frame_idx

                # ---- Optimized face and ReID extraction
                face_emb, face_size = extract_face_embedding_optimized(crop_person, face_boxes_frame, vis_bbox)
                reid_emb = reid_encode(crop_person)
                
                # ---- Carried objects association (EXTREMELY STRICT spatial checks)
                # For an object to be marked as carried, ALL conditions must be true:
                # 1. Object MUST be COMPLETELY INSIDE person's bbox (no parts sticking out)
                # 2. Object must be in UPPER portion of person (hands/shoulders, not waist/hips)
                # 3. Object must be SMALL (handbag < 50% of person height)
                # 4. Object center must be CLOSE to person's horizontal center (±30% person width)
                # This ONLY marks items that are clearly being held/worn
                carried_obj_bboxes = []  # Store object bboxes for combined CLIP crop
                
                if detected_objects:
                    # Person body measurements
                    person_x1, person_y1, person_x2, person_y2 = vis_bbox
                    person_height = person_y2 - person_y1
                    person_width = person_x2 - person_x1
                    person_center_x = (person_x1 + person_x2) / 2.0
                    person_center_y = (person_y1 + person_y2) / 2.0
                    
                    # Upper body area: top to ~50% down (shoulders/hands where items are held)
                    upper_bound_y = person_y1 + person_height * 0.5
                    
                    for class_name, obj_bbox, obj_conf in detected_objects:
                        obj_x1, obj_y1, obj_x2, obj_y2 = obj_bbox
                        obj_center_x = (obj_x1 + obj_x2) / 2.0
                        obj_center_y = (obj_y1 + obj_y2) / 2.0
                        obj_width = obj_x2 - obj_x1
                        obj_height = obj_y2 - obj_y1
                        
                        # Check 1: Object MUST be COMPLETELY inside person bbox (all 4 corners inside)
                        completely_inside = (person_x1 <= obj_x1 and obj_x2 <= person_x2 and
                                           person_y1 <= obj_y1 and obj_y2 <= person_y2)
                        
                        # Check 2: Object must be in UPPER portion (not below 50% mark)
                        is_in_upper_portion = obj_center_y < upper_bound_y
                        
                        # Check 3: Object must be SMALL (max 50% of person height)
                        size_ratio = obj_height / person_height if person_height > 0 else 1.0
                        is_reasonably_small = size_ratio < 0.5
                        
                        # Check 4: Object center must be CLOSE to person's horizontal center
                        # Allow ±30% of person width from center
                        horizontal_center_dist = abs(obj_center_x - person_center_x)
                        horizontal_tolerance = person_width * 0.3
                        is_horizontally_centered = horizontal_center_dist <= horizontal_tolerance
                        
                        # Mark as carried ONLY if ALL conditions met
                        is_carried = (completely_inside and 
                                     is_in_upper_portion and 
                                     is_reasonably_small and 
                                     is_horizontally_centered)
                        
                        if is_carried:
                            carried_objs.append(class_name)
                            carried_obj_bboxes.append(obj_bbox)
                
                # ---- CLIP embedding (person + nearby objects for context-aware embedding)
                # If person is carrying objects, create combined crop for better semantic understanding
                if carried_obj_bboxes:
                    # Create expanded bbox that includes person + all carried objects
                    all_boxes = [vis_bbox] + carried_obj_bboxes
                    combined_x1 = min(box[0] for box in all_boxes)
                    combined_y1 = min(box[1] for box in all_boxes)
                    combined_x2 = max(box[2] for box in all_boxes)
                    combined_y2 = max(box[3] for box in all_boxes)
                    
                    # Clamp to frame boundaries
                    combined_x1 = max(0, combined_x1)
                    combined_y1 = max(0, combined_y1)
                    combined_x2 = min(w, combined_x2)
                    combined_y2 = min(h, combined_y2)
                    
                    # Extract combined crop (person + objects)
                    if combined_x2 > combined_x1 and combined_y2 > combined_y1:
                        combined_crop = frame[combined_y1:combined_y2, combined_x1:combined_x2].copy()
                        clip_emb = clip_encode(combined_crop)  # CLIP sees person WITH object
                    else:
                        clip_emb = clip_encode(crop_person)  # Fallback to person-only
                else:
                    # No objects detected - use person-only crop
                    clip_emb = clip_encode(crop_person)
                
                # Detect clothing colors (every frame for accuracy)
                person_h, person_w = crop_person.shape[:2]
                upper_part = mask_upper_by_face(crop_person, face_boxes_frame, vis_bbox)
                lower_part = crop_person[person_h//2:, :]
                upper_color = get_dominant_color(upper_part)
                lower_color = get_dominant_color(lower_part)
                
                # Detect person attributes (every frame for accuracy)
                detected_attributes = detect_person_attributes(crop_person, None)

                # Update tracklet with face and ReID embeddings
                # Use vis_bbox (person detection box) for accurate visualization
                t = tracklets[current_tid]
                t.update(vis_bbox, frame_idx, face_emb, reid_emb, face_size, clip_emb, carried_objs, upper_color, lower_color, detected_attributes)
                
                # Debug output (reduced frequency)
                if frame_idx <= 100:  # Only first 100 frames
                    face_status = "✓" if face_emb is not None else "✗"
                    yolo_faces = len(face_boxes_frame)
                    person_h, person_w = crop_person.shape[:2]
                    print(f"[Frame {frame_idx}] Tracklet {current_tid}: YOLO faces: {yolo_faces}, Face detected: {face_status} (size: {face_size}px), Person crop: {person_w}x{person_h}px, ReID: {'✓' if reid_emb is not None else '✗'}")
                    if face_emb is None and yolo_faces == 0:
                        print(f"  → No YOLO faces found, tried InsightFace on person crop")
        
        # Process person boxes that weren't matched to ByteTrack tracks (fallback for missed detections)
        # This ensures all detected people are visualized, even if ByteTrack didn't track them
        if USE_BYTETRACK and len(tracked_objects) > 0:
            # Find person boxes that weren't matched to any ByteTrack track
            matched_person_boxes = set()
            for track in tracked_objects:
                track_id = int(track.track_id)
                x1_bt, y1_bt, x2_bt, y2_bt = int(track.tlbr[0]), int(track.tlbr[1]), int(track.tlbr[2]), int(track.tlbr[3])
                if x2_bt <= x1_bt or y2_bt <= y1_bt:
                    continue
                byte_track_bbox = (x1_bt, y1_bt, x2_bt, y2_bt)
                # Find matching person box
                for box in person_boxes:
                    if iou(byte_track_bbox, box) > 0.2:
                        matched_person_boxes.add(box)
            
            # Process unmatched person boxes with IOU-based matching
            for box in person_boxes:
                if box in matched_person_boxes:
                    continue  # Already processed by ByteTrack
                
                # Use IOU-based matching for unmatched person boxes
                x1,y1,x2,y2 = box
                crop_person = frame[y1:y2, x1:x2].copy()
                
                if crop_person.size == 0 or crop_person.shape[0] == 0 or crop_person.shape[1] == 0:
                    continue
                
                best_tid, best_iouv = None, 0
                for tid, t in tracklets.items():
                    if len(t.bboxes) == 0:
                        continue
                    val = iou(box, t.bboxes[-1])
                    if val > best_iouv:
                        best_tid, best_iouv = tid, val
                
                if best_iouv > IOU_THRESHOLD:
                    current_tid = best_tid
                    tracklets[current_tid].bboxes.append(box)
                    tracklets[current_tid].last_frame = frame_idx
                else:
                    # Create new tracklet for unmatched person
                    current_tid = next_tid
                    next_tid += 1
                    tracklets[current_tid] = Tracklet(current_tid, box, frame_idx)
                
                # Extract face and ReID for this unmatched person box
                face_emb, face_size = extract_face_embedding_optimized(crop_person, face_boxes_frame, box)
                reid_emb = reid_encode(crop_person)
                
                # ---- Carried objects + context-aware CLIP (EXTREMELY STRICT)
                carried_objs = []
                carried_obj_bboxes = []
                if detected_objects:
                    # Person body measurements for strict checks
                    px1, py1, px2, py2 = box
                    p_h = max(1, py2 - py1)
                    p_w = max(1, px2 - px1)
                    p_cx = (px1 + px2) / 2.0
                    upper_bound_y = py1 + p_h * 0.5

                    for class_name, obj_bbox, obj_conf in detected_objects:
                        ox1, oy1, ox2, oy2 = obj_bbox
                        ocx = (ox1 + ox2) / 2.0
                        ocy = (oy1 + oy2) / 2.0
                        o_h = max(1, oy2 - oy1)

                        # 1) Completely inside person bbox
                        completely_inside = (px1 <= ox1 and ox2 <= px2 and py1 <= oy1 and oy2 <= py2)
                        # 2) In upper portion of person
                        in_upper = ocy < upper_bound_y
                        # 3) Reasonably small (< 50% of person height)
                        small_enough = (o_h / p_h) < 0.5
                        # 4) Horizontally centered (±30% of person width)
                        horiz_centered = abs(ocx - p_cx) <= (p_w * 0.3)

                        if completely_inside and in_upper and small_enough and horiz_centered:
                            carried_objs.append(class_name)
                            carried_obj_bboxes.append(obj_bbox)
                
                # CLIP with person + objects if available
                if carried_obj_bboxes:
                    all_boxes = [box] + carried_obj_bboxes
                    combined_x1 = max(0, min(b[0] for b in all_boxes))
                    combined_y1 = max(0, min(b[1] for b in all_boxes))
                    combined_x2 = min(w, max(b[2] for b in all_boxes))
                    combined_y2 = min(h, max(b[3] for b in all_boxes))
                    if combined_x2 > combined_x1 and combined_y2 > combined_y1:
                        combined_crop = frame[combined_y1:combined_y2, combined_x1:combined_x2].copy()
                        clip_emb = clip_encode(combined_crop)
                    else:
                        clip_emb = clip_encode(crop_person)
                else:
                    clip_emb = clip_encode(crop_person)

                # Detect clothing colors (reduced frequency)
                if frame_idx % 10 == 0:
                    person_h, person_w = crop_person.shape[:2]
                    upper_part = mask_upper_by_face(crop_person, face_boxes_frame, box)
                    lower_part = crop_person[person_h//2:, :]
                    upper_color = get_dominant_color(upper_part)
                    lower_color = get_dominant_color(lower_part)
                else:
                    upper_color = None
                    lower_color = None

                # Detect person attributes (reduced frequency)
                if frame_idx % 15 == 0:
                    detected_attributes = detect_person_attributes(crop_person, None)
                else:
                    detected_attributes = None

                t = tracklets[current_tid]
                t.update(box, frame_idx, face_emb, reid_emb, face_size, clip_emb, carried_objs, upper_color, lower_color, detected_attributes)
        
        # Fallback: If ByteTrack is not available or no tracked objects, use IOU-based matching
        if not USE_BYTETRACK or len(tracked_objects) == 0:
            # Process each person box with IOU-based matching
            for box in person_boxes:
                x1,y1,x2,y2 = box
                crop_person = frame[y1:y2, x1:x2].copy()
                
                # IOU-based matching (original logic)
                best_tid, best_iouv = None, 0
                for tid, t in tracklets.items():
                    val = iou(box, t.bboxes[-1])
                    if val > best_iouv:
                        best_tid, best_iouv = tid, val
                
                if best_iouv > IOU_THRESHOLD:
                    current_tid = best_tid
                else:
                    current_tid = next_tid
                    next_tid += 1
                    tracklets[current_tid] = Tracklet(current_tid, box, frame_idx)
                
                # Optimized face and ReID extraction
                face_emb, face_size = extract_face_embedding_optimized(crop_person, face_boxes_frame, box)
                reid_emb = reid_encode(crop_person)
                
                # ---- Carried objects + context-aware CLIP (EXTREMELY STRICT)
                carried_objs = []
                carried_obj_bboxes = []
                if detected_objects:
                    # Person body measurements for strict checks
                    px1, py1, px2, py2 = box
                    p_h = max(1, py2 - py1)
                    p_w = max(1, px2 - px1)
                    p_cx = (px1 + px2) / 2.0
                    upper_bound_y = py1 + p_h * 0.5

                    for class_name, obj_bbox, obj_conf in detected_objects:
                        ox1, oy1, ox2, oy2 = obj_bbox
                        ocx = (ox1 + ox2) / 2.0
                        ocy = (oy1 + oy2) / 2.0
                        o_h = max(1, oy2 - oy1)

                        # 1) Completely inside person bbox
                        completely_inside = (px1 <= ox1 and ox2 <= px2 and py1 <= oy1 and oy2 <= py2)
                        # 2) In upper portion of person
                        in_upper = ocy < upper_bound_y
                        # 3) Reasonably small (< 50% of person height)
                        small_enough = (o_h / p_h) < 0.5
                        # 4) Horizontally centered (±30% of person width)
                        horiz_centered = abs(ocx - p_cx) <= (p_w * 0.3)

                        if completely_inside and in_upper and small_enough and horiz_centered:
                            carried_objs.append(class_name)
                            carried_obj_bboxes.append(obj_bbox)
                
                # CLIP with person + objects if available
                if carried_obj_bboxes:
                    all_boxes = [box] + carried_obj_bboxes
                    combined_x1 = max(0, min(b[0] for b in all_boxes))
                    combined_y1 = max(0, min(b[1] for b in all_boxes))
                    combined_x2 = min(w, max(b[2] for b in all_boxes))
                    combined_y2 = min(h, max(b[3] for b in all_boxes))
                    if combined_x2 > combined_x1 and combined_y2 > combined_y1:
                        combined_crop = frame[combined_y1:combined_y2, combined_x1:combined_x2].copy()
                        clip_emb = clip_encode(combined_crop)
                    else:
                        clip_emb = clip_encode(crop_person)
                else:
                    clip_emb = clip_encode(crop_person)
                
                # Detect clothing colors (reduced frequency)
                if frame_idx % 10 == 0:
                    person_h, person_w = crop_person.shape[:2]
                    upper_part = mask_upper_by_face(crop_person, face_boxes_frame, box)
                    lower_part = crop_person[person_h//2:, :]
                    upper_color = get_dominant_color(upper_part)
                    lower_color = get_dominant_color(lower_part)
                else:
                    upper_color = None
                    lower_color = None
                
                # Detect person attributes (reduced frequency)
                if frame_idx % 15 == 0:
                    detected_attributes = detect_person_attributes(crop_person, None)
                else:
                    detected_attributes = None
                
                # Update tracklet
                t = tracklets[current_tid]
                t.update(box, frame_idx, face_emb, reid_emb, face_size, clip_emb, carried_objs, upper_color, lower_color, detected_attributes)

    # --------------------------
    # Verification logic and tracklet insertion
    # --------------------------
    for tid, t in list(tracklets.items()):

        if frame_idx - t.last_frame > TRACKLET_MAX_AGE:
            # Track has ended (no updates for TRACKLET_MAX_AGE frames)
            # Insert ALL tracklets (verified and unverified) when they end
            if not t.inserted and client:
                t.inserted = insert_tracklet_to_qdrant(client, t, video_id=VIDEO_ID, segment_id=None, frame_rate=30.0)
            del tracklets[tid]
            continue

        if not t.verified and (len(t.face_embs) + len(t.reid_embs)) >= AGGREGATION_FRAMES:

            face_avg = t.avg_face()
            reid_avg = t.avg_reid()

            # Use actual detected face size instead of estimate
            face_width = t.avg_face_size()

            # VERIFICATION: Only run in REAL-TIME mode
            if REALTIME_FACE_COMPARISON:
                # REAL-TIME MODE: Compare against reference face
                face_ok, face_score = False, None
                reid_ok, reid_score = False, None

                # Always try face recognition if face embeddings available
                if face_avg is not None and len(t.face_embs) > 0:
                    face_ok, face_score = is_face_match(face_avg, face_width)

                # Always try ReID for all tracks (enabled for all person tracks)
                if reid_avg is not None and len(t.reid_embs) > 0:
                    reid_ok, reid_score = is_reid_match(reid_avg)

                # Verification logic: Track is verified ONLY if face matches reference
                # ReID is collected for all tracks but doesn't affect verification status
                if face_ok:
                    t.verified = True
                    print(f"✓ Tracklet {tid} verified via face recognition")
                else:
                    # Track remains unverified but ReID data is still collected and stored
                    pass
            else:
                # POST-PROCESSING MODE: Skip verification, all tracks remain unverified
                # All tracks will be stored in Qdrant with verified=False
                # Use compare_face_after_processing() later to find matches
                pass

            # ByteTrack handles tracking automatically, no need for manual tracker initialization

    # --------------------------
    # Visualization
    # --------------------------
    vis = frame.copy()
    # Merge overlapping same-class tracks and stabilize object IDs
    merge_object_tracklets(object_tracklets, iou_thresh=0.6)
    # Run reassignment multiple times for better convergence
    for _ in range(5):
        # Phones: keep very aggressive to maintain perfect tracking
        reassign_small_object_ids(object_tracklets,
                                  class_keywords=["phone", "cell phone", "mobile phone"],
                                  iou_thresh=0.35, max_center_dist=120)
    # Bags: use fewer, stricter passes to avoid merging two nearby bags into one ID
    for _ in range(2):
        reassign_small_object_ids(object_tracklets,
                                  class_keywords=["backpack", "bag", "handbag", "suitcase"],
                                  iou_thresh=0.55, max_center_dist=110,
                                  require_same_class=True)
    
    # Draw tracked persons
    for tid, t in tracklets.items():
        if len(t.bboxes) == 0:
            continue  # Skip tracklets with no bboxes
        
        x1,y1,x2,y2 = t.bboxes[-1]
        
        # Validate bbox coordinates
        if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            continue  # Skip invalid bboxes
        
        # Ensure bbox is within frame bounds
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        color = (0,255,0) if t.verified else (0,0,255)
        cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
        cv2.putText(vis, f"Person ID:{tid}{' V' if t.verified else ''}",
                    (x1, max(15, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw tracked objects (laptops, phones, bags, etc.)
    for oid, obj_track in object_tracklets.items():
        if frame_idx - obj_track.last_frame > OBJ_TRACK_MAX_AGE:
            continue  # Skip old tracks
        
        bbox = obj_track.get_latest_bbox()
        if bbox is None:
            continue
        
        x1, y1, x2, y2 = bbox
        # Validate bbox
        if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            continue
        
        # Use different color for objects (cyan)
        obj_color = (255, 255, 0)  # Cyan
        cv2.rectangle(vis, (x1, y1), (x2, y2), obj_color, 2)
        avg_conf = obj_track.get_avg_confidence()
        label = f"{obj_track.class_name} ID:{oid} {avg_conf:.2f}"
        cv2.putText(vis, label, (x1, max(15, y1-8)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj_color, 2)

    # Display with scaling if needed
    if DISPLAY_SCALE != 1.0:
        vis_display = cv2.resize(vis, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_LINEAR)
    else:
        vis_display = vis
    
    cv2.imshow("Hybrid Face+ReID CPU Pipeline (YOLO-face integrated)", vis_display)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# After loop ends, insert any remaining tracklets (verified and unverified) that were not inserted yet
if REALTIME_FACE_COMPARISON:
    for tid, t in list(tracklets.items()):
        if not t.inserted and client:
            t.inserted = insert_tracklet_to_qdrant(client, t, video_id=1, segment_id=None, frame_rate=30.0)

    # Also insert any remaining object tracklets
    for oid, obj_track in list(object_tracklets.items()):
        if not obj_track.inserted and client:
            obj_track.inserted = insert_object_track_to_qdrant(client, obj_track, video_id=1, segment_id=None, frame_rate=30.0)

    cap.release()
    cv2.destroyAllWindows()

    # ========================
    # POST-PROCESSING: Interactive Query Feature
    # ========================
    print("\n" + "="*80)
    print("🎬 VIDEO PROCESSING COMPLETE!")
    print("="*80)
    print(f"Total tracklets processed: {len(tracklets)}")
    print(f"Verified tracklets: {sum(1 for t in tracklets.values() if t.verified)}")
    print("="*80 + "\n")

def query_objects_by_text(text_prompt, top_k=5):
    """
    Query Qdrant for OBJECTS matching a text description using CLIP embeddings.
    Searches the object_tracks collection for laptops, phones, bags, etc. using semantic search.
    """
    if not client:
        print("❌ Qdrant client not connected")
        return []

    try:
        print(f"\n🔍 Searching for objects: '{text_prompt}'")
        query_lower = text_prompt.lower()
        
        # Check if we have CLIP embeddings in object_tracks
        # If all embeddings are zeros, fall back to keyword matching
        test_point, _ = client.scroll(
            collection_name="object_tracks",
            limit=1,
            with_vectors=True,
        )
        
        use_clip_search = False
        if test_point:
            vec = test_point[0].vector.get("object_vec", []) if hasattr(test_point[0], "vector") else []
            # Check if embedding is non-zero
            if isinstance(vec, list) and len(vec) > 0 and sum(abs(x) for x in vec) > 0.01:
                use_clip_search = True
        
        if use_clip_search and USE_CLIP and clip_model is not None:
            # Semantic search using CLIP embeddings
            print(f"   🔮 Using semantic CLIP search")
            
            # Encode text query
            text_token = clip.tokenize([text_prompt]).to(DEVICE)
            with torch.no_grad():
                text_emb = clip_model.encode_text(text_token).cpu().numpy().flatten()
                text_emb = text_emb / (np.linalg.norm(text_emb) + 1e-8)  # Normalize
            
            # CLIP ViT-B/32 outputs 512D - no padding needed
            if len(text_emb) < 512:
                text_vec = np.concatenate([text_emb, np.zeros(512 - len(text_emb), dtype=np.float32)])
            else:
                text_vec = text_emb[:512]
            
            # Scroll all objects and compute similarities manually
            points, _ = client.scroll(
                collection_name="object_tracks",
                limit=1000,
                with_payload=True,
                with_vectors=True,
            )
            
            if not points:
                print("   ⚠ No matching objects found")
                return []
            
            # Calculate similarities with class-prior boosting
            # Define query intent flags (object types)
            q_is_phone = any(k in query_lower for k in ["phone", "mobile", "cell", "smartphone", "iphone"])
            q_is_laptop = any(k in query_lower for k in ["laptop", "computer", "notebook", "macbook", "pc"])
            q_is_bag = any(k in query_lower for k in ["bag", "backpack", "handbag", "rucksack", "pack"]) and not q_is_laptop and not q_is_phone
            q_is_suitcase = any(k in query_lower for k in ["suitcase", "luggage", "trolley", "carry-on"]) and not q_is_laptop and not q_is_phone
            
            # Define color query flags
            q_color = None
            color_keywords = ["black", "white", "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "gray", "grey", "silver", "gold"]
            for color_kw in color_keywords:
                if color_kw in query_lower:
                    q_color = color_kw if color_kw != "grey" else "gray"  # Normalize grey->gray
                    break

            similarities = []
            for p in points:
                if not hasattr(p, "vector") or p.vector is None:
                    continue
                
                vec = None
                if isinstance(p.vector, dict):
                    vec = p.vector.get("object_vec") or p.vector.get("multi_vec")
                else:
                    vec = p.vector
                
                if vec is None:
                    continue
                
                vec_np = np.array(vec, dtype=np.float32)
                if vec_np.size == 0 or vec_np.shape[0] != text_vec.shape[0]:
                    continue
                
                # Check if vector is non-zero (has real embeddings)
                if np.sum(np.abs(vec_np)) < 0.01:
                    continue  # Skip zero embeddings
                
                # Compute cosine similarity
                sim = np.dot(text_vec, vec_np) / (np.linalg.norm(text_vec) * np.linalg.norm(vec_np) + 1e-8)

                # Class-prior boosting based on query keywords and object type
                payload = p.payload if hasattr(p, "payload") else {}
                obj_type_l = str(payload.get("object_type", "")).lower()
                obj_color = str(payload.get("object_color", "")).lower() if payload.get("object_color") else None
                
                is_phone = ("phone" in obj_type_l) or ("cell" in obj_type_l)
                is_laptop = ("laptop" in obj_type_l) or ("computer" in obj_type_l)
                is_bag = any(k in obj_type_l for k in ["bag", "backpack", "handbag"]) and not is_laptop and not is_phone
                is_suitcase = "suitcase" in obj_type_l or "luggage" in obj_type_l

                boost = 0.0
                
                # Class type boosting
                if q_is_phone:
                    if is_phone:
                        boost += 0.08
                    elif is_bag or is_suitcase:
                        boost -= 0.03
                if q_is_laptop:
                    if is_laptop:
                        boost += 0.08
                    elif is_bag or is_suitcase or is_phone:
                        boost -= 0.02
                if q_is_bag:
                    if is_bag:
                        boost += 0.08
                    elif is_suitcase:
                        boost += 0.03
                    elif is_phone or is_laptop:
                        boost -= 0.02
                if q_is_suitcase:
                    if is_suitcase:
                        boost += 0.08
                    elif is_bag:
                        boost += 0.03
                    elif is_phone or is_laptop:
                        boost -= 0.02
                
                # Color boosting (strong signal if color matches)
                if q_color and obj_color:
                    if q_color == obj_color:
                        boost += 0.12  # Strong boost for color match
                    else:
                        boost -= 0.05  # Penalty for wrong color

                similarities.append((sim + boost, p))
            
            if not similarities:
                print("   ⚠ No matching objects found")
                return []
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[0], reverse=True)
            top_results = similarities[:top_k]
            
            print(f"\n{'='*80}")
            print(f"📦 TOP {len(top_results)} OBJECT RESULTS FOR: '{text_prompt}' (CLIP Semantic)")
            print(f"{'='*80}")
            
            results = []
            for idx, (score, p) in enumerate(top_results, 1):
                payload = p.payload if hasattr(p, "payload") else {}
                
                track_id = payload.get("track_id", "Unknown")
                object_type = payload.get("object_type", "Unknown")
                object_color = payload.get("object_color", None)
                video_id = payload.get("video_id", "Unknown")
                start_time = payload.get("start_time", "Unknown")
                end_time = payload.get("end_time", "Unknown")
                num_frames = payload.get("num_frames", 0)
                avg_confidence = payload.get("avg_confidence", 0.0)
                
                # Display with color if available
                color_str = f" ({object_color})" if object_color else ""
                print(f"\n{idx}. Track ID: {track_id} | Object: {object_type}{color_str} | Similarity: {score:.3f}")
                print(f"   Video ID: {video_id} | Time: {start_time}s - {end_time}s | Frames: {num_frames}")
                print(f"   Confidence: {avg_confidence:.2f}")
                
                # Return format consistent with interactive loop: (track_id, score, payload)
                results.append((track_id, score, payload))
            
            return results
        
        else:
            # Fallback: Keyword matching (no CLIP embeddings available)
            print(f"   📝 Using keyword matching (no CLIP embeddings)")
            
            # Scroll all object tracks
            points, _ = client.scroll(
                collection_name="object_tracks",
                limit=1000,
                with_payload=True,
                with_vectors=False,
            )

            if not points:
                print("   ⚠ No objects found in collection")
                return []

            matches = []
            query_lower = text_prompt.lower()
            
            for p in points:
                payload = p.payload if hasattr(p, "payload") else {}
                object_type = payload.get("object_type", "").lower()
                
                # Simple keyword matching for objects
                score = 0.0
                
                # Direct object type matching
                if any(keyword in query_lower for keyword in ["laptop", "computer"]):
                    if "laptop" in object_type:
                        score = 1.0
                elif any(keyword in query_lower for keyword in ["phone", "mobile", "cell"]):
                    if "phone" in object_type or "cell" in object_type:
                        score = 1.0
                elif any(keyword in query_lower for keyword in ["backpack", "bag"]):
                    if "backpack" in object_type or "bag" in object_type:
                        score = 1.0
                elif any(keyword in query_lower for keyword in ["suitcase", "luggage"]):
                    if "suitcase" in object_type:
                        score = 1.0
                # Fallback: partial match
                else:
                    for keyword in query_lower.split():
                        if len(keyword) > 3 and keyword in object_type:
                            score = 0.8
                            break
                
                if score > 0:
                    matches.append((score, p))
            
            if not matches:
                print("   ⚠ No matching objects found")
                return []
            
            matches.sort(key=lambda x: x[0], reverse=True)
            top_results = matches[:top_k]

            print(f"\n{'='*80}")
            print(f"📦 TOP {len(top_results)} OBJECT RESULTS FOR: '{text_prompt}' (Keyword)")
            print(f"{'='*80}")

            results = []
            for idx, (score, p) in enumerate(top_results, 1):
                payload = p.payload if hasattr(p, "payload") else {}
                
                track_id = payload.get("track_id", "Unknown")
                object_type = payload.get("object_type", "Unknown")
                video_id = payload.get("video_id", "Unknown")
                start_time = payload.get("start_time", "Unknown")
                end_time = payload.get("end_time", "Unknown")
                num_frames = payload.get("num_frames", 0)
                avg_confidence = payload.get("avg_confidence", 0.0)
                
                results.append((track_id, score, payload))
                
                print(f"\n📦 Result #{idx}")
                print(f"   Object Type: {object_type}")
                print(f"   Track ID: {track_id}")
                print(f"   Match Score: {score:.2%}")
                print(f"   Video ID: {video_id}")
                print(f"   Time Range: {start_time} → {end_time}")
                print(f"   Duration: {num_frames} frames")
                print(f"   Avg Confidence: {avg_confidence:.3f}")
            
            print(f"\n{'='*80}\n")
            return results
        
    except Exception as e:
        print(f"❌ Error during object query: {e}")
        import traceback
        traceback.print_exc()
        return []

def query_tracklets_by_text(text_prompt, top_k=5):
    """
    Query Qdrant for tracklets matching a text description using CLIP embeddings.
    This uses scroll with vectors to stay compatible with older client versions.
    """
    if not USE_CLIP or clip_model is None:
        print("❌ CLIP not available - cannot perform text-based search")
        return []

    if not client:
        print("❌ Qdrant client not connected")
        return []

    try:
        print(f"\n🔍 Searching for: '{text_prompt}'")
        print("   Encoding text prompt with CLIP...")

        with torch.no_grad():
            text_tokens = clip.tokenize(text_prompt).to(DEVICE)
            text_features = clip_model.encode_text(text_tokens)
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
            text_embedding = text_features.squeeze().cpu().numpy().astype(np.float32)

        # CLIP ViT-B/32 outputs 512D (same as multi_vec in Qdrant)
        if len(text_embedding) < 512:
            text_embedding = np.concatenate([text_embedding, np.zeros(512 - len(text_embedding), dtype=np.float32)])
        else:
            text_embedding = text_embedding[:512]

        print(f"   Text embedding generated: {len(text_embedding)}D")

        # Scroll all points with vectors and payloads
        points, _ = client.scroll(
            collection_name="person_tracks",
            limit=1000,
            with_payload=True,
            with_vectors=True,
        )

        if not points:
            print("   ⚠ No points found in collection")
            return []

        similarities = []
        for p in points:
            if not hasattr(p, "vector") or p.vector is None:
                continue

            vec = None
            if isinstance(p.vector, dict):
                vec = p.vector.get("multi_vec") or p.vector.get("face_vec") or p.vector.get("reid_vec")
            else:
                vec = p.vector

            if vec is None:
                continue

            vec_np = np.array(vec, dtype=np.float32)
            if vec_np.size == 0:
                continue

            # Only compare if dimensions match
            if vec_np.shape[0] != text_embedding.shape[0]:
                continue

            sim = np.dot(text_embedding, vec_np) / (np.linalg.norm(text_embedding) * np.linalg.norm(vec_np) + 1e-8)
            
            # Apply smart boosting based on query keywords and metadata
            boosted_score = sim
            payload = p.payload if hasattr(p, "payload") else {}
            carried_objs = payload.get("object_carried", [])
            verified = payload.get("verified", False)
            
            # Keyword-based boosting for object queries
            query_lower = text_prompt.lower()
            if any(keyword in query_lower for keyword in ["phone", "mobile", "cell"]):
                if any("phone" in obj.lower() for obj in carried_objs):
                    boosted_score += 0.15  # Strong boost for matching object
            elif any(keyword in query_lower for keyword in ["laptop", "computer"]):
                if any("laptop" in obj.lower() for obj in carried_objs):
                    boosted_score += 0.15
            elif any(keyword in query_lower for keyword in ["bag", "backpack", "handbag"]):
                if any(keyword in obj.lower() for obj in carried_objs for keyword in ["bag", "backpack", "handbag"]):
                    boosted_score += 0.15
            elif any(keyword in query_lower for keyword in ["holding", "carrying", "with"]):
                # Generic "holding/carrying" query - small boost if ANY object present
                if carried_objs:
                    boosted_score += 0.05
            
            # Clothing color boosting
            color_keywords = ["red", "blue", "green", "yellow", "black", "white", "gray", "grey", "brown", "pink", "purple", "orange", "cyan", "magenta", "navy", "beige", "tan"]
            upper_color = payload.get("upper_color")
            lower_color = payload.get("lower_color")
            
            for color in color_keywords:
                if color in query_lower:
                    qcolor = "gray" if color == "grey" else color
                    matched = (upper_color and (upper_color.lower() == qcolor)) or (lower_color and (lower_color.lower() == qcolor))
                    if matched:
                        boosted_score += 0.14
                        print(f"   → Color boost applied (+0.14): query has '{color}', payload upper={upper_color}, lower={lower_color}")
                    else:
                        boosted_score -= 0.03
                        print(f"   → Color penalty applied (-0.03): query has '{color}' but payload colors upper={upper_color}, lower={lower_color}")
                    break
            
            # Check for clothing keywords (shirt, pants, jacket, etc.)
            clothing_keywords = ["shirt", "pants", "jacket", "dress", "skirt", "sweater", "coat", "jeans", "hat", "hood"]
            if any(keyword in query_lower for keyword in clothing_keywords):
                # If query mentions clothing and we have color data, moderate boost
                if upper_color or lower_color:
                    boosted_score += 0.05
            
            # === ATTRIBUTE-BASED BOOSTING (more decisive, with mild penalty when absent) ===
            attributes = payload.get("attributes", {})

            def keyword_hit(words):
                return any(w in query_lower for w in words)

            # Hat / Cap / Beanie
            if keyword_hit(["hat", "cap", "beanie", "wearing a hat", "wearing a cap", "with hat", "with cap", "has hat", "has cap"]):
                if attributes.get("has_hat"):
                    boosted_score += 0.25  # stronger boost when hat is detected
                else:
                    boosted_score -= 0.04  # slight penalty if query asks for hat but none detected

            # Hood
            if keyword_hit(["hood", "hooded", "wearing hood", "has hood", "with hood"]):
                if attributes.get("has_hood"):
                    boosted_score += 0.22
                    print("   → Hood boost applied (+0.22): has_hood=True")
                else:
                    # Fallback heuristic: gray/white upper without hat often indicates hood
                    uc = (upper_color or "").lower()
                    if (uc in ["gray", "grey", "white"]) and not attributes.get("has_hat"):
                        boosted_score += 0.08
                        print(f"   → Hood heuristic boost (+0.08): upper={upper_color}, has_hat={attributes.get('has_hat')}")
                    else:
                        boosted_score -= 0.03

            # Glasses / Sunglasses
            if keyword_hit(["glasses", "sunglasses", "wearing glasses", "has glasses", "with glasses"]):
                if attributes.get("has_glasses"):
                    boosted_score += 0.18
                else:
                    boosted_score -= 0.02
            
            # Small boost for verified tracklets (reference person match)
            if "verified" in query_lower and verified:
                boosted_score += 0.03
            
            similarities.append((boosted_score, p))

        if not similarities:
            print("   ⚠ No valid vectors to compare")
            return []

        similarities.sort(key=lambda x: x[0], reverse=True)
        top_results = similarities[:top_k]

        print(f"\n{'='*80}")
        print(f"📊 TOP {len(top_results)} RESULTS FOR: '{text_prompt}'")
        print(f"{'='*80}")

        matches = []
        for idx, (sim, p) in enumerate(top_results, 1):
            similarity_norm = (sim + 1) / 2  # map cosine [-1,1] to [0,1]
            payload = p.payload if hasattr(p, "payload") else {}

            track_id = payload.get("track_id", "Unknown")
            video_id = payload.get("video_id", "Unknown")
            start_time = payload.get("start_time", "Unknown")
            end_time = payload.get("end_time", "Unknown")
            num_frames = payload.get("num_frames", 0)
            verified = payload.get("verified", False)
            carried_objs = payload.get("object_carried", [])

            matches.append((track_id, similarity_norm, payload))

            print(f"\n🎯 Result #{idx}")
            print(f"   Track ID: {track_id}")
            print(f"   Similarity Score: {similarity_norm:.4f} (0=no match, 1=perfect match)")
            print(f"   Video ID: {video_id}")
            print(f"   Time Range: {start_time} → {end_time}")
            print(f"   Duration: {num_frames} frames")
            print(f"   Verified: {'✓ YES' if verified else '✗ NO'}")
            
            # Display clothing colors
            upper_color = payload.get("upper_color")
            lower_color = payload.get("lower_color")
            if upper_color or lower_color:
                color_info = []
                if upper_color:
                    color_info.append(f"upper={upper_color}")
                if lower_color:
                    color_info.append(f"lower={lower_color}")
                print(f"   Clothing Colors: {', '.join(color_info)}")
            else:
                print("   Clothing Colors: Not detected")
            
            # Display detected attributes
            attributes = payload.get("attributes", {})
            detected_attrs = []
            if attributes.get("has_hat"):
                detected_attrs.append("👒 Hat")
            if attributes.get("has_hood"):
                detected_attrs.append("🧥 Hood")
            if attributes.get("has_glasses"):
                detected_attrs.append("👓 Glasses")
            
            if detected_attrs:
                print(f"   Attributes: {', '.join(detected_attrs)}")
            else:
                print("   Attributes: None detected")
            
            if carried_objs:
                print(f"   Objects Carried: {', '.join(carried_objs)}")
            else:
                print("   Objects Carried: None detected")

        print(f"\n{'='*80}\n")
        return matches

    except Exception as e:
        print(f"❌ Error during query: {e}")
        import traceback
        traceback.print_exc()
        return []


# ========================
# POST-PROCESSING FACE COMPARISON
# ========================
def compare_face_after_processing(reference_image_path, video_id=None, top_k=10):
    """
    Post-processing face comparison mode.
    After video processing is complete, this function:
    1. Takes a reference face image
    2. Generates face embedding using InsightFace
    3. Queries Qdrant person_tracks collection for similar face embeddings
    4. Returns the most similar matches
    
    Args:
        reference_image_path: Path to reference face image
        video_id: Optional video_id to filter results (None = search all videos)
        top_k: Number of top matches to return
    
    Returns:
        List of (track_id, similarity_score, payload) tuples
    """
    try:
        print("\n" + "="*80)
        print("🔍 POST-PROCESSING FACE COMPARISON")
        print("="*80)
        print(f"Reference image: {reference_image_path}")
        if video_id:
            print(f"Filtering by video_id: {video_id}")
        print(f"Top results: {top_k}")
        print("="*80 + "\n")
        
        # Load reference image
        ref_img = cv2.imread(reference_image_path)
        if ref_img is None:
            print(f"❌ Error: Could not load reference image: {reference_image_path}")
            return []
        
        print(f"✓ Loaded reference image: {ref_img.shape[1]}x{ref_img.shape[0]}")
        
        # Initialize InsightFace if not already done
        from insightface.app import FaceAnalysis
        fa_temp = FaceAnalysis(allowed_modules=['detection', 'recognition'])
        fa_temp.prepare(ctx_id=-1, det_size=(640, 640))
        print("✓ InsightFace initialized")
        
        # Extract face embedding from reference image
        faces = fa_temp.get(ref_img)
        if not faces or len(faces) == 0:
            print("❌ Error: No face detected in reference image")
            return []
        
        # Use the first face (largest)
        ref_face = faces[0]
        ref_embedding = np.array(ref_face.embedding, dtype=np.float32)
        
        # Normalize embedding
        ref_embedding = ref_embedding / (np.linalg.norm(ref_embedding) + 1e-8)
        
        print(f"✓ Extracted face embedding: {len(ref_embedding)}D")
        print(f"  Face confidence: {ref_face.det_score:.4f}")
        print(f"  Face bbox: {ref_face.bbox}")
        
        # Query Qdrant for similar faces
        print("\n🔎 Searching Qdrant person_tracks collection...")
        
        # Build filter if video_id specified
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        search_filter = None
        if video_id is not None:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="video_id",
                        match=MatchValue(value=video_id)
                    )
                ]
            )
        
        # Search using face_vec vector (matches Qdrant named vectors schema)
        try:
            search_results = client.query_points(
                collection_name="person_tracks",
                query=ref_embedding.tolist(),
                using="face_vec",
                query_filter=search_filter,
                limit=top_k,
                with_payload=True
            ).points
        except AttributeError:
            # Fallback for older qdrant-client versions
            try:
                from qdrant_client.models import PointStruct, SearchRequest
                search_results = client.search(
                    collection_name="person_tracks",
                    query_vector=("face_vec", ref_embedding.tolist()),
                    query_filter=search_filter,
                    limit=top_k,
                    with_payload=True
                )
            except Exception:
                # Try another method for even older versions
                search_results = client.search(
                    collection_name="person_tracks",
                    query_vector=ref_embedding.tolist(),
                    query_filter=search_filter,
                    limit=top_k,
                    with_payload=True,
                    search_params={"hnsw_ef": 128, "exact": False}
                )

        # If filtered search returned zero, try without filter to help diagnose mismatched video_id
        if not search_results and search_filter is not None:
            print("⚠ No results with video_id filter — retrying without filter to diagnose...")
            try:
                search_results = client.query_points(
                    collection_name="person_tracks",
                    query=ref_embedding.tolist(),
                    using="face_vec",
                    limit=top_k,
                    with_payload=True
                ).points
            except AttributeError:
                try:
                    search_results = client.search(
                        collection_name="person_tracks",
                        query_vector=("face_vec", ref_embedding.tolist()),
                        limit=top_k,
                        with_payload=True
                    )
                except:
                    search_results = client.search(
                        collection_name="person_tracks",
                        query_vector=ref_embedding.tolist(),
                        limit=top_k,
                        with_payload=True,
                        search_params={"hnsw_ef": 128, "exact": False}
                    )
            except:
                # Try another method for even older versions
                search_results = client.search(
                    collection_name="person_tracks",
                    query_vector=ref_embedding.tolist(),
                    query_filter=search_filter,
                    limit=top_k,
                    with_payload=True,
                    search_params={"hnsw_ef": 128, "exact": False}
                )
        
        print(f"✓ Found {len(search_results)} matches\n")
        
        # Process and display results
        matches = []
        print(f"{'='*80}")
        print(f"📊 TOP {len(search_results)} FACE MATCHES")
        print(f"{'='*80}")
        
        for idx, result in enumerate(search_results, 1):
            similarity_score = result.score  # Cosine similarity
            payload = result.payload
            
            track_id = payload.get("track_id", "Unknown")
            video_id_result = payload.get("video_id", "Unknown")
            start_time = payload.get("start_time", "Unknown")
            end_time = payload.get("end_time", "Unknown")
            num_frames = payload.get("num_frames", 0)
            verified = payload.get("verified", False)
            carried_objs = payload.get("object_carried", [])
            
            matches.append((track_id, similarity_score, payload))
            
            print(f"\n🎯 Match #{idx}")
            print(f"   Track ID: {track_id}")
            print(f"   Similarity Score: {similarity_score:.4f} (higher = better match)")
            print(f"   Video ID: {video_id_result}")
            print(f"   Time Range: {start_time}s → {end_time}s")
            print(f"   Duration: {num_frames} frames")
            print(f"   Verified: {'✓ YES' if verified else '✗ NO'}")
            
            # Display clothing colors
            upper_color = payload.get("upper_color")
            lower_color = payload.get("lower_color")
            if upper_color or lower_color:
                color_info = []
                if upper_color:
                    color_info.append(f"upper={upper_color}")
                if lower_color:
                    color_info.append(f"lower={lower_color}")
                print(f"   Clothing: {', '.join(color_info)}")
            
            # Display attributes
            attributes = payload.get("attributes", {})
            detected_attrs = []
            if attributes.get("has_hat"):
                detected_attrs.append("Hat")
            if attributes.get("has_hood"):
                detected_attrs.append("Hood")
            if attributes.get("has_glasses"):
                detected_attrs.append("Glasses")
            
            if detected_attrs:
                print(f"   Attributes: {', '.join(detected_attrs)}")
            
            if carried_objs:
                print(f"   Objects: {', '.join(carried_objs)}")
        
        print(f"\n{'='*80}\n")
        
        return matches
        
    except Exception as e:
        print(f"❌ Error during post-processing face comparison: {e}")
        import traceback
        traceback.print_exc()
        return []


# ========================
# Interactive Query Loop
# ========================
# print("\n" + "🎤"*40)
# print("\n📝 INTERACTIVE QUERY MODE")
# print("=" * 80)
# print("Enter text prompts to search for people or objects in the video.")
# print("Examples:")
# print("  PERSON QUERIES:")
# print("    - 'person holding a mobile phone'")
# print("    - 'person wearing a red shirt'")
# print("    - 'person with glasses'")
# print("  OBJECT QUERIES:")
# print("    - 'laptop'")
# print("    - 'find me a phone'")
# print("    - 'backpack'")
# print("Type 'quit' or 'exit' to end.\n")
# print("=" * 80 + "\n")

# # Keep querying until user exits
# while True:
#     try:
#         user_prompt = input("🔎 Enter your query: ").strip()
        
#         if user_prompt.lower() in ['quit', 'exit', 'q']:
#             print("\n✅ Exiting query mode. Goodbye!")
#             break
        
#         if not user_prompt:
#             print("⚠ Empty prompt. Please try again.\n")
#             continue
        
#         # Smart routing: detect if query is for objects only or persons
#         query_lower = user_prompt.lower()
#         is_object_only = (
#             # Query is object-only if it mentions object names without "person"
#             ("person" not in query_lower and "people" not in query_lower) and
#             any(obj_keyword in query_lower for obj_keyword in [
#                 "laptop", "computer", "phone", "mobile", "cell",
#                 "backpack", "bag", "handbag", "suitcase", "luggage"
#             ])
#         )
        
#         # Route to appropriate query function
#         if is_object_only:
#             print("   → Detected OBJECT query, searching object_tracks...")
#             matches = query_objects_by_text(user_prompt, top_k=5)
#         else:
#             print("   → Detected PERSON query, searching person_tracks...")
#             matches = query_tracklets_by_text(user_prompt, top_k=5)
        
#         if matches:
#             print("💡 KEY FINDINGS:")
#             if is_object_only:
#                 # Object results
#                 for track_id, score, payload in matches:
#                     object_type = payload.get('object_type', 'Unknown')
#                     print(f"   • {object_type} (Track ID: {track_id}, Match: {score:.2%})")
#             else:
#                 # Person results
#                 for track_id, score, payload in matches:
#                     carried = payload.get('object_carried', [])
#                     print(f"   • Person Track ID: {track_id} (Confidence: {score:.2%})", end="")
#                     if carried:
#                         print(f" - Carrying: {', '.join(carried)}")
#                     else:
#                         print()
        
#         print()
        
#     except KeyboardInterrupt:
#         print("\n\n✅ Query interrupted by user. Goodbye!")
#         break
#     except Exception as e:
#         print(f"❌ Error: {e}\n")

print("\n" + "="*80)
print("🏁 Pipeline and query session complete!")
print("="*80)
# ===== POST-PROCESSING FACE COMPARISON (if image provided) =====
# If a reference face image path is provided via environment variable, run comparison now
REFERENCE_FACE_IMAGE = os.environ.get("REFERENCE_FACE_IMAGE", None)
print(f"\n[DEBUG] REALTIME_FACE_COMPARISON: {REALTIME_FACE_COMPARISON}")
print(f"[DEBUG] REFERENCE_FACE_IMAGE env var: {REFERENCE_FACE_IMAGE}")

if REFERENCE_FACE_IMAGE and not REALTIME_FACE_COMPARISON:
    print("\n" + "="*80)
    print("🔍 RUNNING POST-PROCESSING FACE COMPARISON")
    print("="*80)
    print(f"Reference image path: {REFERENCE_FACE_IMAGE}")
    print(f"Video ID filter: {VIDEO_ID}")
    
    if os.path.exists(REFERENCE_FACE_IMAGE):
        try:
            # Check if Qdrant client is available
            if not client:
                print("❌ Error: Qdrant client not initialized. Cannot perform face comparison.")
            else:
                print("✓ Qdrant client is available")
                
                # Check if person_tracks collection exists
                try:
                    collections = client.get_collections()
                    collection_names = [c.name for c in collections.collections]
                    print(f"✓ Available collections: {collection_names}")
                    
                    if "person_tracks" not in collection_names:
                        print("❌ Error: person_tracks collection not found in Qdrant")
                    else:
                        print("✓ person_tracks collection exists")
                        
                        # Count points in collection
                        count_result = client.count("person_tracks")
                        print(f"✓ Total points in person_tracks: {count_result.count}")
                        
                        # Now run the comparison
                        matches = compare_face_after_processing(
                            reference_image_path=REFERENCE_FACE_IMAGE,
                            video_id=VIDEO_ID,
                            top_k=10
                        )
                        
                        if matches:
                            print("\n" + "="*80)
                            print("✅ FACE COMPARISON RESULTS")
                            print("="*80)
                            for idx, (track_id, sim_score, payload) in enumerate(matches, 1):
                                print(f"\n#{idx}: Person Track {track_id}")
                                print(f"     Similarity: {sim_score:.4f} ({sim_score*100:.2f}%)")
                                print(f"     Time: {payload.get('start_time')}s - {payload.get('end_time')}s")
                                print(f"     Duration: {payload.get('num_frames')} frames")
                                
                                upper = payload.get('upper_color')
                                lower = payload.get('lower_color')
                                if upper or lower:
                                    print(f"     Clothing: {upper or 'N/A'} (upper), {lower or 'N/A'} (lower)")
                                
                                attrs = payload.get('attributes', {})
                                attr_list = []
                                if attrs.get('has_hat'): attr_list.append('Hat')
                                if attrs.get('has_hood'): attr_list.append('Hood')
                                if attrs.get('has_glasses'): attr_list.append('Glasses')
                                if attr_list:
                                    print(f"     Attributes: {', '.join(attr_list)}")
                                
                                objs = payload.get('object_carried', [])
                                if objs:
                                    print(f"     Carrying: {', '.join(objs)}")
                            print("="*80)
                        else:
                            print("\n⚠ No matches found in Qdrant for the given reference face.")
                            print("This could mean:")
                            print("  1. No face embeddings were stored in Qdrant")
                            print("  2. The reference face doesn't match any detected faces")
                            print("  3. Try with a different reference image")
                            
                except Exception as e:
                    print(f"❌ Error checking Qdrant collections: {e}")
                    import traceback
                    traceback.print_exc()
                    
        except Exception as e:
            print(f"\n❌ Error during post-processing comparison: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n❌ Reference face image not found: {REFERENCE_FACE_IMAGE}")
        print(f"   Current working directory: {os.getcwd()}")
        print(f"   Please provide full path to the image or place it in: {os.getcwd()}")
elif REFERENCE_FACE_IMAGE and REALTIME_FACE_COMPARISON:
    print("\n⚠ REFERENCE_FACE_IMAGE provided but REALTIME_FACE_COMPARISON is enabled.")
    print("   Post-processing comparison requires REALTIME_FACE_COMPARISON=false")
else:
    print("\n[INFO] No reference face image provided via REFERENCE_FACE_IMAGE env var.")
    print("       To run post-processing face comparison, set:")
    print("       export REFERENCE_FACE_IMAGE=/path/to/face.jpg")
