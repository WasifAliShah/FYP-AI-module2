
'''
yolo face detection integrated with insightface + bicubic upscaling + reid
'''
import time
from collections import deque
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from torchvision.models import resnet50
from ultralytics import YOLO
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, NamedVector
from dotenv import load_dotenv
import os
import qdrant_collections  # for collection constants
from PIL import Image

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
    # create_qdrant_schema is defined in qdrant_collections.py; call it qualified
    qdrant_collections.create_qdrant_schema(client)
    print("Qdrant schema created successfully.")
except Exception as e:
    print("Failed to create Qdrant schema:", e)

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

# ByteTrack for multi-object tracking
USE_BYTETRACK = False
byte_tracker = None
byte_tracker_objects = None
try:
    from cjm_byte_track.core import BYTETracker
    USE_BYTETRACK = True
    print("ByteTrack module found")
except Exception as e:
    print(f"ByteTrack not available: {e}")
    USE_BYTETRACK = False
    

# ----------------------
# CONFIG (CPU OPTIMIZED)
# ----------------------
VIDEO_PATH = "combined.mp4"
REF_FACE_PATHS = ["sabbas.jpg"]

YOLO_PERSON_MODEL = "yolov8m.pt"        # your person model
YOLO_FACE_MODEL = "yolov8m-face.pt"     # recommended: yolov8n-face or yolov8m-face
YOLO_OBJECT_MODEL = "yolov8m.pt"       # general object detection (laptops, phones, bags, etc.)
DETECT_EVERY_N_FRAMES = 13

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

# Initialize ByteTrack if available
if USE_BYTETRACK:
    try:
        # Get video FPS for ByteTrack initialization
        cap_temp = cv2.VideoCapture(VIDEO_PATH)
        fps = cap_temp.get(cv2.CAP_PROP_FPS) or 30.0  # Default to 30 if unavailable
        cap_temp.release()
        
        # Initialize ByteTrack with appropriate parameters
        # track_thresh: detection confidence threshold (0.25 = use low confidence detections)
        # track_buffer: frames to keep lost tracks (increased for better continuity)
        # match_thresh: IoU threshold for matching (lower = more lenient, better for fast movement)
        byte_tracker = BYTETracker(
            track_thresh=0.25,  # Use low confidence detections (ByteTrack's strength)
            track_buffer=60,  # Keep lost tracks longer (2x TRACKLET_MAX_AGE for better continuity)
            match_thresh=0.6,  # Lower threshold for more lenient matching (better for fast movement)
            frame_rate=fps
        )
        print(f"✓ ByteTrack initialized for persons (FPS: {fps:.1f})")
        
        # Initialize separate ByteTrack for objects (better handling of sudden movements)
        byte_tracker_objects = BYTETracker(
            track_thresh=0.2,  # Lower threshold for objects (they can be harder to detect)
            track_buffer=40,  # Keep lost tracks for 40 frames
            match_thresh=0.5,  # More lenient matching for objects (handles sudden movements better)
            frame_rate=fps
        )
        print(f"✓ ByteTrack initialized for objects (FPS: {fps:.1f})")
    except Exception as e:
        print(f"⚠ ByteTrack initialization failed: {e}")
        USE_BYTETRACK = False
        byte_tracker = None
        byte_tracker_objects = None
else:
    print("⚠ ByteTrack not available - using IOU-based tracking only")
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

# ----------------------
# Build reference embeddings
# ----------------------
ref_face_embs = []
ref_reid_embs = []

def normalize(v):
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v)+1e-8)

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
            print(f"✓ Loaded reference face embedding from {path} (face detected, embedding size: {len(face_emb)})")
        else:
            print(f"⚠ Warning: No face detected in reference image {path} - face recognition will not work!")
    else:
        print("⚠ Warning: InsightFace not available - face recognition disabled!")

    # reid embedding
    reid_emb = reid_encode(img)
    if reid_emb is not None:
        ref_reid_embs.append(reid_emb)
        print(f"✓ Loaded reference ReID embedding from {path} (embedding size: {len(reid_emb)})")

print(f"\n=== Reference Embeddings Summary ===")
print(f"Face embeddings: {len(ref_face_embs)}")
print(f"ReID embeddings: {len(ref_reid_embs)}")
if len(ref_face_embs) == 0:
    print("⚠ CRITICAL: No face embeddings loaded! Face recognition will be disabled.")
    
# Determine if reference is face-only (face-only images are not suitable for ReID)
USE_REID_VERIFICATION = len(ref_face_embs) > 0 and len(ref_reid_embs) > 0
# If we have face embeddings but reference might be face-only, disable ReID verification
# ReID is unreliable when reference is face-only (not full body)
if len(ref_face_embs) > 0:
    print("⚠ IMPORTANT: Reference contains face images. ReID verification DISABLED (unreliable for face-only references).")
    print("   Only face recognition will be used for verification.")
    USE_REID_VERIFICATION = False
print("=" * 40 + "\n")

# ===== DEBUG: Identify embedding dimensions =====
FACE_EMBEDDING_DIM = len(ref_face_embs[0]) if len(ref_face_embs) > 0 else None
REID_EMBEDDING_DIM = len(ref_reid_embs[0]) if len(ref_reid_embs) > 0 else None

print("=" * 40)
print("🔍 EMBEDDING DIMENSIONS DETECTED:")
print(f"  Face embedding (InsightFace): {FACE_EMBEDDING_DIM}D")
print(f"  ReID embedding (TorchReID OSNet): {REID_EMBEDDING_DIM}D")
print("=" * 40 + "\n")

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
        payload["upper_color"] = None
        payload["lower_color"] = None
        payload["object_carried"] = tracklet.carried_summary()
        payload["verified"] = tracklet.verified  # Indicate if this tracklet was verified against reference
        
        # Convert embeddings to lists for Qdrant
        face_vec = face_avg.tolist() if isinstance(face_avg, np.ndarray) else list(face_avg)
        reid_vec = reid_avg.tolist() if isinstance(reid_avg, np.ndarray) else list(reid_avg)
        
        # multi_vec: prefer CLIP embedding, fallback to face embedding padded to 768
        clip_avg = tracklet.avg_clip()

        def pad_to_768(vec):
            """Pad vector to 768D by appending zeros."""
            vec_list = vec.tolist() if isinstance(vec, np.ndarray) else list(vec)
            if len(vec_list) >= 768:
                return vec_list[:768]
            return vec_list + [0.0] * (768 - len(vec_list))
        
        if clip_avg is not None:
            multi_vec = pad_to_768(clip_avg)
        else:
            multi_vec = pad_to_768(face_avg)
        
        # Build vectors dict for NamedVectors (per spec: face_vec 512D, reid_vec 512D, multi_vec 768D)
        vectors = {
            "face_vec": face_vec,          # 512D (InsightFace)
            "reid_vec": reid_vec,          # 512D (TorchReID)
            "multi_vec": multi_vec,        # 768D (CLIP or padded face_vec)
        }
        
        # DEBUG: Print what's being sent to Qdrant
        print("\n" + "="*80)
        print(f"🔍 DEBUG: Inserting Tracklet {tracklet.id} to Qdrant ({'VERIFIED' if tracklet.verified else 'UNVERIFIED'})")
        print("="*80)
        print(f"Point ID: {point_id}")
        print(f"\n📦 PAYLOAD:")
        print(f"  video_id: {payload['video_id']}")
        print(f"  track_id: {payload['track_id']}")
        print(f"  start_time: {payload['start_time']}")
        print(f"  end_time: {payload['end_time']}")
        print(f"  num_frames: {payload['num_frames']}")
        print(f"  avg_confidence: {payload['avg_confidence']}")
        print(f"  timestamp: {payload['timestamp']}")
        print(f"  person_gender: {payload['person_gender']}")
        print(f"  upper_color: {payload['upper_color']}")
        print(f"  lower_color: {payload['lower_color']}")
        print(f"  object_carried: {payload['object_carried']}")
        print(f"  verified: {payload['verified']}")
        if 'segment_id' in payload:
            print(f"  segment_id: {payload['segment_id']}")
        
        print(f"\n🔢 VECTORS:")
        face_status = "InsightFace" if len(tracklet.face_embs) > 0 else "Zero (no face detected)"
        reid_status = "TorchReID/ResNet50" if len(tracklet.reid_embs) > 0 else "Zero (no ReID)"
        print(f"  face_vec: {len(face_vec)}D ({face_status})")
        print(f"    Sample: [{face_vec[0]:.6f}, {face_vec[1]:.6f}, {face_vec[2]:.6f}, ...]")
        print(f"  reid_vec: {len(reid_vec)}D ({reid_status})")
        print(f"    Sample: [{reid_vec[0]:.6f}, {reid_vec[1]:.6f}, {reid_vec[2]:.6f}, ...]")
        print(f"  multi_vec: {len(multi_vec)}D ({'CLIP' if clip_avg is not None else 'Face (padded)'})")
        print(f"    Sample: [{multi_vec[0]:.6f}, {multi_vec[1]:.6f}, {multi_vec[2]:.6f}, ...]")
        
        print(f"\n📊 TRACKLET STATS:")
        print(f"  Verified: {tracklet.verified}")
        print(f"  Face embeddings collected: {len(tracklet.face_embs)}")
        print(f"  ReID embeddings collected: {len(tracklet.reid_embs)}")
        print(f"  CLIP embeddings collected: {len(tracklet.clip_embs)}")
        print(f"  Carried objects observed: {len(tracklet.carried_objects)}")
        print("="*80 + "\n")
        
        # Insert into Qdrant
        from qdrant_client.models import PointStruct
        point = PointStruct(
            id=point_id,
            vector=vectors,  # NamedVectors
            payload=payload
        )
        
        client.upsert(
            collection_name="person_tracks",
            points=[point]
        )
        
        status = "VERIFIED" if tracklet.verified else "UNVERIFIED"
        print(f"✅ Inserted {status} tracklet {tracklet.id} to Qdrant (ID: {point_id})")
        return True
        
    except Exception as e:
        print(f"⚠ Failed to insert tracklet {tracklet.id} to Qdrant: {e}")
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
        self.verified = False
        self.inserted = False  # set True once pushed to DB
        self.tracker = None

    def update(self, bbox, idx, face_emb=None, reid_emb=None, face_size=0, clip_emb=None, carried=None):
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
        """Return unique list of carried object names."""
        if not self.carried_objects:
            return []
        # Preserve insertion order of most recent observations
        seen = {}
        for name in self.carried_objects:
            if name not in seen:
                seen[name] = True
        return list(seen.keys())

# ----------------------
# ObjectTracklet class (for objects like backpacks, laptops, etc.)
# ----------------------
class ObjectTracklet:
    def __init__(self, oid, class_name, bbox, confidence, frame_idx):
        self.id = oid
        self.class_name = class_name
        self.bboxes = deque(maxlen=30)  # Keep last 30 bboxes
        self.bboxes.append(bbox)
        self.confidences = deque(maxlen=30)
        self.confidences.append(confidence)
        self.last_frame = frame_idx
        self.first_frame = frame_idx

    def update(self, bbox, confidence, frame_idx):
        self.bboxes.append(bbox)
        self.confidences.append(confidence)
        self.last_frame = frame_idx
    
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

# small helper to map face boxes per frame
face_boxes_frame = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    h, w = frame.shape[:2]

    # On non-detection frames, update ByteTrack with empty detections to maintain tracking
    # This is critical for tracking continuity - ByteTrack needs to be updated every frame
    if USE_BYTETRACK and byte_tracker is not None and frame_idx % DETECT_EVERY_N_FRAMES != 0:
        try:
            # Update ByteTrack with empty detections to maintain existing tracks
            # ByteTrack will predict positions for existing tracks even without new detections
            empty_detections = np.array([], dtype=np.float32).reshape(0, 5)
            img_info = (h, w)
            img_size = (w, h)
            tracked_objects = byte_tracker.update(empty_detections, img_info, img_size)
            
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
    
    # run detection every N frames
    if frame_idx % DETECT_EVERY_N_FRAMES == 0:
        # person detection - use lower confidence for ByteTrack (it handles low-confidence detections well)
        # ByteTrack's strength is using low-confidence detections for better association
        p_results = yolo_person.predict(frame, imgsz=640, conf=0.3, classes=[0], verbose=False)
        person_boxes = []
        person_detections = []  # For ByteTrack: [x1, y1, x2, y2, score]
        
        if len(p_results):
            all_boxes = []
            all_scores = []
            all_detections = []
            
            for b in p_results[0].boxes:
                x1,y1,x2,y2 = b.xyxy[0].cpu().numpy().astype(int)
                conf = float(b.conf[0].cpu().numpy())
                x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w,x2),min(h,y2)
                
                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Filter out small/partial detections (likely hands, arms, etc.)
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                frame_area = w * h
                
                # Skip very small boxes (likely body parts, not full persons)
                # Minimum size: at least 2% of frame area, or minimum 100x150 pixels
                min_area = max(frame_area * 0.02, 100 * 150)
                if box_area < min_area:
                    continue
                
                # Skip boxes that are too wide relative to height (likely not a person)
                # Person boxes should be roughly 1:2 to 1:3 width:height ratio
                aspect_ratio = box_width / max(box_height, 1)
                if aspect_ratio > 0.8:  # Too wide (likely not a person)
                    continue
                
                # Skip boxes that are too tall and narrow (likely not a person)
                if aspect_ratio < 0.2:  # Too narrow
                    continue
                
                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(conf)
                all_detections.append((x1,y1,x2,y2))
            
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
                tracked_objects = byte_tracker.update(detections_array, img_info, img_size)
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
                    tracked_obj_tracks = byte_tracker_objects.update(detections_array, img_info, img_size)
                    
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
                                # ByteTrack tracks come from detections, so there should always be a match
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
                        if track_id in object_tracklets:
                            object_tracklets[track_id].update(bbox_to_store, conf, frame_idx)
                        else:
                            object_tracklets[track_id] = ObjectTracklet(track_id, class_name, bbox_to_store, conf, frame_idx)
                    
                    # Clean up old tracks
                    tracks_to_remove = []
                    for oid, obj_track in object_tracklets.items():
                        if frame_idx - obj_track.last_frame > OBJ_TRACK_MAX_AGE:
                            tracks_to_remove.append(oid)
                    for oid in tracks_to_remove:
                        del object_tracklets[oid]
                    
                    if frame_idx <= DETECT_EVERY_N_FRAMES * 3:
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
                    
                    for oid, obj_track in object_tracklets.items():
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
                        object_tracklets[best_match_id].update(bbox, conf, frame_idx)
                    else:
                        object_tracklets[next_obj_id] = ObjectTracklet(next_obj_id, class_name, bbox, conf, frame_idx)
                        next_obj_id += 1
        
        # Update ByteTrack for objects on non-detection frames too (maintain tracking continuity)
        if USE_BYTETRACK and byte_tracker_objects is not None and frame_idx % DETECT_EVERY_N_FRAMES != 0:
            try:
                empty_detections = np.array([], dtype=np.float32).reshape(0, 5)
                img_info = (h, w)
                img_size = (w, h)
                tracked_obj_tracks = byte_tracker_objects.update(empty_detections, img_info, img_size)
                
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
                        object_tracklets[track_id].update(byte_track_bbox, avg_conf, frame_idx)
                    # Note: Don't create new tracklets on non-detection frames - wait for next detection frame
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
        if frame_idx <= DETECT_EVERY_N_FRAMES * 5:
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
                # Show all detected objects for debugging
                if frame_idx <= DETECT_EVERY_N_FRAMES * 2:
                    print(f"[Debug] All objects detected (conf >= 0.15): {list(all_detected.keys())}")

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
                
                # ---- Face detection: Try multiple methods for best results
                face_emb = None
                face_size = 0
                carried_objs = []

                # Method 1: Try YOLO face boxes first (if available)
                matched_face = None
                for fb in face_boxes_frame:
                    if box_inside(fb, box) or iou(fb, box) > 0.1:
                        matched_face = fb
                        break

                if matched_face is not None:
                    fx1,fy1,fx2,fy2 = matched_face
                    # ensure clamp
                    fx1,fy1,fx2,fy2 = max(0,fx1),max(0,fy1),min(w,fx2),min(h,fy2)
                    face_crop = frame[fy1:fy2, fx1:fx2].copy()
                    face_size = (fx2-fx1)

                    # For faces < 60px, ALWAYS upscale before getting embedding for better quality
                    # This improves recognition accuracy for distant faces
                    if face_size < 60:
                        # Use larger upscale factor for very small faces
                        upscale_factor = 4 if face_size < 30 else (3 if face_size < 45 else 2)
                        face_crop_up = sr_enhance(face_crop, factor=upscale_factor)
                        
                        # Try InsightFace on upscaled crop first (better quality)
                        if fa:
                            try:
                                faces_up = fa.get(face_crop_up)
                                if faces_up and len(faces_up) > 0:
                                    # Debug: check what attributes are available
                                    if frame_idx <= DETECT_EVERY_N_FRAMES * 3:
                                        print(f"  → InsightFace found {len(faces_up)} face(s) in upscaled crop, has embedding: {hasattr(faces_up[0], 'embedding')}")
                                    if hasattr(faces_up[0], 'embedding') and faces_up[0].embedding is not None:
                                        face_emb = normalize(np.array(faces_up[0].embedding))
                                        if frame_idx <= DETECT_EVERY_N_FRAMES * 3:
                                            print(f"  → ✓ Got embedding from upscaled face crop!")
                                    # Update face_size from upscaled detection
                                    if hasattr(faces_up[0], 'bbox'):
                                        detected_w = int(faces_up[0].bbox[2] - faces_up[0].bbox[0])
                                        if detected_w > 0:
                                            face_size = detected_w
                            except Exception as e:
                                if frame_idx <= DETECT_EVERY_N_FRAMES * 3:
                                    print(f"  → Error getting embedding from upscaled crop: {e}")
                                pass
                    else:
                        # For larger faces (>= 60px), try InsightFace on original crop
                        if fa:
                            try:
                                faces = fa.get(face_crop)
                                if faces and len(faces) > 0:
                                    # Try to get embedding directly
                                    if hasattr(faces[0], 'embedding') and faces[0].embedding is not None:
                                        face_emb = normalize(np.array(faces[0].embedding))
                                    # Update face_size from InsightFace detection if available
                                    if hasattr(faces[0], 'bbox'):
                                        detected_w = int(faces[0].bbox[2] - faces[0].bbox[0])
                                        if detected_w > 0:
                                            face_size = detected_w
                            except Exception as e:
                                pass
                    
                    # Fallback: if upscaling didn't work for small faces, try original
                    if face_emb is None and face_size < 60:
                        if fa:
                            try:
                                faces = fa.get(face_crop)
                                if faces and len(faces) > 0:
                                    if hasattr(faces[0], 'embedding') and faces[0].embedding is not None:
                                        face_emb = normalize(np.array(faces[0].embedding))
                            except Exception as e:
                                pass
                    
                    # Final fallback: if still no embedding and face is small, try upscaling
                    if face_emb is None and face_size < 80:
                        # Use larger upscale factor for very small faces
                        upscale_factor = 4 if face_size < 30 else (3 if face_size < 50 else 2)
                        face_crop_up = sr_enhance(face_crop, factor=upscale_factor)
                        
                        # Try InsightFace on upscaled crop
                        if fa:
                            try:
                                faces_up = fa.get(face_crop_up)
                                if faces_up and len(faces_up) > 0:
                                    if hasattr(faces_up[0], 'embedding') and faces_up[0].embedding is not None:
                                        face_emb = normalize(np.array(faces_up[0].embedding))
                                    # Update face_size from upscaled detection
                                    if hasattr(faces_up[0], 'bbox'):
                                        detected_w = int(faces_up[0].bbox[2] - faces_up[0].bbox[0])
                                        if detected_w > 0:
                                            face_size = detected_w
                            except Exception as e:
                                pass

                # Method 2: ALWAYS try InsightFace on person crop (most reliable, works even if YOLO misses faces)
                # This is critical because InsightFace is better at detecting faces in person crops
                if face_emb is None and fa:
                    try:
                        # First try on person crop directly - this should work!
                        # InsightFace is very good at detecting faces in person crops
                        faces = fa.get(crop_person)
                        if faces and len(faces) > 0:
                            f = faces[0]
                            # Debug output
                            if frame_idx <= DETECT_EVERY_N_FRAMES * 3:
                                print(f"  → InsightFace found {len(faces)} face(s) in person crop, has embedding: {hasattr(f, 'embedding')}")
                            
                            # face bbox is relative to crop_person: compute absolute width
                            if hasattr(f, 'bbox') and f.bbox is not None:
                                fw = int(f.bbox[2] - f.bbox[0])
                                face_size = max(face_size, fw)  # Use larger of YOLO or InsightFace size
                            
                            # ALWAYS try to get embedding directly first (even for small faces)
                            # InsightFace embeddings work well even on small faces
                            if hasattr(f, 'embedding') and f.embedding is not None:
                                face_emb = normalize(np.array(f.embedding))
                                if frame_idx <= DETECT_EVERY_N_FRAMES * 3:
                                    print(f"  → ✓ Got embedding directly from person crop!")
                            
                            # For faces < 60px, ALSO try upscaling for potentially better quality
                            # This improves recognition accuracy for distant faces
                            if face_size < 60:
                            # Extract face region and upscale it for better embedding quality
                                if hasattr(f, 'bbox') and f.bbox is not None:
                                    bx1 = max(0, int(f.bbox[0])); by1 = max(0, int(f.bbox[1]))
                                    bx2 = min(crop_person.shape[1], int(f.bbox[2])); by2 = min(crop_person.shape[0], int(f.bbox[3]))
                                    if bx2 > bx1 and by2 > by1:
                                        face_region = crop_person[by1:by2, bx1:bx2].copy()
                                        if face_region.size > 0:
                                            # Use larger upscale factor for very small faces
                                            upscale_factor = 4 if face_size < 30 else (3 if face_size < 45 else 2)
                                            face_region_up = sr_enhance(face_region, factor=upscale_factor)
                                            
                                            # Get embedding from upscaled face (better quality)
                                            faces_up = fa.get(face_region_up)
                                            if faces_up and len(faces_up) > 0:
                                                if hasattr(faces_up[0], 'embedding') and faces_up[0].embedding is not None:
                                                    # Use upscaled embedding if we don't have one, or if it's better quality
                                                    emb_up = normalize(np.array(faces_up[0].embedding))
                                                    if face_emb is None:
                                                        face_emb = emb_up
                                                        if frame_idx <= DETECT_EVERY_N_FRAMES * 3:
                                                            print(f"  → ✓ Got embedding from upscaled face region!")
                                                # Update face_size from upscaled detection
                                                if hasattr(faces_up[0], 'bbox'):
                                                    detected_w = int(faces_up[0].bbox[2] - faces_up[0].bbox[0])
                                                    if detected_w > 0:
                                                        face_size = detected_w
                            
                            # Fallback: if upscaling didn't work, try direct embedding
                            if face_emb is None and face_size >= 60:
                                if hasattr(f, 'embedding') and f.embedding is not None:
                                    face_emb = normalize(np.array(f.embedding))
                            
                            # Final fallback: if still no embedding and face is small, try upscaling
                            if face_emb is None and face_size < 80:
                            # Extract face region and upscale it for better quality
                                if hasattr(f, 'bbox') and f.bbox is not None:
                                    bx1 = max(0, int(f.bbox[0])); by1 = max(0, int(f.bbox[1]))
                                    bx2 = min(crop_person.shape[1], int(f.bbox[2])); by2 = min(crop_person.shape[0], int(f.bbox[3]))
                                    if bx2 > bx1 and by2 > by1:
                                        face_region = crop_person[by1:by2, bx1:bx2].copy()
                                        if face_region.size > 0:
                                            # Use larger upscale factor for very small faces
                                            upscale_factor = 4 if face_size < 30 else (3 if face_size < 50 else 2)
                                            face_region_up = sr_enhance(face_region, factor=upscale_factor)
                                            
                                            # Get embedding from upscaled face (better quality)
                                            faces_up = fa.get(face_region_up)
                                            if faces_up and len(faces_up) > 0:
                                                if hasattr(faces_up[0], 'embedding') and faces_up[0].embedding is not None:
                                                    face_emb = normalize(np.array(faces_up[0].embedding))
                                                # Update face_size from upscaled detection
                                                if hasattr(faces_up[0], 'bbox'):
                                                    detected_w = int(faces_up[0].bbox[2] - faces_up[0].bbox[0])
                                                    if detected_w > 0:
                                                        face_size = detected_w
                        else:
                            # If InsightFace didn't find face in person crop, try on expanded region around person
                            # Expand person bbox slightly and try again
                            expand = 30  # Increased expansion
                            x1_exp = max(0, x1 - expand)
                            y1_exp = max(0, y1 - expand)
                            x2_exp = min(w, x2 + expand)
                            y2_exp = min(h, y2 + expand)
                            expanded_crop = frame[y1_exp:y2_exp, x1_exp:x2_exp].copy()
                            if expanded_crop.size > 0:
                                faces_exp = fa.get(expanded_crop)
                                if faces_exp and len(faces_exp) > 0:
                                    f = faces_exp[0]
                                    if hasattr(f, 'bbox') and f.bbox is not None:
                                        fw = int(f.bbox[2] - f.bbox[0])
                                        face_size = fw
                                    if hasattr(f, 'embedding') and f.embedding is not None:
                                        face_emb = normalize(np.array(f.embedding))
                    except Exception as e:
                        # Add debug info for failures
                        if frame_idx <= DETECT_EVERY_N_FRAMES * 3:
                            print(f"  → InsightFace error on person crop: {e}")
                        pass

                # ---- ReID embedding (on person crop)
                reid_emb = reid_encode(crop_person)
                
                # ---- CLIP embedding (on person crop)
                clip_emb = clip_encode(crop_person)

                # ---- Carried objects association (objects whose center lies in person box or good IoU)
                if detected_objects:
                    for class_name, obj_bbox, obj_conf in detected_objects:
                        if box_inside(obj_bbox, vis_bbox) or iou(obj_bbox, vis_bbox) > 0.2:
                            carried_objs.append(class_name)

                # Update tracklet with face and ReID embeddings
                # Use vis_bbox (person detection box) for accurate visualization
                t = tracklets[current_tid]
                t.update(vis_bbox, frame_idx, face_emb, reid_emb, face_size, clip_emb, carried_objs)
                
                # Debug output for first few detections
                if frame_idx <= DETECT_EVERY_N_FRAMES * 3:
                    face_status = "✓" if face_emb is not None else "✗"
                    yolo_faces = len(face_boxes_frame)
                    person_h, person_w = crop_person.shape[:2]
                    upscale_info = ""
                    if face_size > 0 and face_size < 80:
                        upscale_factor = 4 if face_size < 30 else (3 if face_size < 50 else 2)
                        upscale_info = f" (upscaled {upscale_factor}x for better quality)"
                    print(f"[Frame {frame_idx}] Tracklet {current_tid}: "
                          f"YOLO faces: {yolo_faces}, Face detected: {face_status} (size: {face_size}px){upscale_info}, "
                          f"Person crop: {person_w}x{person_h}px, ReID: {'✓' if reid_emb is not None else '✗'}")
                    if face_emb is None:
                        if yolo_faces == 0:
                            print(f"  → No YOLO faces found, trying InsightFace on person crop ({person_w}x{person_h}px)...")
                        else:
                            print(f"  → YOLO found {yolo_faces} face(s) but InsightFace failed to extract embedding!")
                    if face_emb is not None:
                        print(f"  → ✓ Face successfully detected and embedded! (size: {face_size}px)")
        
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
                face_emb = None
                face_size = 0
                
                # Try face detection
                matched_face = None
                for fb in face_boxes_frame:
                    if box_inside(fb, box) or iou(fb, box) > 0.1:
                        matched_face = fb
                        break
                
                if matched_face is not None:
                    fx1,fy1,fx2,fy2 = matched_face
                    fx1,fy1,fx2,fy2 = max(0,fx1),max(0,fy1),min(w,fx2),min(h,fy2)
                    face_crop = frame[fy1:fy2, fx1:fx2].copy()
                    face_size = (fx2-fx1)
                    if fa:
                        try:
                            faces = fa.get(face_crop)
                            if faces and len(faces) > 0:
                                if hasattr(faces[0], 'embedding') and faces[0].embedding is not None:
                                    face_emb = normalize(np.array(faces[0].embedding))
                        except:
                            pass
                
                if face_emb is None and fa:
                    try:
                        faces = fa.get(crop_person)
                        if faces and len(faces) > 0:
                            f = faces[0]
                            if hasattr(f, 'embedding') and f.embedding is not None:
                                face_emb = normalize(np.array(f.embedding))
                    except:
                        pass
                
                reid_emb = reid_encode(crop_person)
                
                clip_emb = clip_encode(crop_person)
                
                if detected_objects:
                    for class_name, obj_bbox, obj_conf in detected_objects:
                        if box_inside(obj_bbox, box) or iou(obj_bbox, box) > 0.2:
                            carried_objs.append(class_name)

                t = tracklets[current_tid]
                t.update(box, frame_idx, face_emb, reid_emb, face_size, clip_emb, carried_objs)
        
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
                
                # Face and ReID extraction (same as above)
                face_emb = None
                face_size = 0
                carried_objs = []
                
                # Method 1: Try YOLO face boxes first
                matched_face = None
                for fb in face_boxes_frame:
                    if box_inside(fb, box) or iou(fb, box) > 0.1:
                        matched_face = fb
                        break
                
                if matched_face is not None:
                    fx1,fy1,fx2,fy2 = matched_face
                    fx1,fy1,fx2,fy2 = max(0,fx1),max(0,fy1),min(w,fx2),min(h,fy2)
                    face_crop = frame[fy1:fy2, fx1:fx2].copy()
                    face_size = (fx2-fx1)
                    
                    if face_size < 60:
                        upscale_factor = 4 if face_size < 30 else (3 if face_size < 45 else 2)
                        face_crop_up = sr_enhance(face_crop, factor=upscale_factor)
                        if fa:
                            try:
                                faces_up = fa.get(face_crop_up)
                                if faces_up and len(faces_up) > 0:
                                    if hasattr(faces_up[0], 'embedding') and faces_up[0].embedding is not None:
                                        face_emb = normalize(np.array(faces_up[0].embedding))
                                    if hasattr(faces_up[0], 'bbox'):
                                        detected_w = int(faces_up[0].bbox[2] - faces_up[0].bbox[0])
                                        if detected_w > 0:
                                            face_size = detected_w
                            except Exception:
                                pass
                    else:
                        if fa:
                            try:
                                faces = fa.get(face_crop)
                                if faces and len(faces) > 0:
                                    if hasattr(faces[0], 'embedding') and faces[0].embedding is not None:
                                        face_emb = normalize(np.array(faces[0].embedding))
                                    if hasattr(faces[0], 'bbox'):
                                        detected_w = int(faces[0].bbox[2] - faces[0].bbox[0])
                                        if detected_w > 0:
                                            face_size = detected_w
                            except Exception:
                                pass
                
                # Method 2: Always try InsightFace on person crop
                if face_emb is None and fa:
                    try:
                        faces = fa.get(crop_person)
                        if faces and len(faces) > 0:
                            f = faces[0]
                            if hasattr(f, 'bbox') and f.bbox is not None:
                                fw = int(f.bbox[2] - f.bbox[0])
                                face_size = max(face_size, fw)
                            if hasattr(f, 'embedding') and f.embedding is not None:
                                face_emb = normalize(np.array(f.embedding))
                    except Exception:
                        pass
                
                # ReID embedding
                reid_emb = reid_encode(crop_person)
                
                clip_emb = clip_encode(crop_person)
                
                if detected_objects:
                    for class_name, obj_bbox, obj_conf in detected_objects:
                        if box_inside(obj_bbox, box) or iou(obj_bbox, box) > 0.2:
                            carried_objs.append(class_name)
                
                # Update tracklet
                t = tracklets[current_tid]
                t.update(box, frame_idx, face_emb, reid_emb, face_size, clip_emb, carried_objs)

    # --------------------------
    # Verification logic and tracklet insertion
    # --------------------------
    for tid, t in list(tracklets.items()):

        if frame_idx - t.last_frame > TRACKLET_MAX_AGE:
            # Track has ended (no updates for TRACKLET_MAX_AGE frames)
            # Insert ALL tracklets (verified and unverified) when they end
            if not t.inserted and client:
                t.inserted = insert_tracklet_to_qdrant(client, t, video_id=1, segment_id=None, frame_rate=30.0)
            del tracklets[tid]
            continue

        if not t.verified and (len(t.face_embs)+len(t.reid_embs)) >= AGGREGATION_FRAMES:

            face_avg = t.avg_face()
            reid_avg = t.avg_reid()
            
            # Use actual detected face size instead of estimate
            face_width = t.avg_face_size()

            # FACE RECOGNITION ONLY: Since reference is face-only, ReID is unreliable and disabled
            face_ok, face_score = False, None
            
            # Only verify via face recognition - ReID is disabled for face-only references
            if face_avg is not None and len(t.face_embs) > 0:
                face_ok, face_score = is_face_match(face_avg, face_width)
                if face_ok:
                    t.verified = True
                    print(f"[VERIFIED] Tracklet {tid} via FACE RECOGNITION  face_score={face_score:.4f}  face_size={face_width}px  face_embs={len(t.face_embs)}")
                else:
                    # Face detected but doesn't match - do NOT verify (ReID disabled for face-only refs)
                    print(f"[REJECTED] Tracklet {tid} face detected but NO MATCH (score={face_score:.4f}) - ReID disabled for face-only reference")
            else:
                # No face embeddings collected - face detection failed
                # DO NOT use ReID as fallback when reference is face-only (unreliable)
                print(f"[REJECTED] Tracklet {tid} NO FACE DETECTED - Cannot verify without face recognition (ReID disabled for face-only reference)")
                print(f"  → Face embeddings: {len(t.face_embs)}, ReID embeddings: {len(t.reid_embs)}")

            # ByteTrack handles tracking automatically, no need for manual tracker initialization

    # --------------------------
    # Visualization
    # --------------------------
    vis = frame.copy()
    
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

    cv2.imshow("Hybrid Face+ReID CPU Pipeline (YOLO-face integrated)", vis)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# After loop ends, insert any remaining tracklets (verified and unverified) that were not inserted yet
for tid, t in list(tracklets.items()):
    if not t.inserted and client:
        t.inserted = insert_tracklet_to_qdrant(client, t, video_id=1, segment_id=None, frame_rate=30.0)

cap.release()
cv2.destroyAllWindows()