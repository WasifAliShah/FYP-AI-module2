
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

# ----------------------
# CONFIG (CPU OPTIMIZED)
# ----------------------
VIDEO_PATH = "combined.mp4"
REF_FACE_PATHS = ["zeeshan.jpg"]

YOLO_PERSON_MODEL = "yolov8m.pt"        # your person model
YOLO_FACE_MODEL = "yolov8m-face.pt"     # recommended: yolov8n-face or yolov8m-face
DETECT_EVERY_N_FRAMES = 15

# thresholds (tune as needed)
# Face recognition thresholds - LOWER scores = better matches (cosine distance)
# Typical good matches: < 0.3 (excellent), 0.3-0.5 (good), 0.5-0.65 (acceptable for small faces)
# Scores > 0.70 are typically NOT matches (different people)
FACE_STRICT = 0.50  # For large, clear faces (>= 100px)
FACE_LOOSE  = 0.60  # For medium faces (60-100px)
FACE_SMALL_MAX = 0.70  # Maximum threshold for very small faces - allows recognition of distant people
REID_THRESHOLD_CPU = 0.45  # Increased to reduce ReID false positives (was 0.45)

AGGREGATION_FRAMES = 10
TRACKLET_MAX_AGE = 30
IOU_THRESHOLD = 0.40

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)
print("Real-ESRGAN available:", USE_REAL_ESRGAN)

# ----------------------
# LOAD MODELS
# ----------------------
yolo_person = YOLO(YOLO_PERSON_MODEL)
yolo_face = YOLO(YOLO_FACE_MODEL)   # face detector

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

# ReID fallback encoder (ResNet50)
def build_resnet_encoder():
    model = resnet50(pretrained=True).eval().to(DEVICE)
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((128, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return model, transform

reid_model, reid_tf = build_resnet_encoder()

def reid_encode(img):
    try:
        x = reid_tf(img[:,:,::-1]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = reid_model(x).cpu().numpy().squeeze()
        feat = feat / (np.linalg.norm(feat)+1e-8)
        return feat.astype(np.float32)
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

# ----------------------
# Tracklet class
# ----------------------
class Tracklet:
    def __init__(self, tid, bbox, frame_idx):
        self.id = tid
        self.bboxes = deque(maxlen=AGGREGATION_FRAMES)
        self.bboxes.append(bbox)
        self.last_frame = frame_idx
        self.face_embs = []
        self.face_sizes = []  # Store actual detected face sizes
        self.reid_embs = []
        self.verified = False
        self.tracker = None

    def update(self, bbox, idx, face_emb=None, reid_emb=None, face_size=0):
        self.bboxes.append(bbox)
        self.last_frame = idx
        if face_emb is not None: 
            self.face_embs.append(face_emb)
            if face_size > 0:
                self.face_sizes.append(face_size)
        if reid_emb is not None: self.reid_embs.append(reid_emb)

    def avg_face(self):
        if not self.face_embs: return None
        avg = np.mean(self.face_embs, axis=0)
        return normalize(avg)

    def avg_reid(self):
        if not self.reid_embs: return None
        avg = np.mean(self.reid_embs, axis=0)
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

# ----------------------
# Main loop
# ----------------------
cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0
tracklets = {}
next_tid = 1

# small helper to map face boxes per frame
face_boxes_frame = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    h, w = frame.shape[:2]

    # run detection every N frames
    if frame_idx % DETECT_EVERY_N_FRAMES == 0:
        # person detection
        p_results = yolo_person.predict(frame, imgsz=640, conf=0.45, classes=[0], verbose=False)
        person_boxes = []
        if len(p_results):
            for b in p_results[0].boxes:
                x1,y1,x2,y2 = b.xyxy[0].cpu().numpy().astype(int)
                x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w,x2),min(h,y2)
                person_boxes.append((x1,y1,x2,y2))

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

        # match boxes to tracklets (use person boxes as primary track regions)
        for box in person_boxes:
            best_tid, best_iouv = None, 0
            for tid, t in tracklets.items():
                val = iou(box, t.bboxes[-1])
                if val > best_iouv:
                    best_tid, best_iouv = tid, val

            x1,y1,x2,y2 = box
            crop_person = frame[y1:y2, x1:x2].copy()

            # ---- Face detection: Try multiple methods for best results
            face_emb = None
            face_size = 0

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

            # update or create tracklet
            current_tid = None
            if best_iouv > IOU_THRESHOLD:
                current_tid = best_tid
                t = tracklets[best_tid]
                t.update(box, frame_idx, face_emb, reid_emb, face_size)
            else:
                current_tid = next_tid; next_tid += 1
                t = Tracklet(current_tid, box, frame_idx)
                t.update(box, frame_idx, face_emb, reid_emb, face_size)
                tracklets[current_tid] = t
                
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

    # --------------------------
    # Verification logic (unchanged)
    # --------------------------
    for tid, t in list(tracklets.items()):

        if frame_idx - t.last_frame > TRACKLET_MAX_AGE:
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

            if t.verified:
                # start tracker (use legacy API for compatibility)
                x1,y1,x2,y2 = t.bboxes[-1]
                wbox, hbox = x2-x1, y2-y1
                try:
                    # Try legacy tracker API first (OpenCV 4.5+)
                    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
                        tracker = cv2.legacy.TrackerCSRT_create()
                    elif hasattr(cv2, 'TrackerCSRT_create'):
                        tracker = cv2.TrackerCSRT_create()
                    else:
                        # Fallback to KCF tracker if CSRT not available
                        tracker = cv2.TrackerKCF_create() if hasattr(cv2, 'TrackerKCF_create') else None
                    
                    if tracker is not None:
                        tracker.init(frame, (x1,y1,wbox,hbox))
                        t.tracker = tracker
                except Exception as e:
                    # Tracker initialization failed - continue without tracker
                    print(f"Warning: Tracker init failed for tracklet {tid}: {e}")
                    t.tracker = None

    # --------------------------
    # Visualization
    # --------------------------
    vis = frame.copy()
    for tid, t in tracklets.items():
        x1,y1,x2,y2 = t.bboxes[-1]
        color = (0,255,0) if t.verified else (0,0,255)
        cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
        cv2.putText(vis, f"ID:{tid}{' V' if t.verified else ''}",
                    (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Hybrid Face+ReID CPU Pipeline (YOLO-face integrated)", vis)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()