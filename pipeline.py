# used RealESRGAN and InsightFace libraries

# # """
# # ai_pipeline_insightface.py
# # Hybrid Face (InsightFace) + SuperResolution + ReID pipeline
# # """

# import time, math, os
# from collections import deque
# import numpy as np
# import cv2

# # Deep libraries
# import torch
# import torchvision.transforms as T
# from torchvision.models import resnet50

# # InsightFace
# try:
#     from insightface.app import FaceAnalysis
#     INSIGHTFACE_AVAILABLE = True
# except Exception as e:
#     print("insightface not available:", e)
#     INSIGHTFACE_AVAILABLE = False

# # Real-ESRGAN (optional)
# USE_SR = False
# # try:
# from realesrgan import RealESRGAN
# USE_SR = True
# # except Exception:
# #     USE_SR = False

# # TorchReID (optional)
# USE_TORCHREID = False
# try:
#     import torchreid
#     USE_TORCHREID = True
# except Exception:
#     USE_TORCHREID = False

# # YOLO person detector (ultralytics)
# from ultralytics import YOLO

# # ---------- CONFIG ----------
# VIDEO_PATH = "new_video10.mp4"
# REF_FACE_PATHS = ["new_wasif1.jpg"]    # face-only reference(s)
# YOLO_MODEL = "yolov8m.pt"              # person detector model
# DETECT_EVERY_N_FRAMES = 20
# MIN_FACE_WIDTH_PX = 20                 # below this SR helps but may still fail
# FACE_SIM_THRESHOLD_STRICT = 0.55
# FACE_SIM_THRESHOLD_LOOSE = 0.68
# REID_SIM_THRESHOLD = 0.42
# AGGREGATION_FRAMES = 5
# TRACKLET_MAX_AGE = 30
# IOU_TRACKER_IOU_THRESH = 0.4
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# # ----------------------------

# # ---------- load detectors / models ----------
# # YOLO person model
# yolo = YOLO(YOLO_MODEL)

# # InsightFace face analyzer (SCRFD + recognition)
# fa = None
# if INSIGHTFACE_AVAILABLE:
#     fa = FaceAnalysis(allowed_modules=['detection', 'landmark', 'recognition'])
#     # det_size: 640 is a good default for person/CCTV; adjust if you have very high-res frames
#     print("Preparing InsightFace (may download SCRFD/recognition models on first run)...")
#     fa.prepare(ctx_id=0 if DEVICE=="cuda" else -1, det_size=(640, 640))
#     print("InsightFace ready.")

# # Super-resolution init
# sr_model = None
# if USE_SR:
#     try:
#         # RealESRGAN scale 2 or 4 depending on model downloaded automatically by repo
#         sr = RealESRGAN(DEVICE, scale=2)
#         # If model weights not available it will try to download
#         sr.load_weights('RealESRGAN_x2plus.pth')
#         sr_model = sr
#         print("RealESRGAN ready.")
#     except Exception as e:
#         print("Real-ESRGAN init failed:", e)
#         sr_model = None

# # ReID appearance encoder
# def build_resnet_encoder():
#     model = resnet50(pretrained=True).eval().to(DEVICE)
#     transform = T.Compose([
#         T.ToPILImage(),
#         T.Resize((128,256)),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
#     ])
#     return model, transform

# if USE_TORCHREID:
#     # try to use OSNet x1_0 feature extractor (torchreid is flexible; we use a simple pretrained build)
#     try:
#         reid_model = torchreid.models.build_model(name='osnet_x1_0', num_classes=1000, pretrained=True)
#         reid_model = reid_model.to(DEVICE).eval()
#         def reid_encode(img):
#             # img: BGR numpy
#             x = T.ToPILImage()(img[:,:,::-1])
#             x = T.Resize((256,128))(x)
#             x = T.ToTensor()(x).unsqueeze(0).to(DEVICE)
#             with torch.no_grad():
#                 feat = reid_model(x).cpu().numpy().squeeze()
#             feat = feat / (np.linalg.norm(feat)+1e-8)
#             return feat.astype(np.float32)
#         print("Using torchreid OSNet for ReID.")
#     except Exception as e:
#         print("torchreid OSNet init failed:", e)
#         USE_TORCHREID = False

# if not USE_TORCHREID:
#     resnet_enc_model, resnet_transform = build_resnet_encoder()
#     def reid_encode(img):
#         try:
#             x = resnet_transform(img[:,:,::-1]).unsqueeze(0).to(DEVICE)
#             with torch.no_grad():
#                 feat = resnet_enc_model(x).cpu().numpy().squeeze()
#             feat = feat / (np.linalg.norm(feat)+1e-8)
#             return feat.astype(np.float32)
#         except Exception as e:
#             print("ResNet reid encode error:", e)
#             return None
#     print("Using ResNet50 fallback for appearance encoding.")

# # ---------- Build reference embeddings ----------
# ref_face_embeddings = []
# ref_reid_embeddings = []

# def normalize(v):
#     v = v.astype(np.float32)
#     return v / (np.linalg.norm(v)+1e-8)

# for p in REF_FACE_PATHS:
#     img = cv2.imread(p)
#     if img is None:
#         print("Warning: could not load reference image:", p)
#         continue
#     # Face embedding via InsightFace (if available) else use DeepFace fallback? We'll use InsightFace if present.
#     face_emb = None
#     if INSIGHTFACE_AVAILABLE:
#         try:
#             faces = fa.get(img)
#             if faces and len(faces)>0:
#                 face_emb = normalize(np.array(faces[0].embedding))
#         except Exception as e:
#             print("Ref face embed error:", e)
#     # ReID embedding from full image (user may provide headshot; still useful)
#     reid_emb = reid_encode(img)
#     if face_emb is not None:
#         ref_face_embeddings.append(face_emb)
#     if reid_emb is not None:
#         ref_reid_embeddings.append(reid_emb)

# if len(ref_face_embeddings)==0:
#     print("Warning: no reference face embeddings found. Face matching will be disabled until reference with face is provided.")
# if len(ref_reid_embeddings)==0:
#     print("Warning: no reference reid embeddings found. ReID matching will be disabled until reference provided.")

# # ---------- helpers ----------
# def cos_dist(a,b):
#     if a is None or b is None: return 1.0
#     return float(1.0 - np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))

# def is_face_match(emb, face_width_px):
#     if emb is None or len(ref_face_embeddings)==0: return False, None
#     # dynamic threshold based on face size
#     if face_width_px >= 100:
#         thr = FACE_SIM_THRESHOLD_STRICT
#     elif face_width_px >= 40:
#         thr = FACE_SIM_THRESHOLD_LOOSE
#     else:
#         # too small—let reid handle it
#         return False, None
#     best = min([cos_dist(emb, r) for r in ref_face_embeddings])
#     return best < thr, best

# def is_reid_match(emb):
#     if emb is None or len(ref_reid_embeddings)==0: return False, None
#     best = min([cos_dist(emb, r) for r in ref_reid_embeddings])
#     return best < REID_SIM_THRESHOLD, best

# # IoU
# def iou(a,b):
#     xA = max(a[0], b[0]); yA = max(a[1], b[1])
#     xB = min(a[2], b[2]); yB = min(a[3], b[3])
#     interW = max(0, xB-xA); interH = max(0, yB-yA)
#     inter = interW*interH
#     areaA = (a[2]-a[0])*(a[3]-a[1]); areaB = (b[2]-b[0])*(b[3]-b[1])
#     if areaA+areaB-inter == 0: return 0
#     return inter/float(areaA+areaB-inter)

# # Tracklet class (aggregate embeddings across frames)
# class Tracklet:
#     def __init__(self, tid, bbox, frame_idx):
#         self.id = tid
#         self.bboxes = deque(maxlen=AGGREGATION_FRAMES)
#         self.bboxes.append(bbox)
#         self.last_frame = frame_idx
#         self.face_embs = []
#         self.reid_embs = []
#         self.verified = False
#         self.verified_time = None
#         self.tracker = None
#     def update(self, bbox, frame_idx, face_emb=None, reid_emb=None):
#         self.bboxes.append(bbox)
#         self.last_frame = frame_idx
#         if face_emb is not None:
#             self.face_embs.append(face_emb)
#         if reid_emb is not None:
#             self.reid_embs.append(reid_emb)
#     def aggregated_face(self):
#         if len(self.face_embs)==0: return None
#         arr = np.vstack(self.face_embs)
#         avg = np.mean(arr, axis=0)
#         return normalize(avg)
#     def aggregated_reid(self):
#         if len(self.reid_embs)==0: return None
#         arr = np.vstack(self.reid_embs)
#         avg = np.mean(arr, axis=0)
#         return normalize(avg)

# # ---------- main loop ----------
# cap = cv2.VideoCapture(VIDEO_PATH)
# frame_idx = 0
# tracklets = {}
# next_tid = 1
# start_time = time.time()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame_idx += 1
#     h, w = frame.shape[:2]

#     # update trackers for verified tracklets
#     for tid, t in list(tracklets.items()):
#         if t.verified and t.tracker is not None:
#             ok, bb = t.tracker.update(frame)
#             if ok:
#                 x,y,ww,hh = map(int, bb)
#                 t.bboxes.append((x,y,x+ww,y+hh))
#                 t.last_frame = frame_idx
#             else:
#                 t.tracker = None
#                 t.verified = False

#     # detection step
#     if frame_idx % DETECT_EVERY_N_FRAMES == 0:
#         results = yolo.predict(frame, imgsz=640, conf=0.35, classes=[0], verbose=False)
#         boxes = []
#         if len(results)>0:
#             for box in results[0].boxes:
#                 xy = box.xyxy[0].cpu().numpy().astype(int)
#                 x1,y1,x2,y2 = int(xy[0]),int(xy[1]),int(xy[2]),int(xy[3])
#                 x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w,x2),min(h,y2)
#                 boxes.append((x1,y1,x2,y2))

#         # assign detections to tracklets by IoU
#         assigned = set()
#         for bbox in boxes:
#             best_tid, best_iou = None, 0
#             for tid, t in tracklets.items():
#                 last_box = t.bboxes[-1]
#                 val = iou(bbox, last_box)
#                 if val > best_iou:
#                     best_iou, best_tid = val, tid
#             if best_iou is not None and best_iou > IOU_TRACKER_IOU_THRESH:
#                 # update tracklet
#                 x1,y1,x2,y2 = bbox
#                 crop = frame[y1:y2, x1:x2]
#                 # Face detection + embedding using InsightFace (prefer)
#                 face_emb = None
#                 face_w = 0
#                 if INSIGHTFACE_AVAILABLE:
#                     try:
#                         # run face detection/recognition ON the person crop - faster than whole frame
#                         faces = fa.get(crop)
#                         if faces and len(faces)>0:
#                             face = faces[0]
#                             face_w = int(face.bbox[2]-face.bbox[0])
#                             # if face very small and SR available - upscale aligned face before embedding
#                             face_img_aligned = face.aligned  # insightface returns aligned crop if available
#                             if face_img_aligned is None:
#                                 face_img_aligned = crop
#                             # optional SR
#                             if sr_model is not None and face_img_aligned.shape[1] < 64:
#                                 try:
#                                     face_sr = sr_model.enhance(face_img_aligned)
#                                     face_emb = normalize(np.array(face.embedding)) if face.embedding is not None else None
#                                     # if insightface embedding was computed on original, recompute on sr if possible:
#                                     # We'll use insightface to get embedding on sr by feeding sr to fa.get (slower).
#                                     # For speed, if insightface already provided embedding we use it; else recompute below:
#                                 except Exception:
#                                     face_emb = normalize(np.array(face.embedding)) if face.embedding is not None else None
#                             else:
#                                 face_emb = normalize(np.array(face.embedding)) if face.embedding is not None else None
#                     except Exception as e:
#                         face_emb = None
#                 else:
#                     # fallback: attempt simple OpenCV face cascade or skip face
#                     face_emb = None

#                 # reid on full person crop
#                 reid_emb = None
#                 try:
#                     reid_emb = reid_encode(crop)
#                 except Exception:
#                     reid_emb = None

#                 tracklets[best_tid].update(bbox, frame_idx, face_emb, reid_emb)
#             else:
#                 # create new tracklet
#                 tid = next_tid; next_tid += 1
#                 t = Tracklet(tid, bbox, frame_idx)
#                 x1,y1,x2,y2 = bbox
#                 crop = frame[y1:y2, x1:x2]
#                 # face embedding
#                 face_emb = None
#                 if INSIGHTFACE_AVAILABLE:
#                     try:
#                         faces = fa.get(crop)
#                         if faces and len(faces)>0:
#                             face = faces[0]
#                             face_emb = normalize(np.array(face.embedding)) if face.embedding is not None else None
#                     except Exception:
#                         face_emb = None
#                 reid_emb = None
#                 try:
#                     reid_emb = reid_encode(crop)
#                 except Exception:
#                     reid_emb = None
#                 t.update(bbox, frame_idx, face_emb, reid_emb)
#                 tracklets[tid] = t

#     # attempt verification for tracklets that have enough frames
#     for tid, t in list(tracklets.items()):
#     # remove old tracklets
#         if frame_idx - t.last_frame > TRACKLET_MAX_AGE:
#             del tracklets[tid]
#             continue

#         # only attempt if not verified and enough embeddings collected
#         if not t.verified and (len(t.face_embs)+len(t.reid_embs)) >= AGGREGATION_FRAMES:
#             face_avg = t.aggregated_face()
#             reid_avg = t.aggregated_reid()
            
#             # estimate face width from last bbox
#             last_bbox = t.bboxes[-1]
#             face_width_px = (last_bbox[2]-last_bbox[0])//4  # rough upper-quarter assumption
            
#             # check face match first (if available)
#             face_ok, face_score = is_face_match(face_avg, face_width_px)
            
#             # check reid match only if face is too small or not detected
#             use_reid = False
#             if face_avg is None or face_width_px < 40:
#                 use_reid = True
            
#             reid_ok, reid_score = False, None
#             if use_reid and reid_avg is not None:
#                 # stricter threshold for ResNet50 on CPU
#                 REID_THRESHOLD_CPU = 0.35
#                 best = min([cos_dist(reid_avg, r) for r in ref_reid_embeddings])
#                 reid_ok = best < REID_THRESHOLD_CPU
#                 reid_score = best

#             # final verification logic
#             if face_ok or reid_ok:
#                 t.verified = True
#                 t.verified_time = frame_idx
#                 print(f"[VERIFIED] Tracklet {tid} at frame {frame_idx} "
#                     f"face_score={face_score} reid_score={reid_score}")
                
#                 # init CSRT tracker for continuity
#                 try:
#                     x1, y1, x2, y2 = t.bboxes[-1]
#                     w_box, h_box = x2-x1, y2-y1
#                     tracker = cv2.TrackerCSRT_create()
#                     tracker.init(frame, (x1, y1, w_box, h_box))
#                     t.tracker = tracker
#                 except Exception as e:
#                     print("Tracker init error:", e)

#     # visualization
#     vis = frame.copy()
#     for tid, t in tracklets.items():
#         x1,y1,x2,y2 = map(int, t.bboxes[-1])
#         color = (0,255,0) if t.verified else (0,0,255)
#         cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
#         label = f"ID:{tid}" + (" V" if t.verified else "")
#         cv2.putText(vis, label, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     cv2.imshow("Hybrid InsightFace+ReID", vis)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# end_time = time.time()
# print("Done. elapsed:", end_time-start_time)
# cap.release()
# cv2.destroyAllWindows()


"""
ai_pipeline_insightface_deepsort.py
Hybrid Face (InsightFace) + SuperResolution + ReID pipeline
Integrated with DeepSORT for robust tracking
"""

# import time, os
# from collections import deque
# from deep_sort_realtime.deep_sort.detection import Detection
# import numpy as np
# import cv2
# import torch
# import torchvision.transforms as T
# from torchvision.models import resnet50

# # InsightFace
# try:
#     from insightface.app import FaceAnalysis
#     INSIGHTFACE_AVAILABLE = True
# except Exception as e:
#     print("InsightFace not available:", e)
#     INSIGHTFACE_AVAILABLE = False

# # Real-ESRGAN (optional)
# USE_SR = False
# try:
#     from realesrgan import RealESRGAN
#     USE_SR = True
# except Exception:
#     USE_SR = False

# # YOLO person detector (ultralytics)
# from ultralytics import YOLO

# # DeepSORT
# from deep_sort_realtime.deepsort_tracker import DeepSort

# # ---------- CONFIG ----------
# VIDEO_PATH = "new_video9.mp4"
# REF_FACE_PATHS = ["new_wasif1.jpg"]
# YOLO_MODEL = "yolov8m.pt"
# DETECT_EVERY_N_FRAMES = 20
# MIN_FACE_WIDTH_PX = 20
# FACE_SIM_THRESHOLD_STRICT = 0.55
# FACE_SIM_THRESHOLD_LOOSE = 0.68
# REID_SIM_THRESHOLD = 0.42
# AGGREGATION_FRAMES = 5
# TRACKLET_MAX_AGE = 30
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# # ----------------------------

# # ---------- load detectors / models ----------
# yolo = YOLO(YOLO_MODEL)

# # InsightFace
# fa = None
# if INSIGHTFACE_AVAILABLE:
#     fa = FaceAnalysis(allowed_modules=['detection', 'landmark', 'recognition'])
#     print("Preparing InsightFace (may download SCRFD/recognition models)...")
#     fa.prepare(ctx_id=0 if DEVICE=="cuda" else -1, det_size=(640, 640))
#     print("InsightFace ready.")

# # Super-resolution
# sr_model = None
# if USE_SR:
#     try:
#         sr = RealESRGAN(DEVICE, scale=2)
#         sr.load_weights('RealESRGAN_x2plus.pth')
#         sr_model = sr
#         print("RealESRGAN ready.")
#     except Exception as e:
#         print("Real-ESRGAN init failed:", e)

# # ReID encoder
# def build_resnet_encoder():
#     model = resnet50(pretrained=True).eval().to(DEVICE)
#     transform = T.Compose([
#         T.ToPILImage(),
#         T.Resize((128,256)),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
#     ])
#     return model, transform

# resnet_enc_model, resnet_transform = build_resnet_encoder()
# def reid_encode(img):
#     try:
#         x = resnet_transform(img[:,:,::-1]).unsqueeze(0).to(DEVICE)
#         with torch.no_grad():
#             feat = resnet_enc_model(x).cpu().numpy().squeeze()
#         feat = feat / (np.linalg.norm(feat)+1e-8)
#         return feat.astype(np.float32)
#     except Exception as e:
#         print("ReID encode error:", e)
#         return None

# # ---------- Build reference embeddings ----------
# ref_face_embeddings = []
# ref_reid_embeddings = []

# def normalize(v):
#     v = v.astype(np.float32)
#     return v / (np.linalg.norm(v)+1e-8)

# for p in REF_FACE_PATHS:
#     img = cv2.imread(p)
#     if img is None:
#         print("Warning: could not load reference image:", p)
#         continue
#     face_emb = None
#     if INSIGHTFACE_AVAILABLE:
#         try:
#             faces = fa.get(img)
#             if faces and len(faces)>0:
#                 face_emb = normalize(np.array(faces[0].embedding))
#         except Exception as e:
#             print("Ref face embed error:", e)
#     reid_emb = reid_encode(img)
#     if face_emb is not None:
#         ref_face_embeddings.append(face_emb)
#     if reid_emb is not None:
#         ref_reid_embeddings.append(reid_emb)

# if len(ref_face_embeddings)==0:
#     print("Warning: no reference face embeddings found.")
# if len(ref_reid_embeddings)==0:
#     print("Warning: no reference reid embeddings found.")

# # ---------- helpers ----------
# def cos_dist(a,b):
#     if a is None or b is None: return 1.0
#     return float(1.0 - np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))

# def is_face_match(emb, face_width_px):
#     if emb is None or len(ref_face_embeddings)==0: return False, None
#     if face_width_px >= 100:
#         thr = FACE_SIM_THRESHOLD_STRICT
#     elif face_width_px >= 40:
#         thr = FACE_SIM_THRESHOLD_LOOSE
#     else:
#         return False, None
#     best = min([cos_dist(emb, r) for r in ref_face_embeddings])
#     return best < thr, best

# # ---------- DeepSORT tracker ----------
# deepsort_tracker = DeepSort(max_age=TRACKLET_MAX_AGE,
#                             n_init=AGGREGATION_FRAMES,
#                             max_cosine_distance=REID_SIM_THRESHOLD,
#                             embedder="mobilenet",
#                             half=False)

# # ---------- main loop ----------
# cap = cv2.VideoCapture(VIDEO_PATH)
# frame_idx = 0
# start_time = time.time()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame_idx += 1
#     h, w = frame.shape[:2]

#     # Detection step
#     if frame_idx % DETECT_EVERY_N_FRAMES == 0:
#         results = yolo.predict(frame, imgsz=640, conf=0.35, classes=[0], verbose=False)
#         detections = []

#         if len(results) > 0 and len(results[0].boxes) > 0:
#             for box in results[0].boxes:
#                 xy = box.xyxy[0].cpu().numpy()
#                 x1, y1, x2, y2 = map(int, xy)
#                 x1, y1 = max(0, x1), max(0, y1)
#                 x2, y2 = min(w, x2), min(h, y2)
#                 bbox = [x1, y1, x2, y2]

#                 # ReID feature
#                 crop = frame[y1:y2, x1:x2]
#                 feature = reid_encode(crop)

#                 # Handle missing features by creating a zero-vector
#                 if feature is None:
#                     feature = np.zeros(2048, dtype=np.float32)  # ResNet50 output size

#                 # Confidence
#                 conf = float(box.conf[0])

#                 detections.append([bbox, conf, feature])

#         # Update DeepSORT tracker
#         tracks = deepsort_tracker.update_tracks(detections, frame=frame)  # no extra args needed

#         # Face verification + visualization
#         for track in tracks:
#             if not track.is_confirmed():
#                 continue
#             tid = track.track_id
#             x1, y1, x2, y2 = map(int, track.to_ltrb())
#             crop = frame[y1:y2, x1:x2]

#             # Face embedding
#             face_emb = None
#             if INSIGHTFACE_AVAILABLE:
#                 try:
#                     faces = fa.get(crop)
#                     if faces and len(faces) > 0:
#                         face_emb = normalize(np.array(faces[0].embedding))
#                 except:
#                     face_emb = None

#             face_ok, face_score = is_face_match(face_emb, (x2-x1)//4)
#             color = (0,255,0) if face_ok else (0,0,255)
#             label = f"ID:{tid}" + (" V" if face_ok else "")
#             if face_ok:
#                 print(f"[VERIFIED] Track {tid} frame {frame_idx} face_score={face_score}")

#             # Draw bbox
#             cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(frame, label, (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     cv2.imshow("Hybrid InsightFace + DeepSORT", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# end_time = time.time()
# print("Done. elapsed:", end_time-start_time)
# cap.release()
# cv2.destroyAllWindows()


# Uses Bicubuc Upscaling instead of RealESRGAN for super-resolution

"""
CPU-Optimized Hybrid InsightFace + Bicubic Upscale + ReID Pipeline
"""

# import time
# from collections import deque
# import numpy as np
# import cv2
# import torch
# import torchvision.transforms as T
# from torchvision.models import resnet50
# from ultralytics import YOLO

# # InsightFace
# try:
#     from insightface.app import FaceAnalysis
#     INSIGHTFACE_AVAILABLE = True
# except:
#     print("InsightFace not available!")
#     INSIGHTFACE_AVAILABLE = False


# # ----------------------
# # CONFIG (CPU OPTIMIZED)
# # ----------------------
# VIDEO_PATH = "new_video9.mp4"
# REF_FACE_PATHS = ["new_wasif1.jpg"]

# YOLO_MODEL = "yolov8m.pt"
# DETECT_EVERY_N_FRAMES = 15

# FACE_STRICT = 0.55
# FACE_LOOSE  = 0.68
# REID_THRESHOLD_CPU = 0.35

# AGGREGATION_FRAMES = 10
# TRACKLET_MAX_AGE = 30
# IOU_THRESHOLD = 0.40

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using device:", DEVICE)


# # ----------------------
# # LOAD MODELS
# # ----------------------
# yolo = YOLO(YOLO_MODEL)

# # InsightFace (larger input size for better accuracy on small faces)
# fa = None
# if INSIGHTFACE_AVAILABLE:
#     fa = FaceAnalysis(allowed_modules=['detection', 'landmark', 'recognition'])
#     print("Preparing InsightFace...")
#     fa.prepare(ctx_id=-1, det_size=(1024, 1024))   # <-- made bigger for better CPU detection
#     print("InsightFace ready.")


# # ReID fallback encoder (ResNet50)
# def build_resnet_encoder():
#     model = resnet50(pretrained=True).eval().to(DEVICE)
#     transform = T.Compose([
#         T.ToPILImage(),
#         T.Resize((128, 256)),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
#     ])
#     return model, transform

# reid_model, reid_tf = build_resnet_encoder()

# def reid_encode(img):
#     try:
#         x = reid_tf(img[:,:,::-1]).unsqueeze(0).to(DEVICE)
#         with torch.no_grad():
#             feat = reid_model(x).cpu().numpy().squeeze()
#         feat = feat / (np.linalg.norm(feat)+1e-8)
#         return feat.astype(np.float32)
#     except:
#         return None


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

#     # face embedding
#     face_emb = None
#     if fa:
#         faces = fa.get(img)
#         if faces:
#             face_emb = normalize(np.array(faces[0].embedding))
#             ref_face_embs.append(face_emb)

#     # reid embedding
#     reid_emb = reid_encode(img)
#     if reid_emb is not None:
#         ref_reid_embs.append(reid_emb)


# # ----------------------
# # Helper functions
# # ----------------------
# def cos_dist(a, b):
#     if a is None or b is None: return 1.0
#     return 1.0 - np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8)

# def upscale_bicubic(img, factor=2):
#     return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

# def is_face_match(emb, face_w):
#     if emb is None or len(ref_face_embs)==0: 
#         return False, None

#     thr = FACE_STRICT if face_w >= 100 else FACE_LOOSE
#     best = min([cos_dist(emb, r) for r in ref_face_embs])
#     return best < thr, best

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


# # ----------------------
# # Tracklet class
# # ----------------------
# class Tracklet:
#     def __init__(self, tid, bbox, frame_idx):
#         self.id = tid
#         self.bboxes = deque(maxlen=AGGREGATION_FRAMES)
#         self.bboxes.append(bbox)
#         self.last_frame = frame_idx
#         self.face_embs = []
#         self.reid_embs = []
#         self.verified = False
#         self.tracker = None

#     def update(self, bbox, idx, face_emb=None, reid_emb=None):
#         self.bboxes.append(bbox)
#         self.last_frame = idx
#         if face_emb is not None: self.face_embs.append(face_emb)
#         if reid_emb is not None: self.reid_embs.append(reid_emb)

#     def avg_face(self):
#         if not self.face_embs: return None
#         avg = np.mean(self.face_embs, axis=0)
#         return normalize(avg)

#     def avg_reid(self):
#         if not self.reid_embs: return None
#         avg = np.mean(self.reid_embs, axis=0)
#         return normalize(avg)


# # ----------------------
# # Main loop
# # ----------------------
# cap = cv2.VideoCapture(VIDEO_PATH)
# frame_idx = 0
# tracklets = {}
# next_tid = 1

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame_idx += 1
#     h, w = frame.shape[:2]

#     # --------------------------
#     # Detection every N frames
#     # --------------------------
#     if frame_idx % DETECT_EVERY_N_FRAMES == 0:
#         results = yolo.predict(frame, imgsz=1280, conf=0.5, classes=[0], verbose=False)  # default 640
#         boxes = []

#         if len(results):
#             for b in results[0].boxes:
#                 x1,y1,x2,y2 = b.xyxy[0].cpu().numpy().astype(int)
#                 x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w,x2),min(h,y2)
#                 boxes.append((x1,y1,x2,y2))

#         # match boxes to tracklets
#         for box in boxes:
#             best_tid, best_iouv = None, 0
#             for tid, t in tracklets.items():
#                 val = iou(box, t.bboxes[-1])
#                 if val > best_iouv:
#                     best_tid, best_iouv = tid, val

#             x1,y1,x2,y2 = box
#             crop = frame[y1:y2, x1:x2]

#             # ---- FACE DETECTION + EMBEDDING (CPU optimized)
#             face_emb = None
#             face_size = 0

#             if fa:
#                 faces = fa.get(crop)
#                 if faces:
#                     f = faces[0]
#                     face_size = int(f.bbox[2]-f.bbox[0])

#                     # upscale small faces (BICUBIC FAST)
#                     if face_size < 40:
#                         crop_up = upscale_bicubic(crop, factor=2)
#                         faces_up = fa.get(crop_up)
#                         if faces_up:
#                             face_emb = normalize(np.array(faces_up[0].embedding))
#                     else:
#                         face_emb = normalize(np.array(f.embedding))

#             # ---- ReID embedding
#             reid_emb = reid_encode(crop)

#             # update or create tracklet
#             if best_iouv > IOU_THRESHOLD:
#                 t = tracklets[best_tid]
#                 t.update(box, frame_idx, face_emb, reid_emb)
#             else:
#                 tid = next_tid; next_tid += 1
#                 t = Tracklet(tid, box, frame_idx)
#                 t.update(box, frame_idx, face_emb, reid_emb)
#                 tracklets[tid] = t

#     # --------------------------
#     # Verification logic
#     # --------------------------
#     for tid, t in list(tracklets.items()):

#         if frame_idx - t.last_frame > TRACKLET_MAX_AGE:
#             del tracklets[tid]
#             continue

#         if not t.verified and (len(t.face_embs)+len(t.reid_embs)) >= AGGREGATION_FRAMES:

#             face_avg = t.avg_face()
#             reid_avg = t.avg_reid()

#             last = t.bboxes[-1]
#             face_width = (last[2]-last[0]) // 4

#             # try face first
#             face_ok, face_score = is_face_match(face_avg, face_width)

#             # if no face, use ReID
#             reid_ok, reid_score = False, None
#             if not face_ok:
#                 reid_ok, reid_score = is_reid_match(reid_avg)

#             if face_ok or reid_ok:
#                 t.verified = True
#                 print(f"[VERIFIED] Tracklet {tid}  face={face_score}  reid={reid_score}")

#                 # start tracker
#                 x1,y1,x2,y2 = t.bboxes[-1]
#                 wbox, hbox = x2-x1, y2-y1
#                 tracker = cv2.TrackerCSRT_create()
#                 tracker.init(frame, (x1,y1,wbox,hbox))
#                 t.tracker = tracker

#     # --------------------------
#     # Visualization
#     # --------------------------
#     vis = frame.copy()
#     for tid, t in tracklets.items():
#         x1,y1,x2,y2 = t.bboxes[-1]
#         color = (0,255,0) if t.verified else (0,0,255)
#         cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
#         cv2.putText(vis, f"ID:{tid}{' V' if t.verified else ''}",
#                     (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     cv2.imshow("Hybrid Face+ReID CPU Pipeline", vis)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()



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
VIDEO_PATH = "new_video10.mp4"
REF_FACE_PATHS = ["new_wasif1.jpg"]

YOLO_PERSON_MODEL = "yolov8m.pt"        # your person model
YOLO_FACE_MODEL = "yolov8m-face.pt"     # recommended: yolov8n-face or yolov8m-face
DETECT_EVERY_N_FRAMES = 15

# thresholds (tune as needed)
# Face recognition thresholds - LOWER scores = better matches (cosine distance)
# Typical good matches: < 0.3 (excellent), 0.3-0.5 (good), 0.5-0.65 (acceptable for small faces)
# Scores > 0.70 are typically NOT matches (different people)
FACE_STRICT = 0.50  # For large, clear faces (>= 100px)
FACE_LOOSE  = 0.60  # For medium faces (60-100px)
FACE_SMALL_MAX = 0.75  # Maximum threshold for very small faces - allows recognition of distant people (increased for compression/lighting issues)
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

def preprocess_face_image(img):
    """Preprocess face image to handle compression, lighting, and quality issues.
    Applies denoising, histogram equalization, and sharpening."""
    if img is None or img.size == 0:
        return img
    
    # Convert to LAB color space for better lighting handling
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    # This handles lighting issues (bright/dark) better than regular histogram eq
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge back
    lab = cv2.merge([l, a, b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Denoise to handle compression artifacts (use fastNlMeansDenoisingColored for color images)
    # This helps with compressed video artifacts
    if img.shape[0] > 20 and img.shape[1] > 20:  # Only if image is large enough
        img = cv2.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
    
    return img

def upscale_bicubic(img, factor=2):
    """Upscale image using bicubic interpolation."""
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

def sr_enhance(img, factor=2, preprocess=True):
    """Use Real-ESRGAN if available, otherwise bicubic upscale.
    For very small faces, use larger upscale factor.
    Preprocessing helps with compression and lighting issues."""
    # Preprocess first to handle compression/lighting
    if preprocess:
        img = preprocess_face_image(img)
    
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
    # Good matches: < 0.3 (excellent), 0.3-0.5 (good), 0.5-0.75 (acceptable for small/distant faces with compression)
    # Scores > 0.75 are typically NOT matches (different people)
    
    # For small/distant faces with compression/lighting issues, we need to be VERY lenient because:
    # 1. Embeddings from small faces are inherently less accurate
    # 2. Compression artifacts degrade embedding quality
    # 3. Lighting issues (bright/dark) affect recognition
    # 4. We need to balance false negatives vs false positives
    if face_w >= 100:
        # Large clear faces - can use strict threshold
        thr = FACE_STRICT
    elif face_w >= 60:
        # Medium faces - slightly more lenient but still strict
        thr = FACE_LOOSE
    elif face_w >= 40:
        # Small-medium faces (40-60px) - more lenient for distant faces
        # This is the critical range for distant recognition with compression
        thr = min(FACE_LOOSE + 0.12, FACE_SMALL_MAX)  # 0.72 max (increased for compression)
    elif face_w >= 25:
        # Small faces (25-40px) - even more lenient for distant/compressed faces
        thr = min(FACE_LOOSE + 0.15, FACE_SMALL_MAX)  # 0.75 max (increased)
    else:
        # Very small faces (< 25px) - most lenient for very distant faces
        # These are very challenging with compression, so we allow higher threshold
        thr = min(FACE_LOOSE + 0.18, FACE_SMALL_MAX)  # 0.78 max (increased)
    
    best = min([cos_dist(emb, r) for r in ref_face_embs])
    is_match = best < thr
    
    # Safety check: reject very high scores (more lenient for small/distant faces with compression)
    # Increased limits to handle compression and lighting issues
    if face_w < 25:
        max_allowed = 0.78  # Very lenient for tiny faces
    elif face_w < 40:
        max_allowed = 0.75  # Lenient for small faces
    elif face_w < 60:
        max_allowed = 0.72  # More lenient for medium-small faces (40-60px range)
    else:
        max_allowed = 0.65  # Standard for larger faces
    
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
                # This improves recognition accuracy for distant faces with compression
                if face_size < 60:
                    # Use larger upscale factor for very small faces (more aggressive for distant faces)
                    # Preprocessing helps with compression and lighting
                    upscale_factor = 6 if face_size < 25 else (5 if face_size < 35 else (4 if face_size < 45 else 3))
                    face_crop_up = sr_enhance(face_crop, factor=upscale_factor, preprocess=True)
                    
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
                    # Use larger upscale factor for very small faces (with preprocessing)
                    upscale_factor = 6 if face_size < 25 else (5 if face_size < 35 else (4 if face_size < 50 else 3))
                    face_crop_up = sr_enhance(face_crop, factor=upscale_factor, preprocess=True)
                    
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
                        # This improves recognition accuracy for distant faces with compression
                        if face_size < 60:
                            # Extract face region and upscale it for better embedding quality
                            if hasattr(f, 'bbox') and f.bbox is not None:
                                bx1 = max(0, int(f.bbox[0])); by1 = max(0, int(f.bbox[1]))
                                bx2 = min(crop_person.shape[1], int(f.bbox[2])); by2 = min(crop_person.shape[0], int(f.bbox[3]))
                                if bx2 > bx1 and by2 > by1:
                                    face_region = crop_person[by1:by2, bx1:bx2].copy()
                                    if face_region.size > 0:
                                        # Use larger upscale factor for very small faces (more aggressive)
                                        # Preprocessing helps with compression and lighting issues
                                        upscale_factor = 6 if face_size < 25 else (5 if face_size < 35 else (4 if face_size < 45 else 3))
                                        face_region_up = sr_enhance(face_region, factor=upscale_factor, preprocess=True)
                                        
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
                                        # Use larger upscale factor for very small faces (more aggressive with preprocessing)
                                        upscale_factor = 6 if face_size < 25 else (5 if face_size < 35 else (4 if face_size < 50 else 3))
                                        face_region_up = sr_enhance(face_region, factor=upscale_factor, preprocess=True)
                                        
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
                    upscale_factor = 6 if face_size < 25 else (5 if face_size < 35 else (4 if face_size < 50 else 3))
                    upscale_info = f" (upscaled {upscale_factor}x + preprocessed for better quality)"
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
