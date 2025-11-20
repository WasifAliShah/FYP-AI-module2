import cv2
import numpy as np

# ------------------------------------------------------------
# PREPROCESSING PIPELINE (DENOISE + CLAHE + GAMMA + SHARPEN)
# ------------------------------------------------------------
def preprocess_frame(frame):
    # 1. Light denoising
    denoised = cv2.fastNlMeansDenoisingColored(frame, None, 5, 5, 7, 21)

    # 2. CLAHE (Improves contrast)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    clahe_bgr = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    # 3. Gamma correction
    gamma = 1.15
    gamma_table = np.array([
        ((i / 255.0) ** (1/gamma)) * 255
        for i in range(256)
    ]).astype("uint8")
    corrected = cv2.LUT(clahe_bgr, gamma_table)

    # 4. Light sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(corrected, -1, kernel)

    return sharp


# ------------------------------------------------------------
# VIDEO STABILIZATION (optical flow)
# ------------------------------------------------------------
def stabilize_frame(prev_gray, curr_gray, curr_frame, transform_matrix):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)

    dx = np.mean(flow[..., 0])
    dy = np.mean(flow[..., 1])

    # Update transform matrix
    transform_matrix[0, 2] -= dx
    transform_matrix[1, 2] -= dy

    stabilized = cv2.warpAffine(curr_frame, transform_matrix[:2], 
                                (curr_frame.shape[1], curr_frame.shape[0]),
                                flags=cv2.INTER_LINEAR)

    return stabilized, transform_matrix


# ------------------------------------------------------------
# MAIN SCRIPT — READ, PROCESS, STABILIZE, SAVE
# ------------------------------------------------------------
VIDEO_PATH = "new_video8.mp4"
OUTPUT_PATH = "preprocessed_output.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))

prev_gray = None
transform_matrix = np.eye(3)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Convert to grayscale for stabilization
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is None:
        stabilized = frame.copy()
    else:
        stabilized, transform_matrix = stabilize_frame(prev_gray, gray, frame, transform_matrix)

    prev_gray = gray

    # Apply image enhancement
    processed = preprocess_frame(stabilized)

    # Write to output file
    out.write(processed)

    # Optional preview
    preview = cv2.resize(processed, (640, 360))
    cv2.imshow("Preprocessed Output", preview)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Saved:", OUTPUT_PATH)
