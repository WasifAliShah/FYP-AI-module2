import cv2
import numpy as np

# --------------------------------------------------------
# PREPROCESSING PIPELINE (DENOISE + CONTRAST + GAMMA + SHARPEN)
# --------------------------------------------------------
def preprocess_frame(frame):
    # 1. Light denoising (very important for your video)
    denoised = cv2.fastNlMeansDenoisingColored(frame, None, 5, 5, 7, 21)

    # 2. CLAHE (local contrast enhancement)
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
        for i in np.arange(256)
    ]).astype("uint8")
    corrected = cv2.LUT(clahe_bgr, gamma_table)

    # 4. Mild sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(corrected, -1, kernel)

    return sharp


# --------------------------------------------------------
# TEST THE PREPROCESSING ON YOUR VIDEO
# --------------------------------------------------------
VIDEO_PATH = "new_video8.mp4"   # <- put your video here
cap = cv2.VideoCapture(VIDEO_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed = preprocess_frame(frame)

    # Show original and processed side-by-side
    combined = cv2.hconcat([
        cv2.resize(frame, (640, 360)),
        cv2.resize(processed, (640, 360))
    ])

    cv2.imshow("Original (Left) vs Preprocessed (Right)", combined)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
