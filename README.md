# Face Recognition Pipeline

A robust face recognition pipeline for video analysis that handles distant faces, compression artifacts, and lighting variations.

## Features

- **Multi-stage Detection**: YOLO + InsightFace for accurate face detection
- **Distant Face Recognition**: Aggressive upscaling and preprocessing for far-away faces
- **Compression Handling**: Denoising and image enhancement for compressed videos
- **Lighting Adaptation**: CLAHE and preprocessing for bright/dark conditions
- **Adaptive Thresholds**: Size-based matching thresholds for optimal accuracy

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd "FYP AI 2"
```

2. Create a virtual environment (recommended):
```bash
python -m venv insight_env
# On Windows:
insight_env\Scripts\activate
# On Linux/Mac:
source insight_env/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download model files (not included in repo due to size):
   - `yolov8m.pt` - YOLO person detector
   - `yolov8m-face.pt` - YOLO face detector
   - InsightFace models will be downloaded automatically on first run

## Usage

### Basic Usage

```bash
python pipeline.py
```

### Configuration

Edit `pipeline.py` to configure:

- `VIDEO_PATH`: Path to input video
- `REF_FACE_PATHS`: List of reference face images
- `FACE_STRICT`, `FACE_LOOSE`: Recognition thresholds
- `DETECT_EVERY_N_FRAMES`: Detection frequency

### Example

```python
VIDEO_PATH = "new_video10.mp4"
REF_FACE_PATHS = ["new_wasif1.jpg"]
```

## Requirements

- Python 3.8+
- PyTorch (CPU or CUDA)
- OpenCV
- Ultralytics YOLO
- InsightFace

See `requirements.txt` for full list.

## Model Files

Model files (*.pt, *.onnx) are not included in the repository due to size. You need to:

1. Download YOLO models from Ultralytics
2. InsightFace models download automatically on first run

## Notes

- For GPU support, install PyTorch with CUDA
- Real-ESRGAN is optional (commented in requirements.txt)
- The pipeline is optimized for CPU but works better with GPU

## License

[Add your license here]

