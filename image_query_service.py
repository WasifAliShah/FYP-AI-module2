"""
Unified Query Service
Flask API for post-processing image/text queries against Qdrant.
- Image queries use InsightFace embeddings against person_tracks
- Text queries use CLIP embeddings against person_tracks and object_tracks
"""
import os
import sys
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from urllib.parse import urlparse
from query_image import handle_image_query
from query_text import handle_text_query

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize Qdrant client
client = None
qdrant_url = os.environ.get("QDRANT_URL")
qdrant_host = os.environ.get("QDRANT_HOST")
qdrant_port = os.environ.get("QDRANT_PORT") or os.environ.get("QDRANT_GRPC_PORT")

try:
    from qdrant_client import QdrantClient
    
    if qdrant_url:
        parsed = urlparse(qdrant_url)
        if parsed.port == 6334:
            host = parsed.hostname or "localhost"
            port = parsed.port
            client = QdrantClient(host=host, port=port, prefer_grpc=True)
        else:
            client = QdrantClient(url=qdrant_url)
    elif qdrant_host and qdrant_port:
        client = QdrantClient(host=qdrant_host, port=int(qdrant_port), prefer_grpc=True)
    else:
        client = QdrantClient()
    
    client.get_collections()
    print(f"✓ Connected to Qdrant")
except Exception as e:
    print(f"✗ Failed to connect to Qdrant: {e}")

# Initialize InsightFace
face_analyzer = None
try:
    from insightface.app import FaceAnalysis
    face_analyzer = FaceAnalysis(allowed_modules=['detection', 'recognition'])
    face_analyzer.prepare(ctx_id=-1, det_size=(640, 640))
    print("✓ InsightFace initialized")
except Exception as e:
    print(f"✗ InsightFace not available: {e}")

# Initialize CLIP
clip_model = None
clip = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
try:
    import clip
    clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()
    print(f"✓ CLIP initialized on {DEVICE}")
except Exception as e:
    print(f"✗ CLIP not available: {e}")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'qdrant': client is not None,
        'insightface': face_analyzer is not None,
        'clip': clip_model is not None
    })


@app.route('/query/image', methods=['POST'])
def query_by_image():
    """
    Query Qdrant by face image.
    
    Accepts:
    - image: Base64 encoded image OR file upload
    - video_id: Optional video ID to filter results
    - top_k: Number of results to return (default: 10)
    
    Returns:
    - List of matching persons with scores and metadata
    """
    try:
        payload, status_code = handle_image_query(request, client, face_analyzer)
        return jsonify(payload), status_code
    except Exception as e:
        print(f"✗ Error in image query: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/query/text', methods=['POST'])
def query_by_text():
    """
    Query Qdrant semantically by text using CLIP embeddings.

    JSON body:
    - text: text prompt (required)
    - video_id: video id to scope search (required)
    - top_k: result count per type (optional, default: 10)
    """
    try:
        payload, status_code = handle_text_query(request, client, clip_model, clip, DEVICE)
        return jsonify(payload), status_code
    except Exception as e:
        print(f"✗ Error in text query: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('QUERY_SERVICE_PORT', os.environ.get('IMAGE_QUERY_PORT', 5001)))
    print(f"\n{'='*60}")
    print(f"🚀 Unified Query Service starting on port {port}")
    print(f"{'='*60}\n")
    app.run(host='0.0.0.0', port=port, debug=True)
