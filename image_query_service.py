"""
Image Query Service
Flask API for post-processing image-based queries against Qdrant.
Uses InsightFace for face embedding extraction and queries person_tracks collection.
"""
import os
import sys
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import base64
import tempfile
from urllib.parse import urlparse

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


def extract_face_embedding(image):
    """Extract face embedding from an image using InsightFace."""
    if face_analyzer is None:
        raise Exception("InsightFace not initialized")
    
    faces = face_analyzer.get(image)
    if not faces or len(faces) == 0:
        return None, "No face detected in image"
    
    # Use the first (largest) face
    ref_face = faces[0]
    embedding = np.array(ref_face.embedding, dtype=np.float32)
    
    # Normalize embedding
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    
    return embedding, None


def search_by_face_embedding(embedding, video_id=None, top_k=10):
    """Search Qdrant person_tracks collection by face embedding."""
    if client is None:
        raise Exception("Qdrant client not initialized")
    
    # Build filter
    search_filter = None
    if video_id:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="video_id",
                    match=MatchValue(value=int(video_id))
                )
            ]
        )
    
    # Try different search methods for compatibility
    search_results = []
    try:
        # Try query_points first (newer API)
        results = client.query_points(
            collection_name="person_tracks",
            query=embedding.tolist(),
            using="face_vec",
            query_filter=search_filter,
            limit=top_k,
            with_payload=True
        )
        search_results = results.points
    except AttributeError:
        # Fallback to search method
        try:
            search_results = client.search(
                collection_name="person_tracks",
                query_vector=("face_vec", embedding.tolist()),
                query_filter=search_filter,
                limit=top_k,
                with_payload=True
            )
        except Exception:
            search_results = client.search(
                collection_name="person_tracks",
                query_vector=embedding.tolist(),
                query_filter=search_filter,
                limit=top_k,
                with_payload=True,
                search_params={"hnsw_ef": 128, "exact": False}
            )
    
    return search_results


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'qdrant': client is not None,
        'insightface': face_analyzer is not None
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
        video_id = request.form.get('video_id') or request.json.get('video_id') if request.is_json else None
        top_k = int(request.form.get('top_k', 10) if request.form else (request.json.get('top_k', 10) if request.is_json else 10))
        
        image = None
        
        # Check for file upload
        if 'image' in request.files:
            file = request.files['image']
            # Read image from file
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Check for base64 encoded image
        elif request.is_json and 'image_base64' in request.json:
            base64_data = request.json['image_base64']
            # Remove data URL prefix if present
            if ',' in base64_data:
                base64_data = base64_data.split(',')[1]
            image_bytes = base64.b64decode(base64_data)
            file_bytes = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                'success': False,
                'error': 'No valid image provided. Send as file upload or base64.'
            }), 400
        
        print(f"✓ Received image: {image.shape[1]}x{image.shape[0]}")
        
        # Extract face embedding
        embedding, error = extract_face_embedding(image)
        if error:
            return jsonify({
                'success': False,
                'error': error
            }), 400
        
        print(f"✓ Extracted face embedding: {len(embedding)}D")
        
        # Search Qdrant
        results = search_by_face_embedding(embedding, video_id, top_k)
        print(f"✓ Found {len(results)} matches")
        
        # Format results
        persons = []
        for idx, result in enumerate(results):
            payload = result.payload
            score = result.score
            
            persons.append({
                'id': str(result.id),
                'trackId': payload.get('track_id'),
                'personId': f"Person-{str(payload.get('track_id', idx + 1)).zfill(3)}",
                'score': float(score),
                'confidence': f"{(float(score) * 100):.1f}%",
                'timeOfAppearance': payload.get('start_time', 'N/A'),
                'endTime': payload.get('end_time', 'N/A'),
                'clothingColors': {
                    'upper': payload.get('upper_color', 'Unknown'),
                    'lower': payload.get('lower_color', 'Unknown')
                },
                'objectCarried': ', '.join(payload.get('object_carried', [])) if isinstance(payload.get('object_carried'), list) else payload.get('object_carried', 'None'),
                'numFrames': payload.get('num_frames', 0),
                'verified': payload.get('verified', False),
                'attributes': payload.get('attributes', {})
            })
        
        return jsonify({
            'success': True,
            'query': 'image',
            'videoId': video_id,
            'results': {
                'persons': persons,
                'objects': []  # Image queries only search person_tracks
            },
            'summary': {
                'totalPersonsFound': len(persons),
                'totalObjectsFound': 0
            }
        })
        
    except Exception as e:
        print(f"✗ Error in image query: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('IMAGE_QUERY_PORT', 5001))
    print(f"\n{'='*60}")
    print(f"🚀 Image Query Service starting on port {port}")
    print(f"{'='*60}\n")
    app.run(host='0.0.0.0', port=port, debug=True)
