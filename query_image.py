import base64
import cv2
import numpy as np


def _build_video_filter(video_id):
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    values = []
    try:
        values.append(int(video_id))
    except Exception:
        pass

    video_id_str = str(video_id)
    if video_id_str not in values:
        values.append(video_id_str)

    if len(values) == 1:
        return Filter(
            must=[
                FieldCondition(
                    key="video_id",
                    match=MatchValue(value=values[0])
                )
            ]
        )

    return Filter(
        should=[
            FieldCondition(key="video_id", match=MatchValue(value=v))
            for v in values
        ]
    )


def extract_face_embedding(face_analyzer, image):
    if face_analyzer is None:
        raise Exception("InsightFace not initialized")

    faces = face_analyzer.get(image)
    if not faces or len(faces) == 0:
        return None, "No face detected in image"

    ref_face = faces[0]
    embedding = np.array(ref_face.embedding, dtype=np.float32)
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    return embedding, None


def search_by_face_embedding(client, embedding, video_id=None, top_k=10):
    if client is None:
        raise Exception("Qdrant client not initialized")

    search_filter = _build_video_filter(video_id)

    search_results = []
    try:
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


def handle_image_query(request, client, face_analyzer):
    video_id = request.form.get('video_id') or (request.json.get('video_id') if request.is_json else None)
    top_k = int(request.form.get('top_k', 10) if request.form else (request.json.get('top_k', 10) if request.is_json else 10))

    if not video_id:
        return {
            'success': False,
            'error': 'Missing required field: video_id'
        }, 400

    image = None

    if 'image' in request.files:
        file = request.files['image']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif request.is_json and 'image_base64' in request.json:
        base64_data = request.json['image_base64']
        if ',' in base64_data:
            base64_data = base64_data.split(',')[1]
        image_bytes = base64.b64decode(base64_data)
        file_bytes = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return {
            'success': False,
            'error': 'No valid image provided. Send as file upload or base64.'
        }, 400

    embedding, error = extract_face_embedding(face_analyzer, image)
    if error:
        return {
            'success': False,
            'error': error
        }, 400

    results = search_by_face_embedding(client, embedding, video_id, top_k)

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

    return {
        'success': True,
        'query': 'image',
        'videoId': video_id,
        'results': {
            'persons': persons,
            'objects': []
        },
        'summary': {
            'totalPersonsFound': len(persons),
            'totalObjectsFound': 0
        }
    }, 200
