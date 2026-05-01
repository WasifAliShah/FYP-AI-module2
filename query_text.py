import numpy as np
import torch


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


def extract_text_embedding(clip_model, clip_module, device, text_prompt):
    if clip_model is None or clip_module is None:
        raise Exception("CLIP not initialized")

    with torch.no_grad():
        text_tokens = clip_module.tokenize([text_prompt]).to(device)
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
        text_embedding = text_features.squeeze().cpu().numpy().astype(np.float32)

    if len(text_embedding) < 512:
        text_embedding = np.concatenate([text_embedding, np.zeros(512 - len(text_embedding), dtype=np.float32)])
    else:
        text_embedding = text_embedding[:512]

    return text_embedding


def _semantic_scroll(client, collection_name, text_embedding, vector_keys, video_id, top_k=10, limit=1000):
    if client is None:
        raise Exception("Qdrant client not initialized")

    scroll_filter = _build_video_filter(video_id)
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=limit,
        with_payload=True,
        with_vectors=True,
        scroll_filter=scroll_filter
    )

    if not points:
        return []

    results = []
    for point in points:
        if not hasattr(point, "vector") or point.vector is None:
            continue

        vec = None
        if isinstance(point.vector, dict):
            for key in vector_keys:
                if point.vector.get(key) is not None:
                    vec = point.vector.get(key)
                    break
        else:
            vec = point.vector

        if vec is None:
            continue

        vec_np = np.array(vec, dtype=np.float32)
        if vec_np.size == 0 or vec_np.shape[0] != text_embedding.shape[0]:
            continue

        sim = np.dot(text_embedding, vec_np) / (np.linalg.norm(text_embedding) * np.linalg.norm(vec_np) + 1e-8)
        results.append({
            "id": point.id,
            "score": float(sim),
            "payload": point.payload if hasattr(point, "payload") else {}
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


def _apply_person_query_boosts(text_query, payload, base_score):
    boosted_score = base_score
    query_lower = text_query.lower()

    carried_objs = payload.get("object_carried", [])
    verified = payload.get("verified", False)

    if any(keyword in query_lower for keyword in ["phone", "mobile", "cell"]):
        if any("phone" in str(obj).lower() for obj in carried_objs):
            boosted_score += 0.15
    elif any(keyword in query_lower for keyword in ["laptop", "computer"]):
        if any("laptop" in str(obj).lower() for obj in carried_objs):
            boosted_score += 0.15
    elif any(keyword in query_lower for keyword in ["bag", "backpack", "handbag"]):
        if any(keyword in str(obj).lower() for obj in carried_objs for keyword in ["bag", "backpack", "handbag"]):
            boosted_score += 0.15
    elif any(keyword in query_lower for keyword in ["holding", "carrying", "with"]):
        if carried_objs:
            boosted_score += 0.05

    color_keywords = ["red", "blue", "green", "yellow", "black", "white", "gray", "grey", "brown", "pink", "purple", "orange", "cyan", "magenta", "navy", "beige", "tan"]
    upper_color = payload.get("upper_color")
    lower_color = payload.get("lower_color")

    for color in color_keywords:
        if color in query_lower:
            qcolor = "gray" if color == "grey" else color
            matched = (upper_color and (str(upper_color).lower() == qcolor)) or (lower_color and (str(lower_color).lower() == qcolor))
            if matched:
                boosted_score += 0.14
            else:
                boosted_score -= 0.03
            break

    clothing_keywords = ["shirt", "pants", "jacket", "dress", "skirt", "sweater", "coat", "jeans", "hat", "hood"]
    if any(keyword in query_lower for keyword in clothing_keywords):
        if upper_color or lower_color:
            boosted_score += 0.05

    attributes = payload.get("attributes", {})

    def keyword_hit(words):
        return any(w in query_lower for w in words)

    if keyword_hit(["hat", "cap", "beanie", "wearing a hat", "wearing a cap", "with hat", "with cap", "has hat", "has cap"]):
        if attributes.get("has_hat"):
            boosted_score += 0.25
        else:
            boosted_score -= 0.04

    if keyword_hit(["hood", "hooded", "wearing hood", "has hood", "with hood"]):
        if attributes.get("has_hood"):
            boosted_score += 0.22
        else:
            uc = str(upper_color or "").lower()
            if (uc in ["gray", "grey", "white"]) and not attributes.get("has_hat"):
                boosted_score += 0.08
            else:
                boosted_score -= 0.03

    if keyword_hit(["glasses", "sunglasses", "wearing glasses", "has glasses", "with glasses"]):
        if attributes.get("has_glasses"):
            boosted_score += 0.18
        else:
            boosted_score -= 0.02

    if "verified" in query_lower and verified:
        boosted_score += 0.03

    return boosted_score


def _semantic_scroll_persons(client, text_embedding, video_id, text_query, top_k=10, limit=1000):
    if client is None:
        raise Exception("Qdrant client not initialized")

    scroll_filter = _build_video_filter(video_id)
    points, _ = client.scroll(
        collection_name='person_tracks',
        limit=limit,
        with_payload=True,
        with_vectors=True,
        scroll_filter=scroll_filter
    )

    if not points:
        return []

    track_best = {}
    for point in points:
        if not hasattr(point, "vector") or point.vector is None:
            continue

        vec = None
        if isinstance(point.vector, dict):
            vec = point.vector.get('multi_vec') or point.vector.get('face_vec') or point.vector.get('reid_vec')
        else:
            vec = point.vector

        if vec is None:
            continue

        vec_np = np.array(vec, dtype=np.float32)
        if vec_np.size == 0 or vec_np.shape[0] != text_embedding.shape[0]:
            continue

        sim = np.dot(text_embedding, vec_np) / (np.linalg.norm(text_embedding) * np.linalg.norm(vec_np) + 1e-8)
        payload = point.payload if hasattr(point, "payload") else {}
        boosted = _apply_person_query_boosts(text_query, payload, float(sim))

        track_id = payload.get("track_id")
        if track_id is None:
            track_id = str(point.id)

        if track_id not in track_best or boosted > track_best[track_id]["score"]:
            track_best[track_id] = {
                "id": point.id,
                "score": boosted,
                "payload": payload,
            }

    results = list(track_best.values())
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


def _format_person_result(result, index):
    payload = result["payload"]
    normalized = (result['score'] + 1.0) / 2.0
    return {
        "id": str(result["id"]),
        "trackId": payload.get("track_id"),
        "personId": f"Person-{str(payload.get('track_id', index + 1)).zfill(3)}",
        "score": normalized,
        "confidence": f"{(normalized * 100):.1f}%",
        "timeOfAppearance": payload.get("start_time", "N/A"),
        "endTime": payload.get("end_time", "N/A"),
        "clothingColors": {
            "upper": payload.get("upper_color", "Unknown"),
            "lower": payload.get("lower_color", "Unknown")
        },
        "objectCarried": ", ".join(payload.get("object_carried", [])) if isinstance(payload.get("object_carried"), list) else payload.get("object_carried", "None"),
        "numFrames": payload.get("num_frames", 0),
        "verified": payload.get("verified", False),
        "attributes": payload.get("attributes", {})
    }


def _format_object_result(result, index):
    payload = result["payload"]
    return {
        "id": str(result["id"]),
        "trackId": payload.get("track_id"),
        "objectId": f"Object-{str(payload.get('track_id', index + 1)).zfill(3)}",
        "objectName": payload.get("object_type", payload.get("object_class", "Unknown")),
        "score": result["score"],
        "confidence": f"{(result['score'] * 100):.1f}%",
        "timeOfAppearance": payload.get("start_time", payload.get("first_appearance_time", "N/A")),
        "endTime": payload.get("end_time", payload.get("last_appearance_time", "N/A")),
        "color": payload.get("object_color", "Unknown"),
        "numFrames": payload.get("num_frames", 0)
    }


def handle_text_query(request, client, clip_model, clip_module, device):
    data = request.get_json(silent=True) or {}
    text_query = data.get('text')
    video_id = data.get('video_id')
    top_k = int(data.get('top_k', 10))

    if not text_query:
        return {
            'success': False,
            'error': 'Missing text parameter'
        }, 400

    if not video_id:
        return {
            'success': False,
            'error': 'Missing video_id parameter'
        }, 400

    text_embedding = extract_text_embedding(clip_model, clip_module, device, text_query)

    person_results = _semantic_scroll_persons(
        client=client,
        text_embedding=text_embedding,
        video_id=video_id,
        text_query=text_query,
        top_k=top_k,
        limit=1000
    )

    object_results = _semantic_scroll(
        client=client,
        collection_name='object_tracks',
        text_embedding=text_embedding,
        vector_keys=['clip_vec', 'multi_vec'],
        video_id=video_id,
        top_k=top_k,
        limit=1000
    )

    formatted_persons = [_format_person_result(result, i) for i, result in enumerate(person_results)]
    formatted_objects = [_format_object_result(result, i) for i, result in enumerate(object_results)]

    return {
        'success': True,
        'query': text_query,
        'videoId': int(video_id),
        'results': {
            'persons': formatted_persons,
            'objects': formatted_objects
        },
        'summary': {
            'totalPersonsFound': len(formatted_persons),
            'totalObjectsFound': len(formatted_objects)
        }
    }, 200
