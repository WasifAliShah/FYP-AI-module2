# Person Re-Identification Integration Guide

## Overview

This guide explains how to integrate the new person re-identification system with your existing pipeline.

## Key Concepts

### 1. Immutable Tracklets
- Each tracklet is stored ONCE in PostgreSQL `person_tracklets` table
- Tracklets are never updated - they represent immutable observations
- Multiple tracklets can map to a single person via `person_tracklet_associations`

### 2. Person Identity
- Stored in PostgreSQL `persons` table
- Represents a hypothesis: "these tracklets belong to the same human"
- Created automatically when a new person is first seen
- Updated in real-time as new tracklets are matched

### 3. Resolution Modes
- **automatic**: ReID similarity ≥ T_HIGH_REID (0.75)
- **hybrid**: ReID ≥ T_MED_REID (0.60) AND Face ≥ T_MED_FACE (0.55)
- No manual review mode (for FYP scope)

## Similarity Thresholds

```python
# ReID + Face Matching Thresholds
T_HIGH_REID = 0.75      # Very high confidence ReID match
T_HIGH_FACE = 0.70      # Very high confidence face match
FUSED_HIGH = 0.72       # Auto-assign person

T_MED_REID = 0.60       # Medium confidence ReID
T_MED_FACE = 0.55       # Medium confidence face
FUSED_MED = 0.62        # Hybrid resolution

T_LOW_REID = 0.50       # Below: consider new person
T_LOW_FACE = 0.45
FUSED_LOW = 0.50

# Quality requirements
MIN_FACE_QUALITY = 0.5  # Face must be at least 50% confidence
MIN_REID_QUALITY = 0.6  # ReID must be at least 60% confidence
```

## Integration Workflow

### Step 1: After Tracklet Extraction
```python
# In cursor_muk_deepsort.py, after tracklet finishes
tracklet_meta = TrackletMetadata(
    qdrant_tracklet_id=tracklet.qdrant_id,
    track_number=tracklet.id,
    video_id=VIDEO_ID,
    start_time=tracklet.start_time_str,
    end_time=tracklet.end_time_str,
    num_frames=tracklet.num_frames,
    face_quality=tracklet.avg_face_quality(),
    reid_quality=tracklet.avg_reid_quality(),
    face_embedding=tracklet.avg_face_emb(),
    reid_embedding=tracklet.avg_reid_emb(),
    attributes=tracklet.get_attributes()
)
```

### Step 2: Search for Similar Tracklets
```python
# Find candidates in same video
candidates = search_similar_tracklets_in_video(
    pg_conn, 
    VIDEO_ID, 
    tracklet_meta,
    limit=10
)
```

### Step 3: Compare and Match
```python
# Find best matching person (if any)
best_match = find_best_person_match(
    pg_conn,
    tracklet_meta,
    candidates,
    tracklet.avg_reid_emb(),  # Current tracklet ReID vector
    tracklet.avg_face_emb()   # Current tracklet face vector
)
```

### Step 4: Store in PostgreSQL
```python
# Insert tracklet and assign to person (new or existing)
tracklet_id, person_id = insert_person_tracklet_to_postgresql(
    pg_conn,
    tracklet_meta,
    similarity_match=best_match
)

print(f"Person {person_id}: Tracklet {tracklet.id} → {tracklet_id}")
```

### Step 5: Optional - Update Qdrant
```python
# Update Qdrant tracklet payload with resolved person_id
qdrant_point = {
    "id": tracklet.qdrant_id,
    "payload": {
        ...existing_payload...,
        "resolved_person_id": str(person_id)  # Link back to person
    },
    "vector": {...existing_vectors...}
}
client.upsert(collection_name="person_tracklets", points=[qdrant_point])
```

## Querying Results

### Get all tracklets for a person
```python
cursor.execute("""
    SELECT tracklet_id, start_time, end_time, video_id
    FROM person_tracklets
    WHERE person_id = %s
    ORDER BY start_time ASC
""", (person_id,))
```

### Get appearance timeline with gaps
```python
timeline = get_person_appearance_timeline(pg_conn, person_id)
# Output:
# [
#   {"appearance_number": 1, "start_time": "00:00:10.000", "end_time": "00:00:45.500", "duration_sec": 35.5},
#   {"appearance_number": 2, "start_time": "00:01:30.000", "end_time": "00:02:15.000", "gap_from_previous_sec": 44.5, "duration_sec": 45.0}
# ]
```

### Get video statistics
```python
stats = get_video_person_statistics(pg_conn, video_id=1)
# Output:
# {
#   "video_id": 1,
#   "total_persons": 5,
#   "persons_with_reappearances": 2,
#   "avg_tracklets_per_person": 3.2,
#   "timestamp": "2026-01-05T14:30:00"
# }
```

## Qdrant Schema - No Changes Needed

Person tracklets in Qdrant remain immutable:
```python
{
  "id": "uuid",
  "payload": {
    "track_id": 37,
    "video_id": 1,
    "start_time": "00:01:12.500",
    "end_time": "00:01:19.200",
    "num_frames": 214,
    "face_quality": 0.81,
    "reid_quality": 0.74,
    "resolved_person_id": "uuid"  # NEW: Link to PostgreSQL person
  },
  "vector": {
    "face_vec": [...],
    "reid_vec": [...]
  }
}
```

## Decision Tree

```
New Tracklet Arrives
    ↓
Query candidates from same video
    ↓
For each candidate:
    1. Compare ReID vectors → reid_sim
    2. Compare face vectors → face_sim
    3. Compute fused = 0.65*reid + 0.35*face
    ↓
Best match found?
    ├─ YES:
    │   ├─ reid_sim ≥ 0.75 → AUTOMATIC MATCH
    │   ├─ reid_sim ≥ 0.60 AND face_sim ≥ 0.55 → HYBRID MATCH
    │   └─ else → NO MATCH
    │
    └─ NO:
        ↓
        CREATE NEW PERSON
        ↓
        Insert tracklet + person + association
```

## Database Relationships

```
persons (1)
    ↓
    ├─ has many ↓
    person_tracklets (N)
    
person_tracklet_associations
    ↓
    maps tracklets → persons with similarity scores
```

## Important Notes

1. **PostgreSQL is authoritative** for person identity
   - Qdrant is read-only observation store
   - Person IDs are generated/resolved in PostgreSQL
   - Qdrant tracklets reference PostgreSQL via `resolved_person_id`

2. **Tracklets are immutable**
   - Never update a tracklet record
   - If you need to change person assignment, create new association (audit trail)

3. **ReID is primary, face is confirmation**
   - Always check ReID first
   - Face is used only to confirm borderline ReID matches
   - Weights: 65% ReID, 35% Face

4. **Quality matters**
   - Only match if embeddings meet quality thresholds
   - Face quality ≥ 0.5, ReID quality ≥ 0.6
   - Poor quality tracklets create new persons

5. **Same video scope**
   - Searches are limited to same video_id
   - Cross-video matching requires different query
   - Stats can show cross-video appearances if needed

## Troubleshooting

### Issue: All tracklets becoming new persons
**Cause**: Embedding quality too low or thresholds too high
**Fix**: Check MIN_FACE_QUALITY and MIN_REID_QUALITY values

### Issue: Too many false matches
**Cause**: Thresholds too low
**Fix**: Increase T_HIGH_REID and FUSED_HIGH

### Issue: Missing reappearances
**Cause**: Large time gaps confusing matching
**Fix**: Check `person_appearance_timeline` table for detection

## Next Steps (Future Enhancements)

1. Cross-video person matching (search across videos)
2. Manual review interface for borderline cases (T_MED range)
3. Clustering algorithm for pre-grouping similar tracklets
4. Face verification service integration
5. Analytics dashboard showing person statistics
