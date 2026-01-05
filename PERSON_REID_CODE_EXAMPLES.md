# Person Re-ID: Code Integration Examples

Quick copy-paste examples for integrating person re-identification into your pipeline.

## 1. Connection Setup

```python
import psycopg2
from person_reid_manager import *

# Initialize PostgreSQL connection
pg_conn = psycopg2.connect(
    host=os.environ.get("PG_HOST", "localhost"),
    port=int(os.environ.get("PG_PORT", 5432)),
    database=os.environ.get("PG_DATABASE", "visionindex"),
    user=os.environ.get("PG_USER", "postgres"),
    password=os.environ.get("PG_PASSWORD")
)

print("✓ PostgreSQL connected for person re-ID")
```

## 2. After Tracklet Finishes (Real-Time Resolution)

Add this in your main loop, right after a tracklet completes tracking:

```python
# In main loop, after tracklet finishes and embeddings are averaged
for tid, tracklet in list(tracklets.items()):
    if tracklet.last_frame == frame_idx:  # Tracklet just finished
        
        # ===== PERSON RE-ID RESOLUTION =====
        print(f"\n🔄 Resolving person identity for tracklet {tid}...")
        
        # 1. Create metadata
        tracklet_meta = TrackletMetadata(
            qdrant_tracklet_id=str(uuid.uuid4()),  # Or from Qdrant insert
            track_number=tid,
            video_id=VIDEO_ID,
            start_time=tracklet.start_time_str,
            end_time=tracklet.end_time_str,
            num_frames=tracklet.num_frames,
            face_quality=tracklet.avg_face_size(),  # Or proper quality metric
            reid_quality=0.75,  # Compute from model confidence
            face_embedding=tracklet.avg_face(),
            reid_embedding=tracklet.avg_reid(),
            attributes=tracklet.get_attribute_summary()
        )
        
        # 2. Search for similar tracklets in same video
        candidates = search_similar_tracklets_in_video(
            pg_conn,
            VIDEO_ID,
            tracklet_meta,
            limit=10
        )
        print(f"   Found {len(candidates)} candidate tracklets to compare")
        
        # 3. Find best person match
        if candidates:
            best_match = find_best_person_match(
                pg_conn,
                tracklet_meta,
                candidates,
                tracklet.avg_reid(),
                tracklet.avg_face()
            )
            
            if best_match:
                print(f"   ✅ Match found: Person {best_match.person_id}")
                print(f"      ReID: {best_match.reid_similarity:.3f}")
                print(f"      Face: {best_match.face_similarity:.3f}")
                print(f"      Fused: {best_match.fused_similarity:.3f}")
                print(f"      Mode: {best_match.resolution_mode}")
            else:
                print(f"   ✨ No match - creating new person")
        else:
            print(f"   ✨ First person in video - creating new person")
        
        # 4. Insert into PostgreSQL
        try:
            tracklet_id, person_id = insert_person_tracklet_to_postgresql(
                pg_conn,
                tracklet_meta,
                similarity_match=best_match if candidates else None
            )
            print(f"   ✓ Stored: Tracklet {tracklet_id} → Person {person_id}")
            
        except Exception as e:
            print(f"   ❌ Error storing tracklet: {e}")
            pg_conn.rollback()
        
        # 5. Update Qdrant with resolved person ID (optional)
        if REALTIME_FACE_COMPARISON:
            try:
                qdrant_point = PointStruct(
                    id=tracklet_meta.qdrant_tracklet_id,
                    vector={...existing_vectors...},
                    payload={
                        **existing_payload,
                        "resolved_person_id": str(person_id)
                    }
                )
                client.upsert(
                    collection_name="person_tracklets",
                    points=[qdrant_point]
                )
            except Exception as e:
                print(f"   ⚠ Could not update Qdrant: {e}")
```

## 3. Batch Resolution (After Video Processing)

If you want post-processing resolution instead of real-time:

```python
def resolve_all_persons_after_processing(pg_conn, video_id):
    """
    Batch process all tracklets to resolve persons.
    Called after all tracklets from video are extracted.
    """
    print(f"\n🔄 Batch person resolution for video {video_id}...")
    
    try:
        cursor = pg_conn.cursor()
        
        # Get all unresolved tracklets
        cursor.execute("""
            SELECT tracklet_id, video_id, qdrant_tracklet_id, track_number,
                   start_time, end_time, num_frames, face_quality, reid_quality,
                   attributes
            FROM person_tracklets
            WHERE video_id = %s
            AND person_id IS NULL
            ORDER BY start_time ASC
        """, (video_id,))
        
        unresolved = cursor.fetchall()
        print(f"   Found {len(unresolved)} unresolved tracklets")
        
        for idx, tracklet_row in enumerate(unresolved):
            tid, vid, qid, track_num, start, end, nframes, fq, rq, attrs = tracklet_row
            
            print(f"   [{idx+1}/{len(unresolved)}] Processing tracklet {tid}...")
            
            # Reconstruct metadata
            tracklet_meta = TrackletMetadata(
                qdrant_tracklet_id=qid,
                track_number=track_num,
                video_id=vid,
                start_time=str(start),
                end_time=str(end),
                num_frames=nframes,
                face_quality=fq,
                reid_quality=rq,
                attributes=json.loads(attrs) if attrs else {}
            )
            
            # Note: Can't retrieve embeddings in post-processing
            # Would need to store them or retrieve from Qdrant
            
            # Search and match
            candidates = search_similar_tracklets_in_video(
                pg_conn, vid, tracklet_meta, limit=10
            )
            
            if candidates:
                best_match = find_best_person_match(
                    pg_conn, tracklet_meta, candidates,
                    None, None  # No embeddings in batch mode
                )
            else:
                best_match = None
            
            # Assign person
            cursor.execute("""
                UPDATE person_tracklets
                SET person_id = %s, updated_at = NOW()
                WHERE tracklet_id = %s
            """, (
                str(best_match.person_id) if best_match and best_match.person_id else str(uuid.uuid4()),
                str(tid)
            ))
            
            pg_conn.commit()
    
    except Exception as e:
        print(f"   ❌ Error in batch resolution: {e}")
        pg_conn.rollback()

# Call after video processing complete
resolve_all_persons_after_processing(pg_conn, VIDEO_ID)
```

## 4. Query: Get Person Timeline

```python
def display_person_timeline(pg_conn, person_id):
    """Display when a person appeared in video"""
    
    timeline = get_person_appearance_timeline(pg_conn, person_id)
    
    print(f"\n📅 Person {person_id} Appearance Timeline:")
    for appearance in timeline:
        print(f"\n   Appearance #{appearance['appearance_number']}")
        print(f"   Time: {appearance['start_time']} → {appearance['end_time']}")
        print(f"   Duration: {appearance['duration_sec']:.1f}s")
        if 'gap_from_previous_sec' in appearance:
            print(f"   Gap from previous: {appearance['gap_from_previous_sec']:.1f}s")

# Usage
for person_id in [uuid.UUID("..."), uuid.UUID("..."), ...]:
    display_person_timeline(pg_conn, person_id)
```

## 5. Query: Video Statistics

```python
def print_video_statistics(pg_conn, video_id):
    """Print summary statistics for persons in video"""
    
    stats = get_video_person_statistics(pg_conn, video_id)
    
    print(f"\n📊 Person Statistics for Video {video_id}:")
    print(f"   Total persons detected: {stats['total_persons']}")
    print(f"   Persons with reappearances: {stats['persons_with_reappearances']}")
    print(f"   Avg tracklets per person: {stats['avg_tracklets_per_person']:.1f}")
    print(f"   Timestamp: {stats['timestamp']}")

# Usage
print_video_statistics(pg_conn, VIDEO_ID)
```

## 6. Query: Export Person Data (for debugging)

```python
def export_person_matches(pg_conn, video_id):
    """Export all person-tracklet mappings with similarity scores"""
    
    cursor = pg_conn.cursor()
    cursor.execute("""
        SELECT 
            ptp.person_id,
            COUNT(ptp.tracklet_id) as tracklet_count,
            STRING_AGG(
                format('%s (reid: %.3f, face: %.3f)', 
                    ptp.track_number,
                    ppta.reid_similarity,
                    ppta.face_similarity
                ), 
                E'\n'
            ) as tracklet_details
        FROM person_tracklets ptp
        LEFT JOIN person_tracklet_associations ppta 
            ON ptp.tracklet_id = ppta.tracklet_id
        WHERE ptp.video_id = %s
        GROUP BY ptp.person_id
        ORDER BY tracklet_count DESC
    """, (video_id,))
    
    results = cursor.fetchall()
    
    for person_id, count, details in results:
        print(f"\nPerson {person_id} ({count} tracklets):")
        print(details)

# Usage
export_person_matches(pg_conn, VIDEO_ID)
```

## 7. Environment Setup

Add to `.env`:

```
# PostgreSQL for person re-ID
PG_HOST=localhost
PG_PORT=5432
PG_DATABASE=visionindex
PG_USER=postgres
PG_PASSWORD=your_password

# Person Re-ID thresholds
REID_WEIGHT=0.65
FACE_WEIGHT=0.35

# High confidence thresholds
T_HIGH_REID=0.75
T_HIGH_FACE=0.70
FUSED_HIGH=0.72

# Medium confidence thresholds
T_MED_REID=0.60
T_MED_FACE=0.55
FUSED_MED=0.62

# Quality requirements
MIN_FACE_QUALITY=0.5
MIN_REID_QUALITY=0.6
```

## 8. Error Handling Template

```python
def safe_person_resolution(pg_conn, tracklet_meta, candidates):
    """Wrapped person resolution with error handling"""
    
    try:
        # Find match
        best_match = None
        if candidates:
            best_match = find_best_person_match(
                pg_conn, tracklet_meta, candidates
            )
        
        # Insert tracklet
        tracklet_id, person_id = insert_person_tracklet_to_postgresql(
            pg_conn, tracklet_meta, best_match
        )
        
        return tracklet_id, person_id, "success"
    
    except ValueError as e:
        print(f"⚠ Validation error: {e}")
        return None, None, "validation_error"
    
    except psycopg2.DatabaseError as e:
        print(f"❌ Database error: {e}")
        pg_conn.rollback()
        return None, None, "database_error"
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        pg_conn.rollback()
        return None, None, "unknown_error"

# Usage
tid, pid, status = safe_person_resolution(pg_conn, tracklet_meta, candidates)
if status == "success":
    print(f"✓ Person {pid} resolved successfully")
else:
    print(f"✗ Resolution failed: {status}")
```

## 9. Cleanup/Debugging Queries

```python
# Find unresolved tracklets
cursor.execute("""
    SELECT COUNT(*) FROM person_tracklets WHERE person_id IS NULL
""")
print(f"Unresolved tracklets: {cursor.fetchone()[0]}")

# Find persons with only 1 tracklet
cursor.execute("""
    SELECT person_id, COUNT(*) as cnt
    FROM person_tracklets
    WHERE video_id = %s
    GROUP BY person_id
    HAVING COUNT(*) = 1
""", (VIDEO_ID,))
print(f"Single-tracklet persons: {len(cursor.fetchall())}")

# Find highest confidence matches
cursor.execute("""
    SELECT person_id, tracklet_id, fused_similarity
    FROM person_tracklet_associations
    WHERE fused_similarity >= 0.85
    ORDER BY fused_similarity DESC
    LIMIT 10
""")
for p, t, f in cursor.fetchall():
    print(f"Person {p}, Tracklet {t}: {f:.4f}")
```

## 10. Transaction Example

```python
def atomic_person_resolution(pg_conn, tracklets_batch):
    """Resolve multiple tracklets in one transaction"""
    
    try:
        cursor = pg_conn.cursor()
        
        for tracklet in tracklets_batch:
            # Do resolution...
            candidate = search_similar_tracklets_in_video(...)
            best_match = find_best_person_match(...)
            
            # Insert
            cursor.execute("""
                INSERT INTO person_tracklets (...)
                VALUES (...)
            """)
        
        pg_conn.commit()
        print(f"✓ Committed {len(tracklets_batch)} tracklets")
    
    except Exception as e:
        pg_conn.rollback()
        print(f"❌ Transaction failed, rolled back: {e}")
        raise
```

---

## Quick Reference

| Function | Purpose | Returns |
|----------|---------|---------|
| `search_similar_tracklets_in_video()` | Find candidates | List of tuples |
| `find_best_person_match()` | Compare & match | `SimilarityMatch` or None |
| `insert_person_tracklet_to_postgresql()` | Store tracklet | `(tracklet_id, person_id)` |
| `get_person_appearance_timeline()` | Get appearance blocks | List of dicts |
| `get_video_person_statistics()` | Get video summary | Dict of stats |
| `cosine_similarity()` | Compare vectors | Float 0-1 |
| `fuse_similarities()` | Combine ReID + face | Float 0-1 |

All imports:
```python
from person_reid_manager import (
    TrackletMetadata,
    SimilarityMatch,
    search_similar_tracklets_in_video,
    find_best_person_match,
    insert_person_tracklet_to_postgresql,
    get_person_appearance_timeline,
    get_video_person_statistics,
    cosine_similarity,
    fuse_similarities,
    should_match_person,
    determine_confidence_level
)
```
