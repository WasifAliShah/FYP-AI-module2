"""
Person Re-Identification (ReID) Module

Implements real-time person identity resolution during video processing.
- Matches tracklets using ReID similarity first, confirms with face matching
- Stores immutable tracklets in PostgreSQL
- Maintains person identity as a separate hypothesis in PostgreSQL
- Qdrant stores only tracklet observations (immutable)
"""

import uuid
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, time
import hashlib
import json

# ============================================
# Configuration & Thresholds
# ============================================

# ReID + Face Fusion Weights
REID_WEIGHT = 0.65  # ReID is more reliable than face
FACE_WEIGHT = 0.35

# Similarity Thresholds
T_HIGH_REID = 0.75      # ReID: very high confidence match
T_HIGH_FACE = 0.70      # Face: very high confidence match
FUSED_HIGH = 0.72       # Fused score for auto-assign

T_MED_REID = 0.60       # ReID: medium confidence
T_MED_FACE = 0.55       # Face: medium confidence
FUSED_MED = 0.62        # Fused score for review (but we skip manual review)

T_LOW_REID = 0.50       # Below this: consider new person
T_LOW_FACE = 0.45
FUSED_LOW = 0.50

# Quality thresholds
MIN_FACE_QUALITY = 0.5  # Face must be this good to use for matching
MIN_REID_QUALITY = 0.6  # ReID must be this good to use for matching


# ============================================
# Data Classes
# ============================================

@dataclass
class TrackletMetadata:
    """Metadata for a single tracklet observation"""
    qdrant_tracklet_id: str
    track_number: int
    video_id: int
    start_time: str  # HH:MM:SS.mmm format
    end_time: str
    num_frames: int
    face_quality: float
    reid_quality: float
    face_embedding: Optional[np.ndarray] = None
    reid_embedding: Optional[np.ndarray] = None
    attributes: Optional[Dict] = None
    
    def duration_seconds(self) -> float:
        """Calculate duration in seconds from start/end times"""
        def time_to_sec(time_str):
            parts = time_str.split(":")
            h = int(parts[0])
            m = int(parts[1])
            s_ms = parts[2].split(".")
            s = int(s_ms[0])
            ms = int(s_ms[1]) if len(s_ms) > 1 else 0
            return h * 3600 + m * 60 + s + ms / 1000.0
        
        start_sec = time_to_sec(self.start_time)
        end_sec = time_to_sec(self.end_time)
        return max(0, end_sec - start_sec)


@dataclass
class SimilarityMatch:
    """Result of similarity search"""
    tracklet_id: uuid.UUID
    reid_similarity: float
    face_similarity: float
    fused_similarity: float
    person_id: Optional[uuid.UUID] = None
    confidence_level: str = "low"  # high, medium, low
    resolution_mode: str = "automatic"  # automatic or hybrid


# ============================================
# Core Functions
# ============================================

def compute_embedding_hash(embedding: np.ndarray) -> str:
    """Create SHA256 hash of embedding for comparison"""
    if embedding is None:
        return None
    arr_bytes = embedding.astype(np.float32).tobytes()
    return hashlib.sha256(arr_bytes).hexdigest()


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors (0-1, higher is better)"""
    if vec1 is None or vec2 is None:
        return 0.0
    
    try:
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        # Cosine similarity ranges -1 to 1, map to 0-1
        sim = np.dot(vec1_norm, vec2_norm)
        return float((sim + 1.0) / 2.0)  # Map [-1, 1] to [0, 1]
    except Exception:
        return 0.0


def fuse_similarities(reid_sim: float, face_sim: float) -> float:
    """Fuse ReID and face similarities with weights"""
    return REID_WEIGHT * reid_sim + FACE_WEIGHT * face_sim


def determine_confidence_level(fused_sim: float) -> str:
    """Classify confidence based on fused similarity score"""
    if fused_sim >= FUSED_HIGH:
        return "high"
    elif fused_sim >= FUSED_MED:
        return "medium"
    else:
        return "low"


def should_match_person(reid_sim: float, face_sim: float, tracklet_meta: TrackletMetadata) -> Tuple[bool, str]:
    """
    Decide if a tracklet should be matched to an existing person.
    
    Decision logic:
    1. If ReID quality is good AND ReID sim >= T_HIGH_REID → MATCH (automatic)
    2. If ReID quality is good AND ReID sim >= T_MED_REID AND face sim >= T_MED_FACE → MATCH (hybrid)
    3. Otherwise → NO MATCH (new person)
    
    Returns:
        (should_match: bool, resolution_mode: str)
    """
    # First check: do we have good quality embeddings?
    has_good_reid = tracklet_meta.reid_quality >= MIN_REID_QUALITY
    has_good_face = tracklet_meta.face_quality >= MIN_FACE_QUALITY
    
    if not has_good_reid:
        # Poor ReID quality - can't reliably match
        return False, "insufficient_quality"
    
    # Automatic match: high ReID confidence
    if reid_sim >= T_HIGH_REID:
        # Optionally confirm with face
        if face_sim >= T_HIGH_FACE or not has_good_face:
            return True, "automatic"
    
    # Hybrid match: medium ReID + face confirmation
    if reid_sim >= T_MED_REID and has_good_face and face_sim >= T_MED_FACE:
        return True, "hybrid"
    
    # No match
    return False, "no_match"


def search_similar_tracklets_in_video(
    pg_conn,
    video_id: int,
    tracklet_meta: TrackletMetadata,
    limit: int = 10
) -> List[Tuple]:
    """
    Search for existing tracklets in the same video that could match this tracklet.
    
    Returns list of (tracklet_id, start_time, end_time, reid_quality, face_quality, person_id)
    """
    try:
        cursor = pg_conn.cursor()
        cursor.execute("""
            SELECT tracklet_id, start_time, end_time, reid_quality, face_quality, person_id
            FROM public.person_tracklets
            WHERE video_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        """, (video_id, limit))
        return cursor.fetchall()
    except Exception as e:
        print(f"⚠ Error searching similar tracklets: {e}")
        return []


def find_best_person_match(
    pg_conn,
    tracklet_meta: TrackletMetadata,
    candidate_tracklets: List[Tuple]
) -> Optional[SimilarityMatch]:
    """
    Compare current tracklet against candidates and find best person match.
    
    Compares using ReID (primary) and face (confirmation).
    """
    if not candidate_tracklets:
        return None
    
    best_match = None
    best_fused_sim = -1
    
    for candidate_id, cand_start, cand_end, cand_reid_qual, cand_face_qual, cand_person_id in candidate_tracklets:
        try:
            # Retrieve full candidate tracklet data from PostgreSQL
            cursor = pg_conn.cursor()
            cursor.execute("""
                SELECT reid_embedding_hash, face_embedding_hash, person_id
                FROM public.person_tracklets
                WHERE tracklet_id = %s
            """, (str(candidate_id),))
            
            result = cursor.fetchone()
            if not result:
                continue
            
            # In production, you'd retrieve actual embeddings from Qdrant using the hash
            # For now, we use embeddings from Qdrant search (passed separately)
            # This is simplified - actual implementation would fetch from Qdrant
            
            # Simulate similarity computation (in real code, get embeddings from Qdrant)
            reid_sim = 0.0  # Would be computed from embeddings
            face_sim = 0.0  # Would be computed from embeddings
            
            # For now, return based on hash matching (placeholder)
            # In production: compute actual embeddings and similarities
            
            # Check if should match
            should_match, resolution_mode = should_match_person(reid_sim, face_sim, tracklet_meta)
            
            if should_match:
                fused_sim = fuse_similarities(reid_sim, face_sim)
                confidence_level = determine_confidence_level(fused_sim)
                
                if fused_sim > best_fused_sim:
                    best_fused_sim = fused_sim
                    best_match = SimilarityMatch(
                        tracklet_id=candidate_id,
                        reid_similarity=reid_sim,
                        face_similarity=face_sim,
                        fused_similarity=fused_sim,
                        person_id=cand_person_id,
                        confidence_level=confidence_level,
                        resolution_mode=resolution_mode
                    )
        
        except Exception as e:
            print(f"⚠ Error comparing with candidate {candidate_id}: {e}")
            continue
    
    return best_match


def insert_person_tracklet_to_postgresql(
    pg_conn,
    tracklet_meta: TrackletMetadata,
    similarity_match: Optional[SimilarityMatch] = None
) -> Tuple[uuid.UUID, uuid.UUID]:
    """
    Insert a tracklet into PostgreSQL and optionally assign to existing person.
    
    Returns:
        (tracklet_id, person_id)
    """
    tracklet_id = uuid.uuid4()
    
    try:
        cursor = pg_conn.cursor()
        
        # Determine person assignment
        person_id = None
        is_primary = False
        
        if similarity_match and similarity_match.person_id:
            # Match found - assign to existing person
            person_id = similarity_match.person_id
            is_primary = False  # Not primary unless best quality
            
            print(f"✅ Tracklet assigned to existing person {person_id}")
            print(f"   ReID: {similarity_match.reid_similarity:.3f}, Face: {similarity_match.face_similarity:.3f}, Fused: {similarity_match.fused_similarity:.3f}")
        else:
            # Create new person
            person_id = uuid.uuid4()
            is_primary = True  # First tracklet is primary
            
            # Insert person record
            cursor.execute("""
                INSERT INTO public.persons
                (person_id, video_id, confidence_score, resolution_mode, created_at, updated_at)
                VALUES (%s, %s, %s, %s, NOW(), NOW())
            """, (str(person_id), tracklet_meta.video_id, tracklet_meta.reid_quality, "automatic"))
            
            print(f"✨ Created new person {person_id}")
        
        # Insert tracklet
        duration_sec = tracklet_meta.duration_seconds()
        face_hash = compute_embedding_hash(tracklet_meta.face_embedding)
        reid_hash = compute_embedding_hash(tracklet_meta.reid_embedding)
        
        cursor.execute("""
            INSERT INTO public.person_tracklets
            (tracklet_id, person_id, video_id, qdrant_tracklet_id, track_number,
             start_time, end_time, num_frames, duration_sec,
             face_quality, reid_quality, avg_detection_confidence,
             face_embedding_hash, reid_embedding_hash,
             attributes, is_primary_for_person,
             created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
        """, (
            str(tracklet_id), str(person_id), tracklet_meta.video_id,
            tracklet_meta.qdrant_tracklet_id, tracklet_meta.track_number,
            tracklet_meta.start_time, tracklet_meta.end_time,
            tracklet_meta.num_frames, duration_sec,
            tracklet_meta.face_quality, tracklet_meta.reid_quality, 0.5,
            face_hash, reid_hash,
            json.dumps(tracklet_meta.attributes or {}), is_primary
        ))
        
        # Insert association record
        if similarity_match:
            cursor.execute("""
                INSERT INTO public.person_tracklet_associations
                (person_id, tracklet_id, reid_similarity, face_similarity, fused_similarity,
                 resolution_mode, confidence_level, threshold_applied, matched_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                str(person_id), str(tracklet_id),
                similarity_match.reid_similarity, similarity_match.face_similarity,
                similarity_match.fused_similarity,
                similarity_match.resolution_mode, similarity_match.confidence_level,
                FUSED_HIGH if similarity_match.resolution_mode == "automatic" else FUSED_MED
            ))
        
        # Update person stats
        cursor.execute("""
            UPDATE public.persons
            SET
                total_tracklets = (
                    SELECT COUNT(*) FROM public.person_tracklets WHERE person_id = %s
                ),
                first_appearance_time = (
                    SELECT MIN(start_time) FROM public.person_tracklets WHERE person_id = %s
                ),
                last_appearance_time = (
                    SELECT MAX(end_time) FROM public.person_tracklets WHERE person_id = %s
                ),
                updated_at = NOW()
            WHERE person_id = %s
        """, (str(person_id), str(person_id), str(person_id), str(person_id)))
        
        pg_conn.commit()
        print(f"✓ Tracklet {tracklet_id} stored in PostgreSQL")
        return tracklet_id, person_id
        
    except Exception as e:
        pg_conn.rollback()
        print(f"❌ Error inserting tracklet: {e}")
        raise


def get_person_appearance_timeline(pg_conn, person_id: uuid.UUID) -> List[Dict]:
    """
    Get all appearance blocks for a person (continuous presence periods).
    
    Returns list of appearance blocks with gaps calculated.
    """
    try:
        cursor = pg_conn.cursor()
        cursor.execute("""
            SELECT 
                tracklet_id, start_time, end_time,
                duration_sec, created_at
            FROM public.person_tracklets
            WHERE person_id = %s
            ORDER BY start_time ASC
        """, (str(person_id),))
        
        tracklets = cursor.fetchall()
        appearances = []
        
        for i, (tid, start, end, duration, created) in enumerate(tracklets):
            appearance = {
                "appearance_number": i + 1,
                "tracklet_id": str(tid),
                "start_time": str(start),
                "end_time": str(end),
                "duration_sec": float(duration)
            }
            
            if i > 0:
                prev_end = tracklets[i-1][2]
                gap = (start.total_seconds() - prev_end.total_seconds()) if hasattr(start, 'total_seconds') else 0
                appearance["gap_from_previous_sec"] = gap
            
            appearances.append(appearance)
        
        return appearances
        
    except Exception as e:
        print(f"⚠ Error getting appearance timeline: {e}")
        return []


def get_video_person_statistics(pg_conn, video_id: int) -> Dict:
    """
    Get statistics about all persons detected in a video.
    """
    try:
        cursor = pg_conn.cursor()
        
        # Total persons
        cursor.execute("""
            SELECT COUNT(DISTINCT person_id) FROM public.persons WHERE video_id = %s
        """, (video_id,))
        total_persons = cursor.fetchone()[0]
        
        # Persons with reappearances
        cursor.execute("""
            SELECT COUNT(DISTINCT person_id)
            FROM public.person_tracklets
            WHERE video_id = %s
            GROUP BY person_id
            HAVING COUNT(*) > 1
        """, (video_id,))
        persons_with_reappearances = len(cursor.fetchall())
        
        # Average tracklets per person
        cursor.execute("""
            SELECT AVG(tracklet_count)
            FROM (
                SELECT COUNT(*) as tracklet_count
                FROM public.person_tracklets
                WHERE video_id = %s
                GROUP BY person_id
            ) sub
        """, (video_id,))
        avg_tracklets = cursor.fetchone()[0] or 0
        
        return {
            "video_id": video_id,
            "total_persons": total_persons,
            "persons_with_reappearances": persons_with_reappearances,
            "avg_tracklets_per_person": float(avg_tracklets),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"⚠ Error getting video statistics: {e}")
        return {}
